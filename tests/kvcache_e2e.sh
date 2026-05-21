#!/usr/bin/env bash
# tests/kvcache_e2e.sh — end-to-end smoke for the KV prefix cache wiring
# (Roadmap 2.1 follow-up / task #82).
#
# Strategy: write a small C harness that opens the engine, attaches an
# in-RAM-only cache, runs prefill+forward TWICE on the same prompt
# tokens, and verifies:
#   - run 1: cache miss + insert
#   - run 2: cache hit
#   - logits at the same token position match within tolerance
#
# Falls through cleanly with a "skip" message when the 0.8B model
# isn't available.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${QW36_TEST_MODEL:-/Users/bytedance/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf}"
if [ ! -f "$MODEL" ]; then
    echo "[kvcache-e2e] skip: model not found at $MODEL"
    exit 0
fi

# CPU build is the simplest target. Metal would exercise the
# device-buffer download/upload roundtrip and is the natural next
# integration test; v0 keeps the CPU path golden.
make -s cpu 1>/dev/null

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/main.c" <<'EOF'
#include "qw36.h"
#include "qw36_gpu.h"
#include "qw36_gguf.h"
#include "qw36_tokenizer.h"
#include "qw36_kvcache.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

qw36_gpu_backend *qw36_backend_create(void);

static int run_once(qw36_engine *eng, const uint32_t *toks, size_t n_toks,
                    float *out_logits, uint32_t vocab) {
    qw36_state *st = qw36_state_new(eng, 64u);
    if (!st) return -1;
    int rc = qw36_prefill(eng, st, toks, n_toks - 1);
    if (rc) { qw36_state_free(st); return rc; }
    rc = qw36_forward(eng, st, toks[n_toks - 1]);
    if (rc) { qw36_state_free(st); return rc; }
    memcpy(out_logits, st->logits, vocab * sizeof(float));
    qw36_state_free(st);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]); return 2; }
    char err[256] = {0};
    qw36_gpu_backend *be = qw36_backend_create();
    qw36_engine *eng = qw36_engine_open(argv[1], be, err, sizeof(err));
    if (!eng) { fprintf(stderr, "engine open: %s\n", err); return 3; }
    qw36_tokenizer *tok = qw36_tokenizer_new(qw36_engine_gguf(eng), err, sizeof(err));
    if (!tok) { fprintf(stderr, "tokenizer: %s\n", err); return 4; }
    const qw36_config *c = qw36_engine_config(eng);
    uint32_t vocab = c->vocab_size;

    const char *prompt = "Once upon a time in the kingdom of bytes, "
                        "a small program decided to remember its past so "
                        "it would not have to compute everything anew.";
    uint32_t *ids = (uint32_t *)malloc(256 * sizeof(uint32_t));
    size_t n_ids = 256;   /* capacity in, count out */
    if (qw36_tokenizer_encode(tok, prompt, ids, &n_ids) || n_ids < 8) {
        fprintf(stderr, "[kvcache-e2e] tokenizer encode failed or too short (%zu)\n", n_ids);
        return 5;
    }
    fprintf(stderr, "[kvcache-e2e] prompt tokenized to %zu ids\n", n_ids);

    qw36_kv_cache_cfg cfg = { .min_prefix_tokens = 8, .writeback_on_evict = 0 };
    /* 0.8B with 24 layers + 28 tokens of host-fp32 KV cache can run into
     * the hundreds of MB. Cap at 1 GiB so we don't accidentally hit the
     * tier eviction path in this smoke. */
    qw36_kv_prefix_cache *cache = qw36_kv_cache_create_default(
        1ull * 1024ull * 1024ull * 1024ull, NULL, 0, &cfg);
    if (!cache) { fprintf(stderr, "cache create failed\n"); return 6; }
    qw36_engine_attach_kv_cache(eng, cache);

    float *logits1 = (float *)malloc(vocab * sizeof(float));
    float *logits2 = (float *)malloc(vocab * sizeof(float));
    if (!logits1 || !logits2) return 7;

    /* Surface the blob size up-front so the regression mode (too-small
     * cache cap → silent insert failure) is obvious. */
    {
        qw36_state *st = qw36_state_new(eng, 64u);
        if (st) {
            size_t bs = qw36_state_blob_size(st, eng);
            fprintf(stderr, "[kvcache-e2e] state_blob_size (seq_pos=0) = %zu bytes\n", bs);
            qw36_state_free(st);
        }
    }

    if (run_once(eng, ids, n_ids, logits1, vocab)) {
        fprintf(stderr, "[kvcache-e2e] run 1 failed\n"); return 8;
    }
    qw36_kv_cache_stats_t s1;
    qw36_kv_cache_stats(cache, &s1);
    fprintf(stderr, "[kvcache-e2e] after run 1: lookups=%llu misses=%llu "
                    "inserts=%llu hits[0]=%llu\n",
            (unsigned long long)s1.lookups,
            (unsigned long long)s1.misses,
            (unsigned long long)s1.inserts,
            (unsigned long long)s1.hits_by_tier[0]);

    if (run_once(eng, ids, n_ids, logits2, vocab)) {
        fprintf(stderr, "[kvcache-e2e] run 2 failed\n"); return 9;
    }
    qw36_kv_cache_stats_t s2;
    qw36_kv_cache_stats(cache, &s2);
    fprintf(stderr, "[kvcache-e2e] after run 2: lookups=%llu misses=%llu "
                    "inserts=%llu hits[0]=%llu\n",
            (unsigned long long)s2.lookups,
            (unsigned long long)s2.misses,
            (unsigned long long)s2.inserts,
            (unsigned long long)s2.hits_by_tier[0]);

    if (s2.hits_by_tier[0] <= s1.hits_by_tier[0]) {
        fprintf(stderr, "[kvcache-e2e] FAIL: run 2 produced no cache hit\n");
        return 10;
    }

    int mismatches = 0;
    float max_abs = 0.0f, max_rel = 0.0f;
    int worst_idx = -1;
    for (uint32_t i = 0; i < vocab; i++) {
        float a = logits1[i], b = logits2[i];
        float d = fabsf(a - b);
        float r = d / fmaxf(fabsf(a), 1e-6f);
        if (d > max_abs) { max_abs = d; worst_idx = (int)i; }
        if (r > max_rel) max_rel = r;
        if (d > 1e-3f && r > 1e-3f) mismatches++;
    }
    fprintf(stderr, "[kvcache-e2e] logits diff: max_abs=%.3e max_rel=%.3e "
                    "mismatches>1e-3 = %d / %u (worst idx %d)\n",
            (double)max_abs, (double)max_rel, mismatches, vocab, worst_idx);
    if (mismatches > 0) {
        fprintf(stderr, "[kvcache-e2e] FAIL: hydrated logits diverge from miss path\n");
        return 11;
    }

    qw36_kv_cache_destroy(cache);
    qw36_tokenizer_free(tok);
    qw36_engine_close(eng);
    free(logits1); free(logits2); free(ids);
    fprintf(stderr, "[kvcache-e2e] OK\n");
    return 0;
}
EOF

COMMON="common/qw36.c common/qw36_dequant.c common/qw36_policy.c \
        common/qw36_ops.c common/qw36_attn_vanilla.c \
        common/qw36_attn_deltanet.c common/qw36_mlp.c common/qw36_moe.c \
        common/qw36_kvcache.c common/qw36_gguf.c common/qw36_tokenizer.c"
cc -O2 -std=c11 -Wall -Wextra -Wno-unused-function -Icommon \
   "$TMP/main.c" $COMMON cpu/qw36_cpu_stub.c \
   -o "$TMP/kvcache_e2e" -lm

echo "[kvcache-e2e] running on $MODEL"
"$TMP/kvcache_e2e" "$MODEL"
