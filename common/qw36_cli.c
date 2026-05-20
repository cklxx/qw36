/* qw36_cli.c — main entry. Same source compiles into all binaries.
 *
 * Owner: Claude. CPU-only path works when qw36_backend_create() returns NULL.
 */

#include "qw36.h"
#include "qw36_gpu.h"
#include "qw36_gguf.h"
#include "qw36_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Provided by the backend object linked in (amd|metal|cuda|cpu). The CPU
 * build links cpu/qw36_cpu_stub.c, which just returns NULL → CLI falls
 * back to the reference forward path in qw36.c. */
qw36_gpu_backend *qw36_backend_create(void);

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s -m <model.gguf> [-p <prompt>] [-n <max_new_tokens>]\n"
        "          [-t <temperature>] [--top-p <p>] [--top-k <k>] [--seed <u64>]\n"
        "          [--seq <capacity>] [--interactive] [--no-special]\n"
        "          [--layer-trace <L> --layer-trace-out <path>] [--layer-trace-pos <pos>]\n"
        "          [--info]    print config + tokenizer summary and exit\n"
        "\n"
        "binary: %s\n", prog, qw36_version());
}

typedef struct {
    const char *model_path;
    const char *prompt;
    int   max_new;
    float temperature;
    float top_p;
    int   top_k;
    uint64_t seed;
    int   interactive;
    int   info_only;
    int   seq_capacity;
    int   no_special;
    int   debug_top;     /* print top-K logits per step */
    int   dump_tokens;   /* tokenize the prompt and exit */
    int   layer_trace;   /* vanilla layer index to dump, -1 disables */
    int   layer_trace_pos; /* optional seq_pos filter for trace */
    const char *layer_trace_out;
} qw36_cli_args;

static int parse_args(int argc, char **argv, qw36_cli_args *a) {
    a->max_new      = 128;
    a->temperature  = 0.0f;
    a->top_p        = 1.0f;
    a->top_k        = 0;
    a->seed         = 42;
    a->seq_capacity = 2048;
    a->layer_trace  = -1;
    a->layer_trace_pos = -1;
    for (int i = 1; i < argc; i++) {
        const char *s = argv[i];
        #define EAT() (i + 1 < argc ? argv[++i] : NULL)
        if      (!strcmp(s, "-m"))            a->model_path = EAT();
        else if (!strcmp(s, "-p"))            a->prompt = EAT();
        else if (!strcmp(s, "-n"))            a->max_new = atoi(EAT());
        else if (!strcmp(s, "-t"))            a->temperature = (float)atof(EAT());
        else if (!strcmp(s, "--top-p"))       a->top_p = (float)atof(EAT());
        else if (!strcmp(s, "--top-k"))       a->top_k = atoi(EAT());
        else if (!strcmp(s, "--seed"))        a->seed = strtoull(EAT(), NULL, 10);
        else if (!strcmp(s, "--seq"))         a->seq_capacity = atoi(EAT());
        else if (!strcmp(s, "--interactive")) a->interactive = 1;
        else if (!strcmp(s, "--info"))        a->info_only = 1;
        else if (!strcmp(s, "--no-special"))  a->no_special = 1;
        else if (!strcmp(s, "--debug-top"))   a->debug_top = atoi(EAT());
        else if (!strcmp(s, "--dump-tokens")) a->dump_tokens = 1;
        else if (!strcmp(s, "--layer-trace")) a->layer_trace = atoi(EAT());
        else if (!strcmp(s, "--layer-trace-out")) a->layer_trace_out = EAT();
        else if (!strcmp(s, "--layer-trace-pos")) a->layer_trace_pos = atoi(EAT());
        else if (!strcmp(s, "-h") || !strcmp(s, "--help")) { usage(argv[0]); return 1; }
        else { fprintf(stderr, "unknown arg: %s\n", s); usage(argv[0]); return -1; }
        #undef EAT
    }
    if (!a->model_path) { usage(argv[0]); return -1; }
    return 0;
}

static double mono_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void print_info(qw36_engine *eng, qw36_tokenizer *tok) {
    const qw36_config *c = qw36_engine_config(eng);
    fprintf(stderr,
        "qw36 info:\n"
        "  hidden=%u  intermediate=%u  layers=%u\n"
        "  heads=%u  kv_heads=%u  head_dim=%u\n"
        "  vocab=%u  context=%u\n"
        "  rms_eps=%g  rope_theta=%g  partial_rotary=%g\n"
        "  moe_experts=%u  moe_per_tok=%u  moe_inter=%u\n"
        "  bos=%u eos=%u im_start=%u im_end=%u\n",
        c->hidden_size, c->intermediate_size, c->num_hidden_layers,
        c->num_attention_heads, c->num_key_value_heads, c->head_dim,
        c->vocab_size, c->max_position_embeddings,
        c->rms_norm_eps, c->rope_theta, c->partial_rotary_factor,
        c->moe_num_experts, c->moe_experts_per_tok, c->moe_intermediate_size,
        qw36_tokenizer_bos(tok), qw36_tokenizer_eos(tok),
        qw36_tokenizer_im_start(tok), qw36_tokenizer_im_end(tok));
}

/* Build the Qwen3 chat-template wrapped prompt:
 *   <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
 * If --no-special is set, just use the raw prompt. */
static int build_prompt_ids(const qw36_tokenizer *tok, const char *prompt,
                            int wrap_chat,
                            uint32_t **out_ids, size_t *out_n)
{
    const char *full = prompt;
    char *scratch = NULL;
    if (wrap_chat) {
        size_t plen = strlen(prompt);
        scratch = (char *)malloc(plen + 64);
        if (!scratch) return -1;
        /* We rely on the tokenizer's special-token vocab containing
         * "<|im_start|>" / "<|im_end|>" — load_specials wires those up. */
        snprintf(scratch, plen + 64,
                 "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
                 prompt);
        full = scratch;
    }

    size_t cap = strlen(full) * 4 + 16;
    uint32_t *ids = (uint32_t *)malloc(cap * sizeof(uint32_t));
    if (!ids) { free(scratch); return -1; }
    size_t n = cap;
    int rc = qw36_tokenizer_encode(tok, full, ids, &n);
    free(scratch);
    if (rc) { free(ids); return -1; }
    *out_ids = ids;
    *out_n   = n;
    return 0;
}

int main(int argc, char **argv) {
    qw36_cli_args a = {0};
    int rc = parse_args(argc, argv, &a);
    if (rc) return rc < 0 ? 1 : 0;

    /* Backend: NULL ⇒ CPU reference path. */
    qw36_gpu_backend *be = qw36_backend_create();
    fprintf(stderr, "qw36: backend = %s\n", be ? be->name : "cpu (reference)");

    char err[256] = {0};
    qw36_engine *eng = qw36_engine_open(a.model_path, be, err, sizeof(err));
    if (!eng) { fprintf(stderr, "qw36: %s\n", err); return 3; }

    qw36_tokenizer *tok = qw36_tokenizer_new(qw36_engine_gguf(eng),
                                             err, sizeof(err));
    if (!tok) { fprintf(stderr, "qw36: tokenizer: %s\n", err); qw36_engine_close(eng); return 4; }

    if (a.info_only) {
        /* Layer-type detection: vanilla Qwen3 uses blk.X.attn_q.weight per
         * layer; Qwen3.5/3.6 uses fused attn_qkv + ssm_* (Gated DeltaNet). */
        const struct qw36_gguf_file *gf = qw36_engine_gguf(eng);
        int n_layers = (int)qw36_engine_config(eng)->num_hidden_layers;
        int has_q = 0, has_fused = 0, has_ssm = 0;
        for (int l = 0; l < n_layers; l++) {
            char buf[128]; qw36_gguf_tensor t;
            snprintf(buf, sizeof(buf), "blk.%d.attn_q.weight", l);
            if (qw36_gguf_get_tensor(gf, buf, &t) == 0) has_q++;
            snprintf(buf, sizeof(buf), "blk.%d.attn_qkv.weight", l);
            if (qw36_gguf_get_tensor(gf, buf, &t) == 0) has_fused++;
            snprintf(buf, sizeof(buf), "blk.%d.ssm_conv1d.weight", l);
            if (qw36_gguf_get_tensor(gf, buf, &t) == 0) has_ssm++;
        }
        fprintf(stderr, "qw36 layer survey: vanilla_q=%d/%d  fused_qkv=%d/%d  ssm=%d/%d\n",
                has_q, n_layers, has_fused, n_layers, has_ssm, n_layers);
        print_info(eng, tok);
        qw36_tokenizer_free(tok);
        qw36_engine_close(eng);
        return 0;
    }

    if (!a.prompt) {
        fprintf(stderr, "qw36: -p <prompt> required (or use --info)\n");
        qw36_tokenizer_free(tok); qw36_engine_close(eng); return 5;
    }

    if (a.dump_tokens) {
        uint32_t *pids = NULL; size_t pn = 0;
        if (build_prompt_ids(tok, a.prompt, !a.no_special, &pids, &pn)) {
            fprintf(stderr, "qw36: tokenize failed\n");
            qw36_tokenizer_free(tok); qw36_engine_close(eng); return 7;
        }
        fprintf(stderr, "tokens (%zu):\n", pn);
        for (size_t i = 0; i < pn; i++) {
            size_t blen = 0;
            const char *s = qw36_tokenizer_decode_one(tok, pids[i], &blen);
            fprintf(stderr, "  [%zu] id=%u  ", i, pids[i]);
            if (s) { fwrite(s, 1, blen, stderr); }
            fputc('\n', stderr);
        }
        free(pids); qw36_tokenizer_free(tok); qw36_engine_close(eng); return 0;
    }

    qw36_state *st = qw36_state_new(eng, (uint32_t)a.seq_capacity);
    if (!st) { fprintf(stderr, "qw36: state alloc failed\n");
               qw36_tokenizer_free(tok); qw36_engine_close(eng); return 6; }

    uint32_t *pids = NULL; size_t pn = 0;
    if (build_prompt_ids(tok, a.prompt, !a.no_special, &pids, &pn)) {
        fprintf(stderr, "qw36: tokenize failed\n");
        qw36_state_free(st); qw36_tokenizer_free(tok); qw36_engine_close(eng); return 7;
    }
    if (a.layer_trace >= 0) {
        if (!a.layer_trace_out || !a.layer_trace_out[0]) {
            fprintf(stderr, "qw36: --layer-trace requires --layer-trace-out <path>\n");
            free(pids); qw36_state_free(st); qw36_tokenizer_free(tok);
            qw36_engine_close(eng); return 7;
        }
        char trace_buf[32];
        snprintf(trace_buf, sizeof(trace_buf), "%d", a.layer_trace);
        setenv("QW36_TRACE_LAYER", trace_buf, 1);
        setenv("QW36_TRACE_OUT", a.layer_trace_out, 1);
        if (a.layer_trace_pos >= 0) {
            snprintf(trace_buf, sizeof(trace_buf), "%d", a.layer_trace_pos);
            setenv("QW36_TRACE_POS", trace_buf, 1);
        } else {
            unsetenv("QW36_TRACE_POS");
        }
    }
    fprintf(stderr, "qw36: prompt tokens = %zu\n", pn);

    double t0 = mono_seconds();
    if (qw36_prefill(eng, st, pids, pn) != 0) {
        fprintf(stderr, "qw36: prefill failed\n");
        free(pids); qw36_state_free(st); qw36_tokenizer_free(tok);
        qw36_engine_close(eng); return 8;
    }
    double t1 = mono_seconds();
    fprintf(stderr, "qw36: prefill %zu tokens in %.3fs (%.1f tok/s)\n",
            pn, t1 - t0, (double)pn / (t1 - t0));

    qw36_sampler sp = {
        .temperature = a.temperature,
        .top_p       = a.top_p,
        .top_k       = a.top_k,
        .rng_seed    = a.seed,
    };

    /* Generate. After prefill, st->logits is the distribution at the last
     * prompt token. Sample, print, feed back. */
    const uint32_t eos = qw36_tokenizer_eos(tok);
    const uint32_t imend = qw36_tokenizer_im_end(tok);
    int generated = 0;
    double tg0 = mono_seconds();
    const uint32_t vocab = qw36_engine_config(eng)->vocab_size;
    for (int i = 0; i < a.max_new; i++) {
        if (a.debug_top > 0) {
            /* Print top-K logits for the current step (before sampling). */
            int K = a.debug_top;
            float *tk_val = (float *)alloca(K * sizeof(float));
            uint32_t *tk_idx = (uint32_t *)alloca(K * sizeof(uint32_t));
            for (int j = 0; j < K; j++) { tk_val[j] = -1e30f; tk_idx[j] = 0; }
            for (uint32_t v = 0; v < vocab; v++) {
                if (st->logits[v] <= tk_val[K-1]) continue;
                int j = K - 1;
                while (j > 0 && st->logits[v] > tk_val[j-1]) {
                    tk_val[j] = tk_val[j-1]; tk_idx[j] = tk_idx[j-1]; j--;
                }
                tk_val[j] = st->logits[v]; tk_idx[j] = v;
            }
            fprintf(stderr, "[step %d] top-%d:\n", i, K);
            for (int j = 0; j < K; j++) {
                size_t blen = 0;
                const char *s = qw36_tokenizer_decode_one(tok, tk_idx[j], &blen);
                fprintf(stderr, "  %.4f  id=%u  '", tk_val[j], tk_idx[j]);
                if (s) fwrite(s, 1, blen, stderr);
                fprintf(stderr, "'\n");
            }
        }
        uint32_t next = qw36_sample(st->logits, vocab, &sp);
        if (next == eos || (imend && next == imend)) break;
        size_t blen = 0;
        const char *s = qw36_tokenizer_decode_one(tok, next, &blen);
        if (s && blen) {
            fwrite(s, 1, blen, stdout);
            fflush(stdout);
        }
        if (qw36_forward(eng, st, next) != 0) break;
        generated++;
    }
    double tg1 = mono_seconds();
    fputc('\n', stdout);
    fprintf(stderr, "qw36: generated %d tokens in %.3fs (%.1f tok/s)\n",
            generated, tg1 - tg0,
            generated > 0 ? (double)generated / (tg1 - tg0) : 0.0);

    free(pids);
    qw36_state_free(st);
    qw36_tokenizer_free(tok);
    qw36_engine_close(eng);
    return 0;
}
