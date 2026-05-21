#!/usr/bin/env bash
# Standalone smoke for the KV prefix cache skeleton (task #75 / AD).
#
# v0 of the cache is dumb storage: insert is a no-op until engine wiring
# lands. This script only verifies that:
#   - the new translation units compile into qw36_cpu
#   - the existing 0.8B smoke still passes (no regression)
#   - the kvcache test binary's bucket / fnv1a64 / lookup_miss paths run
#     clean (built from the header below).
#
# Engine integration (snapshot + hydrate of qw36_state) is the next task
# after codex's 35B Metal work lands.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[kvcache] cpu build smoke"
make -s -C cpu 1>/dev/null

echo "[kvcache] standalone harness compile"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/main.c" <<'EOF'
#include "qw36_kvcache.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    qw36_kv_cache_cfg cfg = {
        .l2_capacity_bytes = 1024 * 1024,
        .l3_dir = NULL,
        .l3_capacity_bytes = 0,
        .min_prefix_tokens = 4,
    };
    qw36_kv_prefix_cache *c = qw36_kv_cache_create(&cfg);
    if (!c) { fprintf(stderr, "create failed\n"); return 1; }

    uint32_t toks[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};

    /* short prefix below threshold → miss */
    size_t hit = qw36_kv_cache_lookup(c, toks, 3, NULL);
    assert(hit == 0);

    /* miss on full prefix when nothing inserted */
    hit = qw36_kv_cache_lookup(c, toks, 16, NULL);
    assert(hit == 0);

    /* insert is no-op in v0 (engine wiring is the next task) */
    int ins = qw36_kv_cache_insert(c, toks, 16, NULL);
    assert(ins == 0);

    qw36_kv_cache_stats_t st;
    qw36_kv_cache_stats(c, &st);
    fprintf(stderr,
            "[kvcache] lookups=%llu misses=%llu inserts=%llu l2_bytes=%zu\n",
            (unsigned long long)st.lookups,
            (unsigned long long)st.misses,
            (unsigned long long)st.inserts,
            st.l2_bytes_used);

    qw36_kv_cache_clear(c);
    qw36_kv_cache_destroy(c);
    fprintf(stderr, "[kvcache] standalone smoke ok\n");
    return 0;
}
EOF

# Compile against the cache source + header.
cc -O2 -std=c11 -Wall -Wextra -Wno-unused-parameter \
   -Icommon "$TMP/main.c" common/qw36_kvcache.c \
   -o "$TMP/kvcache_smoke"
"$TMP/kvcache_smoke"

echo "[kvcache] reusing repo smoke for 0.8B regression check"
./tests/quant_fastest_smoke.sh
