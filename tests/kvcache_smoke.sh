#!/usr/bin/env bash
# Smoke for the KV prefix cache (task #75 / AD).
#
# Exercises both shipped tiers (ram_lru, disk) end-to-end:
#   - lookup miss on empty cache
#   - insert/lookup round-trip with payload integrity
#   - disk persistence: 2nd-process lookup hits the disk tier
#   - promotion: disk-only entry gets pulled into the ram tier
#   - stats: hits_by_tier counters track correctly
# Then re-runs the 0.8B fastest-path smoke as a regression check.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[kvcache] cpu build smoke"
make -s -C cpu 1>/dev/null

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
DISK_DIR="$TMP/kvcache-disk"

cat > "$TMP/main.c" <<'EOF'
#include "qw36_kvcache.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void payload_fill(uint8_t *buf, size_t n, uint32_t seed) {
    for (size_t i = 0; i < n; i++) buf[i] = (uint8_t)(seed * 31u + i);
}
static int payload_check(const uint8_t *buf, size_t n, uint32_t seed) {
    for (size_t i = 0; i < n; i++) if (buf[i] != (uint8_t)(seed * 31u + i)) return 0;
    return 1;
}

int main(int argc, char **argv) {
    const char *disk_dir = argc > 1 ? argv[1] : NULL;
    int phase = argc > 2 ? atoi(argv[2]) : 0;

    qw36_kv_cache_cfg cfg = { .min_prefix_tokens = 4, .writeback_on_evict = 1 };

    /* Single-tier ram lookup miss path (no disk). */
    {
        qw36_kv_prefix_cache *c =
            qw36_kv_cache_create_default(64*1024, NULL, 0, &cfg);
        assert(c);
        uint32_t toks[8] = {1,2,3,4,5,6,7,8};
        assert(qw36_kv_cache_lookup(c, toks, 8, NULL, NULL) == 0);
        qw36_kv_cache_destroy(c);
    }

    /* 2-tier ram + disk round-trip. */
    qw36_kv_prefix_cache *c =
        qw36_kv_cache_create_default(64*1024, disk_dir, 1024*1024, &cfg);
    assert(c);

    uint32_t toks_a[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    uint32_t toks_b[12] = {10,20,30,40,50,60,70,80,90,100,110,120};
    uint8_t  payload_a[512]; payload_fill(payload_a, sizeof payload_a, 7);
    uint8_t  payload_b[128]; payload_fill(payload_b, sizeof payload_b, 11);

    if (phase == 0) {
        /* Phase 0: write into ram+disk. */
        assert(qw36_kv_cache_insert(c, toks_a, 16, payload_a, sizeof payload_a) == 0);
        assert(qw36_kv_cache_insert(c, toks_b, 12, payload_b, sizeof payload_b) == 0);

        const void *buf = NULL;
        size_t bytes = 0;
        size_t hit = qw36_kv_cache_lookup(c, toks_a, 16, &buf, &bytes);
        assert(hit == 16);
        assert(bytes == sizeof payload_a);
        assert(payload_check((const uint8_t *)buf, bytes, 7));

        hit = qw36_kv_cache_lookup(c, toks_b, 12, &buf, &bytes);
        assert(hit == 12);
        assert(bytes == sizeof payload_b);
        assert(payload_check((const uint8_t *)buf, bytes, 11));

        qw36_kv_cache_stats_t s;
        qw36_kv_cache_stats(c, &s);
        fprintf(stderr,
                "[kvcache phase 0] inserts=%llu hits[0]=%llu hits[1]=%llu writebacks=%llu\n",
                (unsigned long long)s.inserts,
                (unsigned long long)s.hits_by_tier[0],
                (unsigned long long)s.hits_by_tier[1],
                (unsigned long long)s.writebacks);
        assert(s.inserts == 2);
        assert(s.hits_by_tier[0] >= 2);
        assert(s.writebacks >= 2);  /* both should mirror to disk */
    } else {
        /* Phase 1: fresh process, ram tier empty, must hit disk. */
        const void *buf = NULL;
        size_t bytes = 0;
        size_t hit = qw36_kv_cache_lookup(c, toks_a, 16, &buf, &bytes);
        assert(hit == 16);
        assert(bytes == sizeof payload_a);
        assert(payload_check((const uint8_t *)buf, bytes, 7));

        qw36_kv_cache_stats_t s;
        qw36_kv_cache_stats(c, &s);
        fprintf(stderr,
                "[kvcache phase 1] hits[0]=%llu hits[1]=%llu promotions=%llu\n",
                (unsigned long long)s.hits_by_tier[0],
                (unsigned long long)s.hits_by_tier[1],
                (unsigned long long)s.promotions);
        assert(s.hits_by_tier[1] == 1);    /* hit on the disk tier */
        assert(s.promotions == 1);          /* promoted into ram */

        /* Second lookup should hit ram now (promoted on phase 1 lookup). */
        hit = qw36_kv_cache_lookup(c, toks_a, 16, &buf, &bytes);
        assert(hit == 16);
        qw36_kv_cache_stats(c, &s);
        assert(s.hits_by_tier[0] == 1);
    }

    qw36_kv_cache_destroy(c);
    fprintf(stderr, "[kvcache] phase %d ok\n", phase);
    return 0;
}
EOF

cc -O2 -std=c11 -Wall -Wextra -Wno-unused-parameter \
   -Icommon "$TMP/main.c" common/qw36_kvcache.c \
   -o "$TMP/kvcache_smoke"

echo "[kvcache] phase 0 (write through ram + disk)"
"$TMP/kvcache_smoke" "$DISK_DIR" 0
echo "[kvcache] phase 1 (fresh process, disk hit + promote)"
"$TMP/kvcache_smoke" "$DISK_DIR" 1
echo "[kvcache] disk files:"
ls -la "$DISK_DIR" 2>/dev/null | head -5

echo "[kvcache] 0.8B fastest-path regression"
./tests/quant_fastest_smoke.sh
