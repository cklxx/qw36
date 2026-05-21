/* qw36_kvcache.h — KV prefix cache (L1/L2/L3) public API.
 *
 * Single-stream prefix reuse: when a new request shares a leading token
 * sequence with a recent one, the engine can rehydrate the cached
 * per-layer KV / conv_state / delta_state and skip prefill for that
 * region. See docs/kvcache_design.md.
 *
 * v0: L2 in-RAM LRU, L3 on-disk persistence, exact-prefix match only.
 * Engine wiring lives elsewhere; this header is the stable contract. */
#ifndef QW36_KVCACHE_H
#define QW36_KVCACHE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qw36_state qw36_state;       /* fwd-decl from qw36.h */
typedef struct qw36_kv_prefix_cache qw36_kv_prefix_cache;

typedef struct {
    /* Maximum bytes the in-RAM LRU is allowed to keep. 0 disables L2. */
    size_t l2_capacity_bytes;
    /* Filesystem directory for L3. NULL or "" disables L3. The
     * directory is created on demand; one file per prefix hash. */
    const char *l3_dir;
    /* Maximum disk bytes for L3. 0 = unlimited, the file system caps. */
    size_t l3_capacity_bytes;
    /* Minimum prefix length (in tokens) to bother caching at all.
     * Short prefixes are cheap to recompute; caching them wastes the
     * cache for longer high-value prefixes. Default 32. */
    uint32_t min_prefix_tokens;
} qw36_kv_cache_cfg;

typedef struct {
    uint64_t lookups;
    uint64_t l2_hits;
    uint64_t l3_hits;
    uint64_t misses;
    uint64_t inserts;
    uint64_t l2_evictions;
    uint64_t l3_writebacks;
    size_t   l2_bytes_used;
    size_t   l3_bytes_used;
    size_t   entries;
} qw36_kv_cache_stats_t;

/* Create a cache. Returns NULL on alloc failure or bad config. */
qw36_kv_prefix_cache *qw36_kv_cache_create(const qw36_kv_cache_cfg *cfg);
void                  qw36_kv_cache_destroy(qw36_kv_prefix_cache *c);

/* Look up the longest cached prefix of `tokens` and rehydrate `dst`
 * up to that position. Returns the number of tokens hydrated (0 on
 * miss, no side effect on dst). The engine should set st->seq_pos to
 * the returned value after a successful hydrate.
 *
 * `dst` must already be allocated by the engine with capacity ≥ matched
 * prefix length; otherwise the hydrate aborts and returns 0. */
size_t qw36_kv_cache_lookup(qw36_kv_prefix_cache *c,
                            const uint32_t *tokens, size_t n_tokens,
                            qw36_state *dst);

/* Snapshot the in-flight state at exactly `n_tokens` written. Idempotent
 * — re-insert on an existing key is a no-op. */
int qw36_kv_cache_insert(qw36_kv_prefix_cache *c,
                         const uint32_t *tokens, size_t n_tokens,
                         const qw36_state *src);

/* Stats are advisory; cache stays consistent without them being read. */
void qw36_kv_cache_stats(const qw36_kv_prefix_cache *c,
                         qw36_kv_cache_stats_t *out);

/* Forget everything (does not delete L3 files). Useful for tests. */
void qw36_kv_cache_clear(qw36_kv_prefix_cache *c);

#ifdef __cplusplus
}
#endif
#endif /* QW36_KVCACHE_H */
