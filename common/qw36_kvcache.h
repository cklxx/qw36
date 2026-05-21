/* qw36_kvcache.h — KV prefix cache, tiered + backend-agnostic.
 *
 * The cache is an **ordered list of tiers** (hottest first). Each tier
 * implements a small `qw36_kv_tier_ops` vtable: get / put / evict /
 * stats / destroy. The cache loop is generic — lookup walks tiers
 * top-down with promotion, insert writes the hot tier, eviction
 * demotes to the next tier — so adding a new storage medium (CUDA
 * VRAM, RDMA / shared / network) is a matter of writing one new tier,
 * not changing the cache. See docs/kvcache_design.md.
 *
 * Concrete tiers shipped here:
 *   - qw36_kv_tier_ram_lru: in-process RAM, LRU with byte budget.
 *   - qw36_kv_tier_disk:    on-disk store, one file per prefix hash,
 *                           bounded by a per-machine byte budget.
 *
 * Mac unified memory naturally uses (ram_lru, disk). Discrete-GPU
 * platforms can prepend a VRAM tier so the hot path doesn't pay an
 * upload on hit. The cache treats every tier identically; only the
 * tier implementation knows about its medium.
 *
 * v0 is exact-prefix match only; future versions may share storage
 * across overlapping prefixes (radix). The tier vtable is stable
 * across that change. */
#ifndef QW36_KVCACHE_H
#define QW36_KVCACHE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qw36_state qw36_state;          /* fwd-decl from qw36.h */
typedef struct qw36_kv_prefix_cache qw36_kv_prefix_cache;
typedef struct qw36_kv_tier         qw36_kv_tier;

/* ---------------------------------------------------------------- */
/* Tier vtable — every storage medium implements this.              */
/* ---------------------------------------------------------------- */

typedef struct qw36_kv_tier_ops {
    /* Stable label for stats / logs. Static lifetime. */
    const char *name;

    /* Lookup: 0 on miss, 1 on hit. On hit, *out_buf / *out_bytes
     * point to a buffer the tier owns until the next get/put call
     * (lifetime is conservative — the cache copies bytes before
     * touching the tier again). Token vector is supplied so the
     * tier can disambiguate hash collisions. */
    int  (*get)(qw36_kv_tier *t, uint64_t key,
                const uint32_t *tokens, uint32_t n_tokens,
                const void **out_buf, size_t *out_bytes);

    /* Insert: 0 on success. The tier copies the bytes; the caller
     * keeps ownership of `buf`. May internally evict to make space.
     * Tiers that don't accept inserts (e.g. read-only mirrors) may
     * return >0 to signal "ignored, not an error". */
    int  (*put)(qw36_kv_tier *t, uint64_t key,
                const uint32_t *tokens, uint32_t n_tokens,
                const void *buf, size_t bytes);

    /* Total bytes resident in this tier (advisory; for cap checks). */
    size_t (*used_bytes)(const qw36_kv_tier *t);

    /* Capacity. SIZE_MAX = unbounded (e.g. disk). */
    size_t (*capacity_bytes)(const qw36_kv_tier *t);

    /* Forget everything in this tier (does not delete external files
     * unless the tier explicitly opts in). */
    void (*clear)(qw36_kv_tier *t);

    /* Free per-tier state. */
    void (*destroy)(qw36_kv_tier *t);
} qw36_kv_tier_ops;

struct qw36_kv_tier {
    const qw36_kv_tier_ops *ops;
    void                   *state;   /* tier-private */
};

/* ---------------------------------------------------------------- */
/* Concrete tiers.                                                  */
/* ---------------------------------------------------------------- */

/* In-process RAM LRU. capacity_bytes=0 disables. */
qw36_kv_tier *qw36_kv_tier_ram_lru_create(size_t capacity_bytes);

/* On-disk store. One file per prefix hash under `dir`, file name
 * `qw36_kv_<hex>.bin`. capacity_bytes=0 disables (still readable if
 * dir non-empty? No — disabled means tier rejects all gets too).
 * `dir` is duplicated internally; callers may free their copy. */
qw36_kv_tier *qw36_kv_tier_disk_create(const char *dir,
                                       size_t capacity_bytes);

/* ---------------------------------------------------------------- */
/* Cache — composes tiers, hot first.                               */
/* ---------------------------------------------------------------- */

typedef struct {
    /* Minimum prefix length (in tokens) to bother caching. Short
     * prefixes are cheap to recompute. Default 32. */
    uint32_t min_prefix_tokens;
    /* On evict-from-tier-i, write back to tier-(i+1)? Default 1.
     * Set 0 if downstream tiers should be strictly explicit. */
    uint8_t  writeback_on_evict;
} qw36_kv_cache_cfg;

typedef struct {
    uint64_t lookups;
    uint64_t hits_by_tier[8];      /* up to 8 tiers; tier 0 = hottest */
    uint64_t misses;
    uint64_t inserts;
    uint64_t writebacks;
    uint64_t promotions;
    size_t   bytes_by_tier[8];
    uint32_t n_tiers;
} qw36_kv_cache_stats_t;

/* Construct a cache from a left-to-right ordered tier list (hot first).
 * The cache takes ownership of the tier pointers and will destroy them
 * with the cache. Pass at most 8 tiers (v0 limit; bump if needed). */
qw36_kv_prefix_cache *qw36_kv_cache_create(qw36_kv_tier **tiers,
                                           uint32_t n_tiers,
                                           const qw36_kv_cache_cfg *cfg);

/* Convenience: the Mac-typical (ram_lru, disk) duo. Either capacity
 * 0 omits that tier. Returns NULL if both are 0. */
qw36_kv_prefix_cache *qw36_kv_cache_create_default(size_t l2_ram_bytes,
                                                   const char *l3_disk_dir,
                                                   size_t l3_disk_bytes,
                                                   const qw36_kv_cache_cfg *cfg);

void                  qw36_kv_cache_destroy(qw36_kv_prefix_cache *c);

/* Longest-prefix lookup. Returns matched prefix length (0 on miss).
 * On hit the matching tier's payload is the cache's working buffer
 * (caller-owned read; copy before next cache call if needed beyond
 * one operation). Promotes the hit into all hotter tiers. */
size_t qw36_kv_cache_lookup(qw36_kv_prefix_cache *c,
                            const uint32_t *tokens, size_t n_tokens,
                            const void **out_buf, size_t *out_bytes);

/* Insert into the hot tier. The cache duplicates the buffer; the
 * caller keeps ownership. Eviction cascades to colder tiers when
 * writeback_on_evict is set. */
int qw36_kv_cache_insert(qw36_kv_prefix_cache *c,
                         const uint32_t *tokens, size_t n_tokens,
                         const void *buf, size_t bytes);

void qw36_kv_cache_clear(qw36_kv_prefix_cache *c);
void qw36_kv_cache_stats(const qw36_kv_prefix_cache *c,
                         qw36_kv_cache_stats_t *out);

/* ---------------------------------------------------------------- */
/* Engine glue (separate file): qw36_kv_state_snapshot /            */
/* qw36_kv_state_hydrate live alongside qw36_state because they     */
/* depend on its internals. The cache only stores opaque blobs.     */
/* ---------------------------------------------------------------- */

#ifdef __cplusplus
}
#endif
#endif /* QW36_KVCACHE_H */
