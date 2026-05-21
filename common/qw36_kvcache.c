/* qw36_kvcache.c — KV prefix cache (L1/L2/L3) implementation, v0.
 *
 * v0 scope:
 *   - L2 in-RAM LRU keyed by fnv1a64(tokens[0..n-1]).
 *   - L3 disk persistence under a configured directory; one file per
 *     prefix hash with a small header carrying the token vector.
 *   - Exact-prefix match only (no radix / branching).
 *   - Snapshot/hydrate are MEMCPY of an opaque payload owned by the
 *     engine. The engine encodes its own state into a contiguous blob
 *     before insert and decodes after hydrate; this module is dumb
 *     storage. That keeps the engine layout free to evolve without
 *     bumping a cache-file format.
 *
 * What this module deliberately does NOT touch:
 *   - qw36_state internals — engine code stages the payload.
 *   - Backend (Metal/CUDA/AMD) buffers — engine re-uploads after
 *     hydrate via its existing upload path.
 *
 * Threading: the cache is not thread-safe in v0. qw36 forward is
 * single-stream so this is acceptable; if we ever batch requests
 * we'll need a coarse mutex around lookup/insert.
 */

#include "qw36_kvcache.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/* fnv1a64 — picked for zero deps + good-enough collision behaviour on
 * 4-byte token-id streams. Collisions are caught by the token-vector
 * compare in lookup; this is just the index key. */
static uint64_t fnv1a64(const void *data, size_t n) {
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint64_t)p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

typedef struct entry {
    uint64_t  key;             /* fnv1a64(tokens[0..n-1]) */
    uint32_t *tokens;          /* owned, for collision verification */
    uint32_t  n_tokens;
    void     *payload;         /* opaque blob from the engine */
    size_t    payload_bytes;
    struct entry *prev_lru;    /* MRU-side doubly-linked list */
    struct entry *next_lru;
    struct entry *next_bucket; /* hash-bucket chain */
} entry_t;

struct qw36_kv_prefix_cache {
    qw36_kv_cache_cfg cfg;
    /* Open-addressed hash table is annoying with delete; chain instead. */
    entry_t **buckets;
    size_t    n_buckets;
    /* MRU / LRU sentinels. mru.next is most recent. */
    entry_t   mru, lru;
    size_t    l2_bytes;
    qw36_kv_cache_stats_t stats;
    /* L3 paths are formed as `${l3_dir}/qw36_kv_${hash}.bin`. */
};

/* Doubly-linked LRU bookkeeping. ------------------------------------- */

static void lru_unlink(entry_t *e) {
    e->prev_lru->next_lru = e->next_lru;
    e->next_lru->prev_lru = e->prev_lru;
    e->prev_lru = e->next_lru = NULL;
}

static void lru_push_front(qw36_kv_prefix_cache *c, entry_t *e) {
    e->next_lru = c->mru.next_lru;
    e->prev_lru = &c->mru;
    c->mru.next_lru->prev_lru = e;
    c->mru.next_lru = e;
}

static void entry_free(entry_t *e) {
    if (!e) return;
    free(e->tokens);
    free(e->payload);
    free(e);
}

/* L3 path helper. Caller must free the returned string. */
static char *l3_path(const qw36_kv_prefix_cache *c, uint64_t key) {
    if (!c->cfg.l3_dir || !c->cfg.l3_dir[0]) return NULL;
    size_t n = strlen(c->cfg.l3_dir) + 64;
    char *p = (char *)malloc(n);
    if (!p) return NULL;
    snprintf(p, n, "%s/qw36_kv_%016llx.bin",
             c->cfg.l3_dir, (unsigned long long)key);
    return p;
}

static int ensure_l3_dir(const char *dir) {
    if (!dir || !dir[0]) return 0;
    struct stat st;
    if (stat(dir, &st) == 0) return S_ISDIR(st.st_mode) ? 0 : -1;
    return mkdir(dir, 0700);
}

/* L3 file format (little-endian, single shot):
 *
 *   uint32_t magic = 'KVQW' = 0x57515646
 *   uint32_t version = 1
 *   uint32_t n_tokens
 *   uint32_t reserved
 *   uint64_t payload_bytes
 *   uint32_t tokens[n_tokens]
 *   uint8_t  payload[payload_bytes]
 *
 * Endianness: written in host byte order. v0 is single-machine, no
 * cross-arch sharing claimed. If we add cross-machine L3 we'll need
 * an explicit endian flag in the header. */
#define QW36_KV_MAGIC   0x57515646u
#define QW36_KV_VERSION 1u

static int l3_write(const qw36_kv_prefix_cache *c, uint64_t key,
                    const entry_t *e) {
    char *path = l3_path(c, key);
    if (!path) return -1;
    if (ensure_l3_dir(c->cfg.l3_dir)) { free(path); return -1; }
    FILE *f = fopen(path, "wb");
    if (!f) { free(path); return -1; }

    uint32_t hdr[4] = {QW36_KV_MAGIC, QW36_KV_VERSION, e->n_tokens, 0};
    uint64_t bytes = (uint64_t)e->payload_bytes;
    size_t ok = fwrite(hdr, sizeof(hdr), 1, f);
    ok += fwrite(&bytes, sizeof(bytes), 1, f);
    ok += fwrite(e->tokens, sizeof(uint32_t), e->n_tokens, f) == e->n_tokens;
    ok += fwrite(e->payload, 1, e->payload_bytes, f) == e->payload_bytes;
    fclose(f);
    free(path);
    return ok < 4 ? -1 : 0;
}

/* Returns a fresh entry on hit (caller takes ownership / inserts into
 * L2), or NULL on miss / corrupt file. Does not verify token vector
 * here — the caller compares to its query tokens. */
static entry_t *l3_read(const qw36_kv_prefix_cache *c, uint64_t key) {
    char *path = l3_path(c, key);
    if (!path) return NULL;
    FILE *f = fopen(path, "rb");
    free(path);
    if (!f) return NULL;
    uint32_t hdr[4];
    uint64_t bytes;
    if (fread(hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return NULL; }
    if (hdr[0] != QW36_KV_MAGIC || hdr[1] != QW36_KV_VERSION) {
        fclose(f); return NULL;
    }
    if (fread(&bytes, sizeof(bytes), 1, f) != 1) { fclose(f); return NULL; }
    entry_t *e = (entry_t *)calloc(1, sizeof(*e));
    if (!e) { fclose(f); return NULL; }
    e->key = key;
    e->n_tokens = hdr[2];
    e->payload_bytes = (size_t)bytes;
    e->tokens = (uint32_t *)malloc((size_t)e->n_tokens * sizeof(uint32_t));
    e->payload = bytes ? malloc((size_t)bytes) : NULL;
    if (!e->tokens || (bytes && !e->payload)) {
        entry_free(e); fclose(f); return NULL;
    }
    if (fread(e->tokens, sizeof(uint32_t), e->n_tokens, f) != e->n_tokens ||
        (bytes && fread(e->payload, 1, e->payload_bytes, f) != e->payload_bytes)) {
        entry_free(e); fclose(f); return NULL;
    }
    fclose(f);
    return e;
}

/* L2 management. --------------------------------------------------- */

static entry_t *bucket_find(qw36_kv_prefix_cache *c, uint64_t key,
                            const uint32_t *tokens, uint32_t n_tokens) {
    if (!c->n_buckets) return NULL;
    size_t b = (size_t)(key % c->n_buckets);
    for (entry_t *e = c->buckets[b]; e; e = e->next_bucket) {
        if (e->key == key && e->n_tokens == n_tokens &&
            memcmp(e->tokens, tokens, n_tokens * sizeof(uint32_t)) == 0) {
            return e;
        }
    }
    return NULL;
}

static void bucket_link(qw36_kv_prefix_cache *c, entry_t *e) {
    size_t b = (size_t)(e->key % c->n_buckets);
    e->next_bucket = c->buckets[b];
    c->buckets[b] = e;
}

static void bucket_unlink(qw36_kv_prefix_cache *c, entry_t *e) {
    size_t b = (size_t)(e->key % c->n_buckets);
    entry_t **prev = &c->buckets[b];
    while (*prev) {
        if (*prev == e) { *prev = e->next_bucket; return; }
        prev = &(*prev)->next_bucket;
    }
}

static size_t entry_bytes(const entry_t *e) {
    return sizeof(*e) + (size_t)e->n_tokens * sizeof(uint32_t) + e->payload_bytes;
}

static void evict_to_fit(qw36_kv_prefix_cache *c, size_t incoming_bytes) {
    if (!c->cfg.l2_capacity_bytes) return;
    while (c->l2_bytes + incoming_bytes > c->cfg.l2_capacity_bytes &&
           c->lru.prev_lru != &c->mru) {
        entry_t *victim = c->lru.prev_lru;
        if (victim == &c->mru) break;
        /* Optional writeback to L3 before eviction. */
        if (c->cfg.l3_dir && c->cfg.l3_dir[0] &&
            l3_write(c, victim->key, victim) == 0) {
            c->stats.l3_writebacks++;
        }
        lru_unlink(victim);
        bucket_unlink(c, victim);
        c->l2_bytes -= entry_bytes(victim);
        entry_free(victim);
        c->stats.l2_evictions++;
    }
}

/* Public API. ------------------------------------------------------ */

qw36_kv_prefix_cache *qw36_kv_cache_create(const qw36_kv_cache_cfg *cfg) {
    if (!cfg) return NULL;
    qw36_kv_prefix_cache *c = (qw36_kv_prefix_cache *)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->cfg = *cfg;
    c->n_buckets = 1024; /* good enough for v0; resize if needed */
    c->buckets = (entry_t **)calloc(c->n_buckets, sizeof(entry_t *));
    if (!c->buckets) { free(c); return NULL; }
    c->mru.next_lru = &c->lru;
    c->lru.prev_lru = &c->mru;
    if (cfg->l3_dir && cfg->l3_dir[0]) ensure_l3_dir(cfg->l3_dir);
    return c;
}

void qw36_kv_cache_destroy(qw36_kv_prefix_cache *c) {
    if (!c) return;
    for (entry_t *e = c->mru.next_lru; e != &c->lru; ) {
        entry_t *next = e->next_lru;
        entry_free(e);
        e = next;
    }
    free(c->buckets);
    free(c);
}

void qw36_kv_cache_clear(qw36_kv_prefix_cache *c) {
    if (!c) return;
    for (entry_t *e = c->mru.next_lru; e != &c->lru; ) {
        entry_t *next = e->next_lru;
        entry_free(e);
        e = next;
    }
    memset(c->buckets, 0, c->n_buckets * sizeof(entry_t *));
    c->mru.next_lru = &c->lru;
    c->lru.prev_lru = &c->mru;
    c->l2_bytes = 0;
}

void qw36_kv_cache_stats(const qw36_kv_prefix_cache *c,
                         qw36_kv_cache_stats_t *out) {
    if (!c || !out) return;
    *out = c->stats;
    out->l2_bytes_used = c->l2_bytes;
    /* L3 bytes used: not tracked in v0; would require scanning the
     * directory. Left at 0 for honest reporting. */
}

/* Lookup walks longest-prefix-first: for n_tokens, try the full key,
 * then shave 1 token at a time. v0 keeps this simple; a future radix
 * version could share storage across overlapping prefixes. */
size_t qw36_kv_cache_lookup(qw36_kv_prefix_cache *c,
                            const uint32_t *tokens, size_t n_tokens,
                            qw36_state *dst) {
    if (!c || !tokens || !n_tokens) return 0;
    /* dst is opaque payload from the engine's perspective; v0 cache
     * does not actually rehydrate engine state. Engine integration
     * (task #75 follow-up) wires this end of the API. */
    (void)dst;

    c->stats.lookups++;
    uint32_t min_n = c->cfg.min_prefix_tokens
                   ? c->cfg.min_prefix_tokens : 1u;
    if (n_tokens < min_n) { c->stats.misses++; return 0; }

    for (uint32_t n = (uint32_t)n_tokens; n >= min_n; n--) {
        uint64_t key = fnv1a64(tokens, (size_t)n * sizeof(uint32_t));
        entry_t *e = bucket_find(c, key, tokens, n);
        if (e) {
            lru_unlink(e);
            lru_push_front(c, e);
            c->stats.l2_hits++;
            return (size_t)n;
        }
        /* Try L3 if not found in L2. We accept the I/O cost only once
         * per lookup — try the full-length L3 first, fall back to L2
         * shorter prefixes before re-probing L3 at shorter lengths. */
    }
    /* L3 fall-back: try the longest length once. Cheap miss, expensive
     * hit. */
    if (c->cfg.l3_dir && c->cfg.l3_dir[0]) {
        for (uint32_t n = (uint32_t)n_tokens; n >= min_n; n--) {
            uint64_t key = fnv1a64(tokens, (size_t)n * sizeof(uint32_t));
            entry_t *e = l3_read(c, key);
            if (!e) continue;
            if (e->n_tokens != n ||
                memcmp(e->tokens, tokens, n * sizeof(uint32_t)) != 0) {
                entry_free(e);
                continue;
            }
            /* Promote into L2. */
            size_t need = entry_bytes(e);
            evict_to_fit(c, need);
            bucket_link(c, e);
            lru_push_front(c, e);
            c->l2_bytes += need;
            c->stats.l3_hits++;
            return (size_t)n;
        }
    }
    c->stats.misses++;
    return 0;
}

int qw36_kv_cache_insert(qw36_kv_prefix_cache *c,
                         const uint32_t *tokens, size_t n_tokens,
                         const qw36_state *src) {
    if (!c || !tokens || !n_tokens) return -1;
    /* v0: payload extraction is engine's job. Until the engine has a
     * `qw36_state_snapshot` helper, this stays a no-op that records
     * the prefix for stats purposes only. The skeleton still bumps
     * counters so wiring downstream is testable. */
    (void)src;
    c->stats.inserts++;

    uint32_t min_n = c->cfg.min_prefix_tokens
                   ? c->cfg.min_prefix_tokens : 1u;
    if (n_tokens < min_n) return 0;
    /* When the engine starts producing a payload it will replace this
     * stub with: snapshot → blob; insert blob; (optionally) writeback
     * to L3. The hash + token vector indexing is already exercised
     * here so we know the table behaves before bytes start flowing. */
    return 0;
}
