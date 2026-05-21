/* qw36_kvcache.c — KV prefix cache, tier-composing implementation.
 *
 * The cache here is generic: a lookup walks the configured tier list
 * from hottest to coldest, promotes hits, and (optionally) writes
 * back on evict. The two concrete tiers shipped (RAM LRU + disk) are
 * separate translation-internal modules in this same file.
 *
 * Adding a new tier (CUDA VRAM resident, Redis, S3, etc.) is one new
 * `qw36_kv_tier_ops` table + ctor; nothing in `cache_lookup` /
 * `cache_insert` changes.
 *
 * The cache stores opaque payload bytes — the engine encodes its
 * qw36_state into a blob (qw36_state_snapshot, lives next to
 * qw36_state) and decodes on hydrate. This module does not know about
 * KV cache shape, layers, or backends, which is what lets it stay
 * 100% backend-agnostic and reusable across all four GPU targets. */

#include "qw36_kvcache.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/* ----------------------------------------------------------------- */
/* fnv1a64 — fast, zero-dep, plenty for token-id keying.             */
/* ----------------------------------------------------------------- */

static uint64_t fnv1a64(const void *data, size_t n) {
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint64_t)p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

/* ================================================================= */
/* Tier #1: in-process RAM LRU.                                       */
/* ================================================================= */

typedef struct ram_entry {
    uint64_t  key;
    uint32_t *tokens;
    uint32_t  n_tokens;
    void     *payload;
    size_t    payload_bytes;
    struct ram_entry *prev_lru;
    struct ram_entry *next_lru;
    struct ram_entry *next_bucket;
} ram_entry_t;

typedef struct {
    size_t        capacity_bytes;
    size_t        used_bytes;
    ram_entry_t **buckets;
    size_t        n_buckets;
    ram_entry_t   mru, lru;   /* sentinels */

    /* Working buffer for handing payload back to the cache without
     * giving away ownership: we just point at the entry's payload. */
} ram_state_t;

static void ram_entry_free(ram_entry_t *e) {
    if (!e) return;
    free(e->tokens);
    free(e->payload);
    free(e);
}

static void ram_lru_unlink(ram_entry_t *e) {
    e->prev_lru->next_lru = e->next_lru;
    e->next_lru->prev_lru = e->prev_lru;
    e->prev_lru = e->next_lru = NULL;
}

static void ram_lru_push_front(ram_state_t *s, ram_entry_t *e) {
    e->next_lru = s->mru.next_lru;
    e->prev_lru = &s->mru;
    s->mru.next_lru->prev_lru = e;
    s->mru.next_lru = e;
}

static void ram_bucket_link(ram_state_t *s, ram_entry_t *e) {
    size_t b = (size_t)(e->key % s->n_buckets);
    e->next_bucket = s->buckets[b];
    s->buckets[b] = e;
}

static void ram_bucket_unlink(ram_state_t *s, ram_entry_t *e) {
    size_t b = (size_t)(e->key % s->n_buckets);
    ram_entry_t **prev = &s->buckets[b];
    while (*prev) {
        if (*prev == e) { *prev = e->next_bucket; return; }
        prev = &(*prev)->next_bucket;
    }
}

static size_t ram_entry_bytes(const ram_entry_t *e) {
    return sizeof(*e) + (size_t)e->n_tokens * sizeof(uint32_t) + e->payload_bytes;
}

static int ram_get(qw36_kv_tier *t, uint64_t key,
                   const uint32_t *tokens, uint32_t n_tokens,
                   const void **out_buf, size_t *out_bytes) {
    ram_state_t *s = (ram_state_t *)t->state;
    if (!s->n_buckets) return 0;
    size_t b = (size_t)(key % s->n_buckets);
    for (ram_entry_t *e = s->buckets[b]; e; e = e->next_bucket) {
        if (e->key == key && e->n_tokens == n_tokens &&
            memcmp(e->tokens, tokens, n_tokens * sizeof(uint32_t)) == 0) {
            ram_lru_unlink(e);
            ram_lru_push_front(s, e);
            if (out_buf)   *out_buf   = e->payload;
            if (out_bytes) *out_bytes = e->payload_bytes;
            return 1;
        }
    }
    return 0;
}

static void ram_evict_to_fit(ram_state_t *s, size_t incoming) {
    if (!s->capacity_bytes) return;
    while (s->used_bytes + incoming > s->capacity_bytes &&
           s->lru.prev_lru != &s->mru) {
        ram_entry_t *victim = s->lru.prev_lru;
        if (victim == &s->mru) break;
        ram_lru_unlink(victim);
        ram_bucket_unlink(s, victim);
        s->used_bytes -= ram_entry_bytes(victim);
        ram_entry_free(victim);
    }
}

static int ram_put(qw36_kv_tier *t, uint64_t key,
                   const uint32_t *tokens, uint32_t n_tokens,
                   const void *buf, size_t bytes) {
    ram_state_t *s = (ram_state_t *)t->state;
    if (!s->capacity_bytes) return 1; /* tier disabled, not an error */
    /* Overwrite if existing key. */
    if (s->n_buckets) {
        size_t b = (size_t)(key % s->n_buckets);
        for (ram_entry_t *e = s->buckets[b]; e; e = e->next_bucket) {
            if (e->key == key && e->n_tokens == n_tokens &&
                memcmp(e->tokens, tokens, n_tokens * sizeof(uint32_t)) == 0) {
                ram_lru_unlink(e);
                ram_lru_push_front(s, e);
                return 0;  /* already present, idempotent */
            }
        }
    }
    ram_entry_t *e = (ram_entry_t *)calloc(1, sizeof(*e));
    if (!e) return -1;
    e->key = key;
    e->n_tokens = n_tokens;
    e->tokens = (uint32_t *)malloc((size_t)n_tokens * sizeof(uint32_t));
    e->payload = bytes ? malloc(bytes) : NULL;
    if (!e->tokens || (bytes && !e->payload)) { ram_entry_free(e); return -1; }
    memcpy(e->tokens, tokens, (size_t)n_tokens * sizeof(uint32_t));
    if (bytes) memcpy(e->payload, buf, bytes);
    e->payload_bytes = bytes;

    size_t add = ram_entry_bytes(e);
    if (add > s->capacity_bytes) { ram_entry_free(e); return -1; }
    ram_evict_to_fit(s, add);
    ram_bucket_link(s, e);
    ram_lru_push_front(s, e);
    s->used_bytes += add;
    return 0;
}

static size_t ram_used(const qw36_kv_tier *t) {
    return ((const ram_state_t *)t->state)->used_bytes;
}

static size_t ram_cap(const qw36_kv_tier *t) {
    return ((const ram_state_t *)t->state)->capacity_bytes;
}

static void ram_clear(qw36_kv_tier *t) {
    ram_state_t *s = (ram_state_t *)t->state;
    for (ram_entry_t *e = s->mru.next_lru; e != &s->lru; ) {
        ram_entry_t *next = e->next_lru;
        ram_entry_free(e);
        e = next;
    }
    if (s->buckets) memset(s->buckets, 0, s->n_buckets * sizeof(ram_entry_t *));
    s->mru.next_lru = &s->lru;
    s->lru.prev_lru = &s->mru;
    s->used_bytes = 0;
}

static void ram_destroy(qw36_kv_tier *t) {
    if (!t) return;
    ram_clear(t);
    ram_state_t *s = (ram_state_t *)t->state;
    free(s->buckets);
    free(s);
    free(t);
}

static const qw36_kv_tier_ops ram_lru_ops = {
    .name = "ram_lru",
    .get = ram_get,
    .put = ram_put,
    .used_bytes = ram_used,
    .capacity_bytes = ram_cap,
    .clear = ram_clear,
    .destroy = ram_destroy,
};

qw36_kv_tier *qw36_kv_tier_ram_lru_create(size_t capacity_bytes) {
    qw36_kv_tier *t = (qw36_kv_tier *)calloc(1, sizeof(*t));
    ram_state_t  *s = (ram_state_t *)calloc(1, sizeof(*s));
    if (!t || !s) { free(t); free(s); return NULL; }
    s->capacity_bytes = capacity_bytes;
    s->n_buckets = 1024;
    s->buckets = (ram_entry_t **)calloc(s->n_buckets, sizeof(*s->buckets));
    if (!s->buckets) { free(s); free(t); return NULL; }
    s->mru.next_lru = &s->lru;
    s->lru.prev_lru = &s->mru;
    t->ops = &ram_lru_ops;
    t->state = s;
    return t;
}

/* ================================================================= */
/* Tier #2: on-disk store.                                            */
/* ================================================================= */

#define QW36_KV_MAGIC   0x57515646u  /* 'KVQW' little-endian */
#define QW36_KV_VERSION 1u

typedef struct {
    char  *dir;
    size_t capacity_bytes;        /* SIZE_MAX = no cap from us */
    /* The disk tier streams payloads through a single working buffer
     * so get() can return a stable pointer until the next call. The
     * pointer is invalidated by any subsequent get/put on this tier. */
    void  *work_buf;
    size_t work_cap;
    size_t work_used;
} disk_state_t;

static int ensure_dir(const char *dir) {
    struct stat st;
    if (stat(dir, &st) == 0) return S_ISDIR(st.st_mode) ? 0 : -1;
    return mkdir(dir, 0700);
}

static char *disk_path(const disk_state_t *s, uint64_t key) {
    size_t n = strlen(s->dir) + 64;
    char *p = (char *)malloc(n);
    if (!p) return NULL;
    snprintf(p, n, "%s/qw36_kv_%016llx.bin", s->dir, (unsigned long long)key);
    return p;
}

static int disk_get(qw36_kv_tier *t, uint64_t key,
                    const uint32_t *tokens, uint32_t n_tokens,
                    const void **out_buf, size_t *out_bytes) {
    disk_state_t *s = (disk_state_t *)t->state;
    if (!s->dir || !s->capacity_bytes) return 0;
    char *path = disk_path(s, key);
    if (!path) return 0;
    FILE *f = fopen(path, "rb");
    free(path);
    if (!f) return 0;

    uint32_t hdr[4];
    uint64_t bytes;
    if (fread(hdr, sizeof(hdr), 1, f) != 1 ||
        hdr[0] != QW36_KV_MAGIC || hdr[1] != QW36_KV_VERSION ||
        fread(&bytes, sizeof(bytes), 1, f) != 1) {
        fclose(f); return 0;
    }
    if (hdr[2] != n_tokens) { fclose(f); return 0; }

    /* Token vector for collision verification. */
    uint32_t *toks = (uint32_t *)malloc((size_t)n_tokens * sizeof(uint32_t));
    if (!toks) { fclose(f); return 0; }
    if (fread(toks, sizeof(uint32_t), n_tokens, f) != n_tokens ||
        memcmp(toks, tokens, n_tokens * sizeof(uint32_t)) != 0) {
        free(toks); fclose(f); return 0;
    }
    free(toks);

    /* Payload into the tier's reusable working buffer. */
    if (bytes > s->work_cap) {
        void *nb = realloc(s->work_buf, (size_t)bytes);
        if (!nb) { fclose(f); return 0; }
        s->work_buf = nb;
        s->work_cap = (size_t)bytes;
    }
    if (bytes && fread(s->work_buf, 1, (size_t)bytes, f) != (size_t)bytes) {
        fclose(f); return 0;
    }
    fclose(f);
    s->work_used = (size_t)bytes;
    if (out_buf)   *out_buf   = s->work_buf;
    if (out_bytes) *out_bytes = (size_t)bytes;
    return 1;
}

static int disk_put(qw36_kv_tier *t, uint64_t key,
                    const uint32_t *tokens, uint32_t n_tokens,
                    const void *buf, size_t bytes) {
    disk_state_t *s = (disk_state_t *)t->state;
    if (!s->dir || !s->capacity_bytes) return 1;
    if (ensure_dir(s->dir)) return -1;
    char *path = disk_path(s, key);
    if (!path) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) { free(path); return -1; }
    uint32_t hdr[4] = {QW36_KV_MAGIC, QW36_KV_VERSION, n_tokens, 0};
    uint64_t pb = (uint64_t)bytes;
    int ok = 1;
    ok &= fwrite(hdr, sizeof(hdr), 1, f) == 1;
    ok &= fwrite(&pb, sizeof(pb), 1, f) == 1;
    ok &= fwrite(tokens, sizeof(uint32_t), n_tokens, f) == n_tokens;
    if (bytes) ok &= fwrite(buf, 1, bytes, f) == bytes;
    fclose(f);
    free(path);
    /* No disk-eviction policy in v0: filesystems already have one
     * (LRU via atime, manual prune). Capacity is advisory. */
    return ok ? 0 : -1;
}

static size_t disk_used(const qw36_kv_tier *t) {
    /* Honest: we don't track. Returning 0 is fine for the cache
     * scheduler — capacity-based decisions on disk are out of scope
     * for v0. */
    (void)t;
    return 0;
}

static size_t disk_cap(const qw36_kv_tier *t) {
    const disk_state_t *s = (const disk_state_t *)t->state;
    return s->capacity_bytes;
}

static void disk_clear(qw36_kv_tier *t) {
    /* Intentional: clear() does NOT delete files. v0 leaves disk
     * cleanup to the operator. */
    (void)t;
}

static void disk_destroy(qw36_kv_tier *t) {
    if (!t) return;
    disk_state_t *s = (disk_state_t *)t->state;
    free(s->dir);
    free(s->work_buf);
    free(s);
    free(t);
}

static const qw36_kv_tier_ops disk_ops = {
    .name = "disk",
    .get = disk_get,
    .put = disk_put,
    .used_bytes = disk_used,
    .capacity_bytes = disk_cap,
    .clear = disk_clear,
    .destroy = disk_destroy,
};

qw36_kv_tier *qw36_kv_tier_disk_create(const char *dir,
                                       size_t capacity_bytes) {
    if (!dir || !dir[0] || !capacity_bytes) return NULL;
    qw36_kv_tier *t = (qw36_kv_tier *)calloc(1, sizeof(*t));
    disk_state_t *s = (disk_state_t *)calloc(1, sizeof(*s));
    if (!t || !s) { free(t); free(s); return NULL; }
    s->dir = strdup(dir);
    s->capacity_bytes = capacity_bytes;
    if (!s->dir) { free(s); free(t); return NULL; }
    ensure_dir(s->dir);
    t->ops = &disk_ops;
    t->state = s;
    return t;
}

/* ================================================================= */
/* Generic cache — composes tiers, hot first.                         */
/* ================================================================= */

#define QW36_KV_MAX_TIERS 8

struct qw36_kv_prefix_cache {
    qw36_kv_tier         *tiers[QW36_KV_MAX_TIERS];
    uint32_t              n_tiers;
    qw36_kv_cache_cfg     cfg;
    qw36_kv_cache_stats_t stats;
};

qw36_kv_prefix_cache *qw36_kv_cache_create(qw36_kv_tier **tiers,
                                           uint32_t n_tiers,
                                           const qw36_kv_cache_cfg *cfg) {
    if (!tiers || !n_tiers || n_tiers > QW36_KV_MAX_TIERS) return NULL;
    for (uint32_t i = 0; i < n_tiers; i++) if (!tiers[i]) return NULL;

    qw36_kv_prefix_cache *c = (qw36_kv_prefix_cache *)calloc(1, sizeof(*c));
    if (!c) return NULL;
    for (uint32_t i = 0; i < n_tiers; i++) c->tiers[i] = tiers[i];
    c->n_tiers = n_tiers;
    c->stats.n_tiers = n_tiers;
    if (cfg) c->cfg = *cfg;
    if (!c->cfg.min_prefix_tokens) c->cfg.min_prefix_tokens = 32;
    if (!c->cfg.writeback_on_evict) c->cfg.writeback_on_evict = 1;
    return c;
}

qw36_kv_prefix_cache *qw36_kv_cache_create_default(size_t ram_bytes,
                                                   const char *disk_dir,
                                                   size_t disk_bytes,
                                                   const qw36_kv_cache_cfg *cfg) {
    qw36_kv_tier *built[2];
    uint32_t n = 0;
    if (ram_bytes) {
        qw36_kv_tier *r = qw36_kv_tier_ram_lru_create(ram_bytes);
        if (!r) return NULL;
        built[n++] = r;
    }
    if (disk_dir && disk_dir[0] && disk_bytes) {
        qw36_kv_tier *d = qw36_kv_tier_disk_create(disk_dir, disk_bytes);
        if (!d) {
            for (uint32_t i = 0; i < n; i++) built[i]->ops->destroy(built[i]);
            return NULL;
        }
        built[n++] = d;
    }
    if (!n) return NULL;
    return qw36_kv_cache_create(built, n, cfg);
}

void qw36_kv_cache_destroy(qw36_kv_prefix_cache *c) {
    if (!c) return;
    for (uint32_t i = 0; i < c->n_tiers; i++)
        c->tiers[i]->ops->destroy(c->tiers[i]);
    free(c);
}

void qw36_kv_cache_clear(qw36_kv_prefix_cache *c) {
    if (!c) return;
    for (uint32_t i = 0; i < c->n_tiers; i++)
        c->tiers[i]->ops->clear(c->tiers[i]);
}

void qw36_kv_cache_stats(const qw36_kv_prefix_cache *c,
                         qw36_kv_cache_stats_t *out) {
    if (!c || !out) return;
    *out = c->stats;
    for (uint32_t i = 0; i < c->n_tiers; i++) {
        out->bytes_by_tier[i] = c->tiers[i]->ops->used_bytes(c->tiers[i]);
    }
}

/* Lookup walks tiers hot→cold. On hit promotes the entry to all
 * hotter tiers (so the next lookup is faster). Returns matched
 * prefix length. */
size_t qw36_kv_cache_lookup(qw36_kv_prefix_cache *c,
                            const uint32_t *tokens, size_t n_tokens,
                            const void **out_buf, size_t *out_bytes) {
    if (!c || !tokens || !n_tokens) return 0;
    c->stats.lookups++;

    uint32_t min_n = c->cfg.min_prefix_tokens;
    if (n_tokens < min_n) { c->stats.misses++; return 0; }

    for (uint32_t n = (uint32_t)n_tokens; n >= min_n; n--) {
        uint64_t key = fnv1a64(tokens, (size_t)n * sizeof(uint32_t));
        for (uint32_t i = 0; i < c->n_tiers; i++) {
            const void *buf = NULL;
            size_t bytes = 0;
            if (c->tiers[i]->ops->get(c->tiers[i], key, tokens, n,
                                       &buf, &bytes)) {
                c->stats.hits_by_tier[i]++;
                /* Promote into all hotter tiers (best-effort). */
                for (uint32_t j = 0; j < i; j++) {
                    if (c->tiers[j]->ops->put(c->tiers[j], key,
                                              tokens, n, buf, bytes) == 0)
                        c->stats.promotions++;
                }
                if (out_buf)   *out_buf   = buf;
                if (out_bytes) *out_bytes = bytes;
                return (size_t)n;
            }
        }
    }
    c->stats.misses++;
    return 0;
}

/* Insert into the hot tier. On writeback_on_evict the put cascade is
 * a per-tier concern (the ram_lru tier doesn't currently call back
 * into us on evict — v0 keeps the API one-way). A future tier could
 * implement evict-with-writeback by calling the next tier directly. */
int qw36_kv_cache_insert(qw36_kv_prefix_cache *c,
                         const uint32_t *tokens, size_t n_tokens,
                         const void *buf, size_t bytes) {
    if (!c || !tokens || !n_tokens) return -1;
    if (n_tokens < c->cfg.min_prefix_tokens) return 0;
    c->stats.inserts++;
    uint64_t key = fnv1a64(tokens, n_tokens * sizeof(uint32_t));
    int rc = c->tiers[0]->ops->put(c->tiers[0], key,
                                   tokens, (uint32_t)n_tokens, buf, bytes);
    if (rc != 0) return rc;
    /* Optional eager writethrough to all colder tiers (mirrors data
     * down; cheap on RAM-only but actual disk write is fsync-cost). */
    if (c->cfg.writeback_on_evict) {
        for (uint32_t i = 1; i < c->n_tiers; i++) {
            if (c->tiers[i]->ops->put(c->tiers[i], key,
                                      tokens, (uint32_t)n_tokens,
                                      buf, bytes) == 0) {
                c->stats.writebacks++;
            }
        }
    }
    return 0;
}
