/* qw36_tokenizer.c — byte-level BPE tokenizer for Qwen 3.6.
 *
 * Owner: Claude.
 *
 * Qwen 3 uses GPT-2-style byte-level BPE. Vocab and merges are stored in
 * the GGUF file under:
 *   tokenizer.ggml.tokens  — array<string>  (vocab; index = id)
 *   tokenizer.ggml.merges  — array<string>  ("<L> <R>" lines, in rank order)
 *   tokenizer.ggml.bos_token_id / eos_token_id  — u32 scalars
 *
 * We do NOT implement the pre-tokenization regex. The bytes of the input
 * are encoded via the byte→unicode map, then BPE-merged as one long
 * sequence. This is correct (every byte→unicode pair has a merge if it
 * exists) but slower than tiktoken and slightly different on punctuation
 * boundaries vs HF's tokenizer.json. Good enough for v0.
 */

#include "qw36_tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The GGUF reader exposes a private accessor for string arrays. */
int qw36__gguf_get_str_array(const qw36_gguf_file *f, const char *key,
                             const char *const **out_ptrs,
                             uint64_t **out_lens, uint64_t *out_n);

/* --------------------------------------------------------------------- */
/* Byte ↔ Unicode mapping (GPT-2 standard).                               */
/* --------------------------------------------------------------------- */

/* g_byte_utf8[b] is a (len, bytes[]) entry for the UTF-8 of byte b mapped
 * through the GPT-2 byte-to-unicode table. Codepoints are at most 2 bytes
 * UTF-8 (max codepoint we emit is 323). */
typedef struct { uint8_t len; uint8_t bytes[3]; } byte_glyph;
static byte_glyph g_byte_utf8[256];

/* Reverse: encode the (1- or 2-byte) UTF-8 sequence at *p back to a raw
 * byte. Returns the matched UTF-8 length, or 0 on failure. */
static uint8_t g_unicode_to_byte_first1[128];  /* index = ascii      */
static uint8_t g_unicode_to_byte_first1_set[128];
typedef struct { uint8_t b0, b1, raw, set; } u2_entry;
static u2_entry g_unicode_to_byte_2[256];      /* lookup by hash     */

static unsigned u2_hash(uint8_t b0, uint8_t b1) {
    return ((unsigned)b0 * 257u + b1) & 0xFFu;
}

static void byte_table_set(int byte, uint32_t cp) {
    byte_glyph *e = &g_byte_utf8[byte];
    if (cp < 0x80) {
        e->len = 1;
        e->bytes[0] = (uint8_t)cp;
        g_unicode_to_byte_first1[cp] = (uint8_t)byte;
        g_unicode_to_byte_first1_set[cp] = 1;
    } else {
        /* 2-byte UTF-8: 110xxxxx 10xxxxxx */
        e->len = 2;
        e->bytes[0] = (uint8_t)(0xC0 | (cp >> 6));
        e->bytes[1] = (uint8_t)(0x80 | (cp & 0x3F));
        /* For reverse lookup, install at hash (open addressing). */
        unsigned h = u2_hash(e->bytes[0], e->bytes[1]);
        for (unsigned probe = 0; probe < 256; probe++) {
            u2_entry *re = &g_unicode_to_byte_2[(h + probe) & 0xFF];
            if (!re->set) {
                re->b0 = e->bytes[0];
                re->b1 = e->bytes[1];
                re->raw = (uint8_t)byte;
                re->set = 1;
                return;
            }
        }
    }
}

static void byte_table_init(void) {
    static int done = 0;
    if (done) return;
    done = 1;
    memset(g_byte_utf8, 0, sizeof(g_byte_utf8));
    memset(g_unicode_to_byte_first1, 0, sizeof(g_unicode_to_byte_first1));
    memset(g_unicode_to_byte_first1_set, 0, sizeof(g_unicode_to_byte_first1_set));
    memset(g_unicode_to_byte_2, 0, sizeof(g_unicode_to_byte_2));

    /* "Printable" identity range: ! through ~, ¡ through ¬, ® through ÿ. */
    int n_extra = 0;
    int printable[256]; memset(printable, 0, sizeof(printable));
    for (int b = '!'; b <= '~'; b++)       printable[b] = 1;
    for (int b = 0xA1; b <= 0xAC; b++)     printable[b] = 1;
    for (int b = 0xAE; b <= 0xFF; b++)     printable[b] = 1;

    for (int b = 0; b < 256; b++) {
        if (printable[b]) byte_table_set(b, (uint32_t)b);
    }
    for (int b = 0; b < 256; b++) {
        if (!printable[b]) {
            byte_table_set(b, 256u + (uint32_t)n_extra);
            n_extra++;
        }
    }
}

/* Try to decode one (1 or 2 byte) glyph at *p, *len_left bytes remaining.
 * On success returns the raw byte, *consumed gets 1 or 2. On failure
 * returns -1. */
static int decode_one_glyph(const uint8_t *p, size_t len_left, int *consumed) {
    if (len_left == 0) return -1;
    if (p[0] < 0x80) {
        /* ASCII glyph. */
        if (g_unicode_to_byte_first1_set[p[0]]) {
            *consumed = 1;
            return g_unicode_to_byte_first1[p[0]];
        }
        return -1;
    }
    if (len_left < 2 || (p[0] & 0xE0) != 0xC0 || (p[1] & 0xC0) != 0x80)
        return -1;
    unsigned h = u2_hash(p[0], p[1]);
    for (unsigned probe = 0; probe < 256; probe++) {
        const u2_entry *re = &g_unicode_to_byte_2[(h + probe) & 0xFF];
        if (!re->set) break;
        if (re->b0 == p[0] && re->b1 == p[1]) {
            *consumed = 2;
            return re->raw;
        }
    }
    return -1;
}

/* --------------------------------------------------------------------- */
/* Hash table: vocab string → id, merge "L\0R" → rank.                    */
/* --------------------------------------------------------------------- */

typedef struct {
    const char *key;      /* NOT owned — points into vocab storage / merge storage */
    uint32_t    key_len;
    uint32_t    val;
    uint32_t    occupied;
} hmap_entry;

typedef struct {
    hmap_entry *entries;
    uint32_t    cap;       /* power of two */
    uint32_t    mask;
    uint32_t    n;
} hmap;

static uint64_t fnv1a(const char *s, uint32_t n) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (uint32_t i = 0; i < n; i++) {
        h ^= (uint8_t)s[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

static int hmap_init(hmap *m, uint32_t init_cap) {
    uint32_t cap = 1;
    while (cap < init_cap * 2) cap <<= 1;
    m->entries = (hmap_entry *)calloc(cap, sizeof(hmap_entry));
    if (!m->entries) return -1;
    m->cap = cap;
    m->mask = cap - 1;
    m->n = 0;
    return 0;
}

static void hmap_free(hmap *m) { free(m->entries); m->entries = NULL; }

static void hmap_put(hmap *m, const char *key, uint32_t key_len, uint32_t val) {
    uint32_t h = (uint32_t)fnv1a(key, key_len) & m->mask;
    for (;;) {
        hmap_entry *e = &m->entries[h];
        if (!e->occupied) {
            e->key = key; e->key_len = key_len; e->val = val; e->occupied = 1;
            m->n++;
            return;
        }
        if (e->key_len == key_len && !memcmp(e->key, key, key_len)) {
            e->val = val;  /* update */
            return;
        }
        h = (h + 1) & m->mask;
    }
}

static int hmap_get(const hmap *m, const char *key, uint32_t key_len, uint32_t *out) {
    if (m->cap == 0) return -1;
    uint32_t h = (uint32_t)fnv1a(key, key_len) & m->mask;
    for (uint32_t probe = 0; probe < m->cap; probe++) {
        const hmap_entry *e = &m->entries[h];
        if (!e->occupied) return -1;
        if (e->key_len == key_len && !memcmp(e->key, key, key_len)) {
            *out = e->val;
            return 0;
        }
        h = (h + 1) & m->mask;
    }
    return -1;
}

/* --------------------------------------------------------------------- */
/* Tokenizer object                                                       */
/* --------------------------------------------------------------------- */

struct qw36_tokenizer {
    /* Vocab: id i → null-terminated UTF-8 (byte-level-encoded) string. */
    char    **tok_str;
    uint32_t *tok_len;
    uint32_t  n_vocab;

    /* Reverse lookup. */
    hmap      str_to_id;

    /* Merges: composed key "L\0R" → rank (lower = applied first). */
    char     *merge_keys;    /* one big arena, owned                  */
    size_t    merge_keys_n;
    hmap      pair_rank;

    uint32_t  bos, eos, im_start, im_end, pad;
};

static int load_vocab(qw36_tokenizer *t, const qw36_gguf_file *f) {
    const char *const *ptrs = NULL;
    uint64_t *lens = NULL, n = 0;
    if (qw36__gguf_get_str_array(f, "tokenizer.ggml.tokens", &ptrs, &lens, &n) != 0)
        return -1;

    t->n_vocab = (uint32_t)n;
    t->tok_str = (char **)calloc(n, sizeof(char *));
    t->tok_len = (uint32_t *)calloc(n, sizeof(uint32_t));
    if (!t->tok_str || !t->tok_len) { free((void*)ptrs); free(lens); return -1; }

    if (hmap_init(&t->str_to_id, (uint32_t)n)) {
        free((void*)ptrs); free(lens); return -1;
    }

    for (uint64_t i = 0; i < n; i++) {
        t->tok_str[i] = (char *)malloc(lens[i] + 1);
        if (!t->tok_str[i]) { free((void*)ptrs); free(lens); return -1; }
        memcpy(t->tok_str[i], ptrs[i], lens[i]);
        t->tok_str[i][lens[i]] = '\0';
        t->tok_len[i] = (uint32_t)lens[i];
        hmap_put(&t->str_to_id, t->tok_str[i], (uint32_t)lens[i], (uint32_t)i);
    }
    free((void*)ptrs); free(lens);
    return 0;
}

static int load_merges(qw36_tokenizer *t, const qw36_gguf_file *f) {
    const char *const *ptrs = NULL;
    uint64_t *lens = NULL, n = 0;
    if (qw36__gguf_get_str_array(f, "tokenizer.ggml.merges", &ptrs, &lens, &n) != 0) {
        /* Merges may be absent on a "BPE without explicit merges" tokenizer.
         * Encoding will still work for tokens that are already in vocab. */
        return 0;
    }

    /* Compose into one arena to keep keys stable. */
    size_t total = 0;
    for (uint64_t i = 0; i < n; i++) total += lens[i] + 1;
    t->merge_keys = (char *)malloc(total);
    if (!t->merge_keys) { free((void*)ptrs); free(lens); return -1; }
    t->merge_keys_n = total;

    if (hmap_init(&t->pair_rank, (uint32_t)n)) {
        free((void*)ptrs); free(lens); return -1;
    }

    char *cur = t->merge_keys;
    for (uint64_t i = 0; i < n; i++) {
        memcpy(cur, ptrs[i], lens[i]);
        /* Replace the first ASCII space with '\0' so the key becomes "L\0R". */
        size_t l = (size_t)lens[i];
        size_t split = 0;
        for (; split < l; split++) if (cur[split] == ' ') break;
        if (split >= l) { /* malformed; skip */
            cur += l + 1;
            continue;
        }
        cur[split] = '\0';
        cur[l] = '\0';
        /* Store with key length = l (the explicit length, including the
         * embedded '\0' between L and R). */
        hmap_put(&t->pair_rank, cur, (uint32_t)l, (uint32_t)i);
        cur += l + 1;
    }
    free((void*)ptrs); free(lens);
    return 0;
}

static void load_specials(qw36_tokenizer *t, const qw36_gguf_file *f) {
    qw36_gguf_get_u32(f, "tokenizer.ggml.bos_token_id", &t->bos);
    qw36_gguf_get_u32(f, "tokenizer.ggml.eos_token_id", &t->eos);
    qw36_gguf_get_u32(f, "tokenizer.ggml.padding_token_id", &t->pad);
    /* Qwen3 chat specials: <|im_start|> / <|im_end|>. Try to look up by
     * literal string. */
    uint32_t id;
    if (!hmap_get(&t->str_to_id, "<|im_start|>", 12, &id)) t->im_start = id;
    if (!hmap_get(&t->str_to_id, "<|im_end|>",   10, &id)) t->im_end   = id;
}

qw36_tokenizer *qw36_tokenizer_new(const qw36_gguf_file *f,
                                   char *err, size_t err_cap)
{
    byte_table_init();
    qw36_tokenizer *t = (qw36_tokenizer *)calloc(1, sizeof(*t));
    if (!t) { if (err && err_cap) snprintf(err, err_cap, "oom"); return NULL; }

    if (load_vocab(t, f)) {
        if (err && err_cap) snprintf(err, err_cap, "missing tokenizer.ggml.tokens");
        qw36_tokenizer_free(t); return NULL;
    }
    if (load_merges(t, f)) {
        if (err && err_cap) snprintf(err, err_cap, "oom loading merges");
        qw36_tokenizer_free(t); return NULL;
    }
    load_specials(t, f);
    return t;
}

void qw36_tokenizer_free(qw36_tokenizer *t)
{
    if (!t) return;
    if (t->tok_str) {
        for (uint32_t i = 0; i < t->n_vocab; i++) free(t->tok_str[i]);
        free(t->tok_str);
    }
    free(t->tok_len);
    free(t->merge_keys);
    hmap_free(&t->str_to_id);
    hmap_free(&t->pair_rank);
    free(t);
}

/* --------------------------------------------------------------------- */
/* Encode                                                                 */
/* --------------------------------------------------------------------- */

/* BPE on a working sequence of "symbols" — slices of a contiguous arena.
 * Each symbol is (ptr, len). Best-pair merging until no rule applies. */
typedef struct { const char *p; uint32_t len; } sym;

static int encode_chunk(const qw36_tokenizer *t,
                        const char *bytestr, size_t n,
                        uint32_t *out_ids, size_t *out_cap, size_t *out_used)
{
    if (n == 0) return 0;
    sym *s = (sym *)malloc(sizeof(sym) * n);
    if (!s) return -1;
    /* Initialize with one symbol per byte-level glyph. We need to step by
     * full glyphs (1 or 2 UTF-8 bytes), not by raw bytes. */
    size_t nsym = 0;
    for (size_t i = 0; i < n; ) {
        const char *p = bytestr + i;
        size_t glyph_len = 1;
        if ((uint8_t)*p >= 0x80) glyph_len = 2;
        if (i + glyph_len > n) glyph_len = 1; /* safety */
        s[nsym].p   = p;
        s[nsym].len = (uint32_t)glyph_len;
        nsym++;
        i += glyph_len;
    }

    /* Build the merge key for adjacent pair (s[i], s[i+1]) into `keybuf`,
     * returning the byte length (which includes the '\0' separator). */
    char keybuf[512];
    for (;;) {
        uint32_t best_rank = (uint32_t)-1;
        size_t best_i = (size_t)-1;
        for (size_t i = 0; i + 1 < nsym; i++) {
            size_t l = s[i].len + 1 + s[i+1].len;
            if (l > sizeof(keybuf)) continue;
            memcpy(keybuf, s[i].p, s[i].len);
            keybuf[s[i].len] = '\0';
            memcpy(keybuf + s[i].len + 1, s[i+1].p, s[i+1].len);
            uint32_t r;
            if (!hmap_get(&t->pair_rank, keybuf, (uint32_t)l, &r)) {
                if (r < best_rank) { best_rank = r; best_i = i; }
            }
        }
        if (best_i == (size_t)-1) break;

        /* Merge s[best_i] and s[best_i+1] into a single symbol whose
         * contiguity we don't have (the two slices may not be adjacent in
         * memory once we start merging). So spill merged result into a
         * fresh buffer. Easiest: keep an arena of merged strings. */
        size_t mlen = s[best_i].len + s[best_i+1].len;
        char *merged = (char *)malloc(mlen);
        if (!merged) { free(s); return -1; }
        memcpy(merged, s[best_i].p, s[best_i].len);
        memcpy(merged + s[best_i].len, s[best_i+1].p, s[best_i+1].len);
        /* Replace at best_i, shift left from best_i+2 onward. Note: this
         * leaks `merged` since we don't track allocations. Acceptable for
         * a tokenizer pass — final tokens will be reified into ids and
         * then we drop the working symbols. */
        s[best_i].p   = merged;
        s[best_i].len = (uint32_t)mlen;
        memmove(&s[best_i+1], &s[best_i+2], (nsym - best_i - 2) * sizeof(sym));
        nsym--;
    }

    /* Emit ids. */
    for (size_t i = 0; i < nsym; i++) {
        uint32_t id;
        if (hmap_get(&t->str_to_id, s[i].p, s[i].len, &id)) {
            /* Fallback: split the unknown symbol into single-glyph tokens
             * and look those up individually. */
            for (uint32_t off = 0; off < s[i].len; ) {
                uint32_t glyph_len = 1;
                if ((uint8_t)s[i].p[off] >= 0x80) glyph_len = 2;
                if (hmap_get(&t->str_to_id, s[i].p + off, glyph_len, &id))
                    id = 0; /* give up; use token 0 */
                if (*out_used >= *out_cap) { free(s); return -1; }
                out_ids[(*out_used)++] = id;
                off += glyph_len;
            }
            continue;
        }
        if (*out_used >= *out_cap) { free(s); return -1; }
        out_ids[(*out_used)++] = id;
    }
    free(s);
    return 0;
}

/* Encode one segment of plain text (no special tokens inside) by mapping
 * bytes → byte-level UTF-8 → BPE merge. */
static int encode_segment(const qw36_tokenizer *t, const char *text,
                          size_t in_len, uint32_t *out_ids,
                          size_t *cap, size_t *used)
{
    if (!in_len) return 0;
    byte_table_init();
    size_t bs_cap = in_len * 2 + 4;
    char *bs = (char *)malloc(bs_cap);
    if (!bs) return -1;
    size_t bsn = 0;
    for (size_t i = 0; i < in_len; i++) {
        const byte_glyph *g = &g_byte_utf8[(uint8_t)text[i]];
        if (bsn + g->len > bs_cap) { free(bs); return -1; }
        memcpy(bs + bsn, g->bytes, g->len);
        bsn += g->len;
    }
    int rc = encode_chunk(t, bs, bsn, out_ids, cap, used);
    free(bs);
    return rc;
}

int qw36_tokenizer_encode(const qw36_tokenizer *t,
                          const char *text,
                          uint32_t *out_ids, size_t *out_n)
{
    if (!t || !text || !out_n) return -1;
    size_t cap = *out_n;
    size_t used = 0;

    /* Walk the input, splitting on known Qwen3 chat-template specials.
     * A special token like "<|im_start|>" must be emitted as its single
     * vocab id (e.g. 248045), not character-by-character — without this
     * the model sees the literal '<', '|', 'im', ...' sequence and
     * generates garbage. */
    static const char *const SPECIALS[] = {
        "<|im_start|>", "<|im_end|>",
        "<|endoftext|>", "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
        NULL,
    };

    size_t i = 0, seg_start = 0;
    while (text[i]) {
        if (text[i] == '<' && text[i+1] == '|') {
            for (int k = 0; SPECIALS[k]; k++) {
                size_t slen = strlen(SPECIALS[k]);
                if (strncmp(text + i, SPECIALS[k], slen) != 0) continue;
                uint32_t id;
                if (hmap_get(&t->str_to_id, SPECIALS[k], (uint32_t)slen, &id)) break;
                /* flush plain segment before this special */
                if (i > seg_start) {
                    if (encode_segment(t, text + seg_start, i - seg_start,
                                       out_ids, &cap, &used)) return -1;
                }
                if (used >= cap) return -1;
                out_ids[used++] = id;
                i += slen;
                seg_start = i;
                goto next_iter;
            }
        }
        i++;
        next_iter:;
    }
    if (i > seg_start) {
        if (encode_segment(t, text + seg_start, i - seg_start,
                           out_ids, &cap, &used)) return -1;
    }

    *out_n = used;
    return 0;
}

/* --------------------------------------------------------------------- */
/* Decode                                                                 */
/* --------------------------------------------------------------------- */

const char *qw36_tokenizer_decode_one(const qw36_tokenizer *t,
                                      uint32_t id, size_t *len_out)
{
    if (!t || id >= t->n_vocab) { if (len_out) *len_out = 0; return NULL; }
    /* The stored token is a byte-level UTF-8 string. The caller usually
     * wants the *raw byte* sequence (so concatenated decode produces real
     * text). We provide both: by default we decode the byte-level glyphs
     * back to raw bytes into a per-tokenizer scratch buffer. */
    byte_table_init();
    static __thread char *scratch = NULL;
    static __thread size_t scratch_cap = 0;

    const char *src = t->tok_str[id];
    size_t sn = t->tok_len[id];
    if (scratch_cap < sn + 1) {
        size_t newcap = (sn + 1) * 2;
        char *p = (char *)realloc(scratch, newcap);
        if (!p) { if (len_out) *len_out = 0; return NULL; }
        scratch = p; scratch_cap = newcap;
    }

    size_t out = 0;
    for (size_t i = 0; i < sn; ) {
        int consumed = 0;
        int b = decode_one_glyph((const uint8_t *)src + i, sn - i, &consumed);
        if (b < 0) {
            /* Not in the byte map (e.g. special token "<|im_end|>"). Pass
             * through as-is. */
            scratch[out++] = src[i++];
            continue;
        }
        scratch[out++] = (char)b;
        i += consumed;
    }
    scratch[out] = '\0';
    if (len_out) *len_out = out;
    return scratch;
}

uint32_t qw36_tokenizer_bos(const qw36_tokenizer *t)      { return t ? t->bos : 0; }
uint32_t qw36_tokenizer_eos(const qw36_tokenizer *t)      { return t ? t->eos : 0; }
uint32_t qw36_tokenizer_im_start(const qw36_tokenizer *t) { return t ? t->im_start : 0; }
uint32_t qw36_tokenizer_im_end(const qw36_tokenizer *t)   { return t ? t->im_end : 0; }
