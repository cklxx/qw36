/* qw36_gguf.c — GGUF v3 reader.
 *
 * Owner: Claude.
 *
 * Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
 *
 * Layout:
 *   header  : magic "GGUF" (u32), version (u32, ==3), tensor_count (u64),
 *             metadata_kv_count (u64)
 *   kv*     : key(string) + value_type(u32) + value(varies)
 *   tensor* : name(string) + n_dims(u32) + dims[n_dims](u64)
 *             + ggml_type(u32) + offset(u64)
 *   padding : up to general.alignment (default 32)
 *   data    : tensor blobs, each at its own offset
 *
 * We mmap the file and keep pointers into it; nothing is copied except
 * key strings and array storage for kv entries.
 */

#include "qw36_gguf.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define GGUF_MAGIC      0x46554747u   /* "GGUF" little-endian */
#define GGUF_VERSION    3
#define GGUF_DEFAULT_ALIGN 32u

typedef enum {
    GGUF_T_UINT8   = 0,
    GGUF_T_INT8    = 1,
    GGUF_T_UINT16  = 2,
    GGUF_T_INT16   = 3,
    GGUF_T_UINT32  = 4,
    GGUF_T_INT32   = 5,
    GGUF_T_FLOAT32 = 6,
    GGUF_T_BOOL    = 7,
    GGUF_T_STRING  = 8,
    GGUF_T_ARRAY   = 9,
    GGUF_T_UINT64  = 10,
    GGUF_T_INT64   = 11,
    GGUF_T_FLOAT64 = 12
} gguf_value_type;

/* ggml type IDs we care about. */
enum {
    GGML_TYPE_F32   = 0,
    GGML_TYPE_F16   = 1,
    GGML_TYPE_Q4_0  = 2,
    GGML_TYPE_Q4_1  = 3,
    GGML_TYPE_Q8_0  = 8,
    GGML_TYPE_Q8_1  = 9,
    GGML_TYPE_Q2_K  = 10,
    GGML_TYPE_Q3_K  = 11,
    GGML_TYPE_Q4_K  = 12,
    GGML_TYPE_Q5_K  = 13,
    GGML_TYPE_Q6_K  = 14,
    GGML_TYPE_Q8_K  = 15,
    GGML_TYPE_BF16  = 30
};

typedef struct {
    char           *key;       /* null-terminated, owned */
    gguf_value_type type;
    /* For scalar types we copy into the union. For STRING / ARRAY we store
     * a pointer that aliases the mmap. */
    union {
        uint8_t  u8;  int8_t  i8;
        uint16_t u16; int16_t i16;
        uint32_t u32; int32_t i32;
        uint64_t u64; int64_t i64;
        float    f32; double  f64;
        uint8_t  b;
        struct { const char *ptr; uint64_t len; } str;
        struct { gguf_value_type elem_type;
                 uint64_t        n;
                 const void     *raw;  /* aliases mmap */
        } arr;
    } v;
} gguf_kv;

typedef struct {
    char       *name;
    uint32_t    n_dims;
    uint64_t    dims[4];
    uint32_t    ggml_type;
    uint64_t    offset;      /* relative to tensor-data section */
    const void *data;        /* absolute pointer into mmap */
    size_t      bytes;
} gguf_tensor_info;

struct qw36_gguf_file {
    int               fd;
    const uint8_t    *map;
    size_t            map_len;

    gguf_kv          *kvs;
    size_t            n_kv;

    gguf_tensor_info *tensors;
    size_t            n_tensors;

    uint32_t          alignment;
    const uint8_t    *data_start;
};

/* --------------------------------------------------------------------- */
/* Cursor helpers — bounds-checked little-endian reads.                   */
/* --------------------------------------------------------------------- */

typedef struct {
    const uint8_t *p;
    const uint8_t *end;
    int           overflow;
} cur_t;

static int cur_avail(const cur_t *c, size_t n) {
    return !c->overflow && (size_t)(c->end - c->p) >= n;
}

static void cur_skip(cur_t *c, size_t n) {
    if (!cur_avail(c, n)) { c->overflow = 1; return; }
    c->p += n;
}

#define DEF_READ(T, NAME)                                       \
    static T NAME(cur_t *c) {                                   \
        if (!cur_avail(c, sizeof(T))) { c->overflow = 1; return (T)0; } \
        T v; memcpy(&v, c->p, sizeof(T)); c->p += sizeof(T);    \
        return v;                                               \
    }

DEF_READ(uint8_t,  rd_u8)
DEF_READ(int8_t,   rd_i8)
DEF_READ(uint16_t, rd_u16)
DEF_READ(int16_t,  rd_i16)
DEF_READ(uint32_t, rd_u32)
DEF_READ(int32_t,  rd_i32)
DEF_READ(uint64_t, rd_u64)
DEF_READ(int64_t,  rd_i64)
DEF_READ(float,    rd_f32)
DEF_READ(double,   rd_f64)

static char *rd_string(cur_t *c) {
    uint64_t n = rd_u64(c);
    if (c->overflow || !cur_avail(c, n)) { c->overflow = 1; return NULL; }
    char *s = (char *)malloc(n + 1);
    if (!s) { c->overflow = 1; return NULL; }
    memcpy(s, c->p, n);
    s[n] = '\0';
    c->p += n;
    return s;
}

/* Like rd_string but returns a pointer aliasing the mmap (no copy). */
static int rd_string_view(cur_t *c, const char **out_ptr, uint64_t *out_len) {
    uint64_t n = rd_u64(c);
    if (c->overflow || !cur_avail(c, n)) { c->overflow = 1; return -1; }
    *out_ptr = (const char *)c->p;
    *out_len = n;
    c->p += n;
    return 0;
}

/* --------------------------------------------------------------------- */
/* Element size of a GGUF scalar value type (0 for STRING/ARRAY).         */
/* --------------------------------------------------------------------- */
static size_t scalar_size(gguf_value_type t) {
    switch (t) {
        case GGUF_T_UINT8: case GGUF_T_INT8: case GGUF_T_BOOL: return 1;
        case GGUF_T_UINT16: case GGUF_T_INT16:               return 2;
        case GGUF_T_UINT32: case GGUF_T_INT32: case GGUF_T_FLOAT32: return 4;
        case GGUF_T_UINT64: case GGUF_T_INT64: case GGUF_T_FLOAT64: return 8;
        default: return 0;
    }
}

/* KEEP — skip a value of a given type in the cursor without copying.
 * Currently unused because the loader pre-allocates every value, but
 * left in for the future "only-read-the-keys-we-care-about" pass that
 * would cut load time for huge GGUFs by skipping irrelevant metadata. */
static void skip_value(cur_t *c, gguf_value_type t) {
    size_t s = scalar_size(t);
    if (s) { cur_skip(c, s); return; }
    if (t == GGUF_T_STRING) {
        uint64_t n = rd_u64(c);
        cur_skip(c, n);
        return;
    }
    if (t == GGUF_T_ARRAY) {
        gguf_value_type et = (gguf_value_type)rd_u32(c);
        uint64_t n = rd_u64(c);
        size_t es = scalar_size(et);
        if (es) { cur_skip(c, n * es); return; }
        if (et == GGUF_T_STRING) {
            for (uint64_t i = 0; i < n && !c->overflow; i++) {
                uint64_t l = rd_u64(c);
                cur_skip(c, l);
            }
            return;
        }
        /* nested arrays — punt */
        c->overflow = 1;
        return;
    }
    c->overflow = 1;
}

/* --------------------------------------------------------------------- */
/* Read a single KV value into kv->v.                                     */
/* --------------------------------------------------------------------- */
static int read_value(cur_t *c, gguf_kv *kv) {
    kv->type = (gguf_value_type)rd_u32(c);
    if (c->overflow) return -1;
    switch (kv->type) {
        case GGUF_T_UINT8:   kv->v.u8  = rd_u8(c);  break;
        case GGUF_T_INT8:    kv->v.i8  = rd_i8(c);  break;
        case GGUF_T_UINT16:  kv->v.u16 = rd_u16(c); break;
        case GGUF_T_INT16:   kv->v.i16 = rd_i16(c); break;
        case GGUF_T_UINT32:  kv->v.u32 = rd_u32(c); break;
        case GGUF_T_INT32:   kv->v.i32 = rd_i32(c); break;
        case GGUF_T_FLOAT32: kv->v.f32 = rd_f32(c); break;
        case GGUF_T_BOOL:    kv->v.b   = rd_u8(c);  break;
        case GGUF_T_UINT64:  kv->v.u64 = rd_u64(c); break;
        case GGUF_T_INT64:   kv->v.i64 = rd_i64(c); break;
        case GGUF_T_FLOAT64: kv->v.f64 = rd_f64(c); break;
        case GGUF_T_STRING:
            if (rd_string_view(c, &kv->v.str.ptr, &kv->v.str.len)) return -1;
            break;
        case GGUF_T_ARRAY: {
            kv->v.arr.elem_type = (gguf_value_type)rd_u32(c);
            kv->v.arr.n         = rd_u64(c);
            kv->v.arr.raw       = c->p;
            /* Skip the array body so subsequent reads start past it.
             * Storing raw lets accessors decode lazily. */
            size_t es = scalar_size(kv->v.arr.elem_type);
            if (es) {
                cur_skip(c, kv->v.arr.n * es);
            } else if (kv->v.arr.elem_type == GGUF_T_STRING) {
                for (uint64_t i = 0; i < kv->v.arr.n && !c->overflow; i++) {
                    uint64_t l = rd_u64(c);
                    cur_skip(c, l);
                }
            } else {
                c->overflow = 1;
            }
            break;
        }
        default:
            c->overflow = 1;
            return -1;
    }
    return c->overflow ? -1 : 0;
}

/* --------------------------------------------------------------------- */
/* Tensor byte-size given ggml type + element count. Quantized formats   */
/* are block-based; we only handle the dense ones here and return 0 for  */
/* unsupported types (caller treats as opaque blob).                     */
/* --------------------------------------------------------------------- */

#define QK_K       256
#define QK4_0      32
#define QK4_1      32
#define QK8_0      32
#define QK8_1      32

static size_t tensor_bytes(uint32_t ggml_type, uint64_t numel) {
    switch (ggml_type) {
        case GGML_TYPE_F32:  return numel * 4;
        case GGML_TYPE_F16:  return numel * 2;
        case GGML_TYPE_BF16: return numel * 2;
        /* Block layouts from ggml-quants.h (sizes only). */
        case GGML_TYPE_Q4_0: return (numel / QK4_0) * (2 + QK4_0/2);
        case GGML_TYPE_Q4_1: return (numel / QK4_1) * (2 + 2 + QK4_1/2);
        case GGML_TYPE_Q8_0: return (numel / QK8_0) * (2 + QK8_0);
        case GGML_TYPE_Q8_1: return (numel / QK8_1) * (2 + 2 + QK8_1);
        case GGML_TYPE_Q2_K: return (numel / QK_K) * (QK_K/16 + QK_K/4 + 2 + 2);
        case GGML_TYPE_Q3_K: return (numel / QK_K) * (QK_K/8 + QK_K/4 + 12 + 2);
        case GGML_TYPE_Q4_K: return (numel / QK_K) * (2 + 2 + 12 + QK_K/2);
        case GGML_TYPE_Q5_K: return (numel / QK_K) * (2 + 2 + 12 + QK_K/8 + QK_K/2);
        case GGML_TYPE_Q6_K: return (numel / QK_K) * (QK_K/2 + QK_K/4 + QK_K/16 + 2);
        case GGML_TYPE_Q8_K: return (numel / QK_K) * (4 + QK_K + QK_K/16 * 2);
        default: return 0;
    }
}

/* Map ggml type → qw36 dtype enum. Unknown ⇒ QW36_DTYPE_UNSUPPORTED. */
static qw36_dtype ggml_to_qw36(uint32_t t) {
    switch (t) {
        case GGML_TYPE_F32:  return QW36_DTYPE_F32;
        case GGML_TYPE_F16:  return QW36_DTYPE_F16;
        case GGML_TYPE_BF16: return QW36_DTYPE_BF16;
        case GGML_TYPE_Q8_0: return QW36_DTYPE_Q8_0;
        case GGML_TYPE_Q4_K: return QW36_DTYPE_Q4_K;
        case GGML_TYPE_Q5_K: return QW36_DTYPE_Q5_K;
        case GGML_TYPE_Q6_K: return QW36_DTYPE_Q6_K;
        case GGML_TYPE_Q3_K: return QW36_DTYPE_Q3_K;
        case GGML_TYPE_Q2_K: return QW36_DTYPE_Q2_K;
        default:             return QW36_DTYPE_UNSUPPORTED;
    }
}

/* --------------------------------------------------------------------- */
/* Public API                                                             */
/* --------------------------------------------------------------------- */

static void file_destroy(qw36_gguf_file *f) {
    if (!f) return;
    if (f->kvs) {
        for (size_t i = 0; i < f->n_kv; i++) free(f->kvs[i].key);
        free(f->kvs);
    }
    if (f->tensors) {
        for (size_t i = 0; i < f->n_tensors; i++) free(f->tensors[i].name);
        free(f->tensors);
    }
    if (f->map && f->map != MAP_FAILED) munmap((void *)f->map, f->map_len);
    if (f->fd >= 0) close(f->fd);
    free(f);
}

qw36_gguf_file *qw36_gguf_open(const char *path, char *err, size_t err_cap)
{
    qw36_gguf_file *f = (qw36_gguf_file *)calloc(1, sizeof(*f));
    if (!f) { if (err && err_cap) snprintf(err, err_cap, "oom"); return NULL; }
    f->fd = -1;
    f->alignment = GGUF_DEFAULT_ALIGN;

    f->fd = open(path, O_RDONLY);
    if (f->fd < 0) {
        int e = errno;
        if (err && err_cap) snprintf(err, err_cap, "open(%s): %s",
                                     path, strerror(e));
        free(f); return NULL;
    }
    struct stat st;
    if (fstat(f->fd, &st) < 0) {
        if (err && err_cap) snprintf(err, err_cap, "fstat: %s", strerror(errno));
        file_destroy(f); return NULL;
    }
    f->map_len = (size_t)st.st_size;
    f->map = (const uint8_t *)mmap(NULL, f->map_len, PROT_READ, MAP_PRIVATE, f->fd, 0);
    if (f->map == MAP_FAILED) {
        if (err && err_cap) snprintf(err, err_cap, "mmap: %s", strerror(errno));
        file_destroy(f); return NULL;
    }

    cur_t c = { f->map, f->map + f->map_len, 0 };

    uint32_t magic   = rd_u32(&c);
    uint32_t version = rd_u32(&c);
    if (magic != GGUF_MAGIC || version != GGUF_VERSION) {
        if (err && err_cap) snprintf(err, err_cap,
            "not a GGUF v3 file (magic=0x%08x version=%u)", magic, version);
        file_destroy(f); return NULL;
    }
    uint64_t n_tensors = rd_u64(&c);
    uint64_t n_kv      = rd_u64(&c);

    f->kvs     = (gguf_kv *)calloc(n_kv, sizeof(gguf_kv));
    f->n_kv    = n_kv;
    f->tensors = (gguf_tensor_info *)calloc(n_tensors, sizeof(gguf_tensor_info));
    f->n_tensors = n_tensors;
    if ((n_kv && !f->kvs) || (n_tensors && !f->tensors)) {
        if (err && err_cap) snprintf(err, err_cap, "oom (kv/tensor table)");
        file_destroy(f); return NULL;
    }

    for (uint64_t i = 0; i < n_kv && !c.overflow; i++) {
        f->kvs[i].key = rd_string(&c);
        if (!f->kvs[i].key) { c.overflow = 1; break; }
        if (read_value(&c, &f->kvs[i])) break;
        /* Pick up alignment override if present. */
        if (f->kvs[i].key && !strcmp(f->kvs[i].key, "general.alignment") &&
            f->kvs[i].type == GGUF_T_UINT32) {
            f->alignment = f->kvs[i].v.u32 ? f->kvs[i].v.u32 : GGUF_DEFAULT_ALIGN;
        }
    }
    if (c.overflow) {
        if (err && err_cap) snprintf(err, err_cap, "truncated kv section");
        file_destroy(f); return NULL;
    }

    for (uint64_t i = 0; i < n_tensors && !c.overflow; i++) {
        f->tensors[i].name = rd_string(&c);
        f->tensors[i].n_dims = rd_u32(&c);
        if (f->tensors[i].n_dims > 4) { c.overflow = 1; break; }
        for (uint32_t d = 0; d < f->tensors[i].n_dims; d++)
            f->tensors[i].dims[d] = rd_u64(&c);
        f->tensors[i].ggml_type = rd_u32(&c);
        f->tensors[i].offset    = rd_u64(&c);
    }
    if (c.overflow) {
        if (err && err_cap) snprintf(err, err_cap, "truncated tensor info");
        file_destroy(f); return NULL;
    }

    /* Pad to alignment to locate the data section. */
    size_t pos = (size_t)(c.p - f->map);
    size_t pad = (f->alignment - (pos % f->alignment)) % f->alignment;
    if (pos + pad > f->map_len) {
        if (err && err_cap) snprintf(err, err_cap, "data section past EOF");
        file_destroy(f); return NULL;
    }
    f->data_start = f->map + pos + pad;

    /* Resolve absolute tensor pointers + byte sizes. */
    for (size_t i = 0; i < f->n_tensors; i++) {
        gguf_tensor_info *t = &f->tensors[i];
        uint64_t numel = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) numel *= t->dims[d];
        t->bytes = tensor_bytes(t->ggml_type, numel);
        const uint8_t *p = f->data_start + t->offset;
        if (t->bytes && p + t->bytes > f->map + f->map_len) {
            if (err && err_cap) snprintf(err, err_cap,
                "tensor '%s' out of bounds (offset=%llu bytes=%zu)",
                t->name ? t->name : "(null)",
                (unsigned long long)t->offset, t->bytes);
            file_destroy(f); return NULL;
        }
        t->data = (const void *)p;
    }

    return f;
}

void qw36_gguf_close(qw36_gguf_file *f) { file_destroy(f); }

/* Linear scan — vocab-sized lookups; we don't optimize until needed. */
static const gguf_kv *find_kv(const qw36_gguf_file *f, const char *key) {
    for (size_t i = 0; i < f->n_kv; i++)
        if (f->kvs[i].key && !strcmp(f->kvs[i].key, key))
            return &f->kvs[i];
    return NULL;
}

int qw36_gguf_get_u32(const qw36_gguf_file *f, const char *key, uint32_t *out) {
    const gguf_kv *k = find_kv(f, key);
    if (!k) return -1;
    switch (k->type) {
        case GGUF_T_UINT8:  *out = k->v.u8;  return 0;
        case GGUF_T_UINT16: *out = k->v.u16; return 0;
        case GGUF_T_UINT32: *out = k->v.u32; return 0;
        case GGUF_T_UINT64: *out = (uint32_t)k->v.u64; return 0;
        case GGUF_T_INT32:  *out = (uint32_t)k->v.i32; return 0;
        case GGUF_T_BOOL:   *out = k->v.b;   return 0;
        default: return -1;
    }
}

int qw36_gguf_get_f32(const qw36_gguf_file *f, const char *key, float *out) {
    const gguf_kv *k = find_kv(f, key);
    if (!k) return -1;
    if (k->type == GGUF_T_FLOAT32) { *out = k->v.f32; return 0; }
    if (k->type == GGUF_T_FLOAT64) { *out = (float)k->v.f64; return 0; }
    return -1;
}

int qw36_gguf_get_str(const qw36_gguf_file *f, const char *key, const char **out) {
    const gguf_kv *k = find_kv(f, key);
    if (!k || k->type != GGUF_T_STRING) return -1;
    /* We aliased the mmap, which is not null-terminated. Build a static
     * scratch — but that breaks reentrancy. Instead, leak a tiny copy. The
     * file owns nothing else like it, so this is fine for v0. */
    static __thread char *scratch = NULL;
    static __thread size_t scratch_cap = 0;
    if (scratch_cap <= k->v.str.len) {
        size_t newcap = (k->v.str.len + 1) * 2;
        char *p = (char *)realloc(scratch, newcap);
        if (!p) return -1;
        scratch = p;
        scratch_cap = newcap;
    }
    memcpy(scratch, k->v.str.ptr, k->v.str.len);
    scratch[k->v.str.len] = '\0';
    *out = scratch;
    return 0;
}

int qw36_gguf_get_u32_array(const qw36_gguf_file *f, const char *key,
                            uint32_t *out, uint32_t cap)
{
    const gguf_kv *k = find_kv(f, key);
    if (!k || k->type != GGUF_T_ARRAY) return -1;
    gguf_value_type et = k->v.arr.elem_type;
    if (et != GGUF_T_UINT32 && et != GGUF_T_INT32) return -1;
    uint32_t n = (uint32_t)k->v.arr.n;
    if (n > cap) n = cap;
    const uint8_t *p = (const uint8_t *)k->v.arr.raw;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t v; memcpy(&v, p + i * 4, 4);
        out[i] = v;
    }
    return (int)n;
}

size_t qw36_gguf_tensor_count(const qw36_gguf_file *f) {
    return f ? f->n_tensors : 0;
}

int qw36_gguf_get_tensor_by_index(const qw36_gguf_file *f, size_t i,
                                  qw36_gguf_tensor *out) {
    if (!f || i >= f->n_tensors) return -1;
    const gguf_tensor_info *t = &f->tensors[i];
    out->name      = t->name;
    out->dtype     = ggml_to_qw36(t->ggml_type);
    out->ggml_type = t->ggml_type;
    out->n_dims    = t->n_dims;
    memcpy(out->dims, t->dims, sizeof(out->dims));
    out->data      = t->data;
    out->nbytes    = t->bytes;
    return 0;
}

int qw36_gguf_get_tensor(const qw36_gguf_file *f, const char *name,
                         qw36_gguf_tensor *out)
{
    for (size_t i = 0; i < f->n_tensors; i++) {
        const gguf_tensor_info *t = &f->tensors[i];
        if (t->name && !strcmp(t->name, name)) {
            out->name      = t->name;
            out->dtype     = ggml_to_qw36(t->ggml_type);
            out->ggml_type = t->ggml_type;
            out->n_dims    = t->n_dims;
            memcpy(out->dims, t->dims, sizeof(out->dims));
            out->data      = t->data;
            out->nbytes    = t->bytes;
            return 0;
        }
    }
    return -1;
}

/* Expose the array accessor we need for tokenizer vocab loading. Internal
 * — declared via a private header sibling once the tokenizer lands. */
int qw36__gguf_get_str_array(const qw36_gguf_file *f, const char *key,
                             const char *const **out_ptrs,
                             uint64_t **out_lens, uint64_t *out_n);

int qw36__gguf_get_str_array(const qw36_gguf_file *f, const char *key,
                             const char *const **out_ptrs,
                             uint64_t **out_lens, uint64_t *out_n)
{
    const gguf_kv *k = find_kv(f, key);
    if (!k || k->type != GGUF_T_ARRAY || k->v.arr.elem_type != GGUF_T_STRING)
        return -1;
    uint64_t n = k->v.arr.n;
    const char **ptrs = (const char **)malloc(n * sizeof(char *));
    uint64_t *lens    = (uint64_t *)malloc(n * sizeof(uint64_t));
    if (!ptrs || !lens) { free(ptrs); free(lens); return -1; }
    const uint8_t *p = (const uint8_t *)k->v.arr.raw;
    const uint8_t *end = f->map + f->map_len;
    for (uint64_t i = 0; i < n; i++) {
        if ((size_t)(end - p) < 8) { free(ptrs); free(lens); return -1; }
        uint64_t l; memcpy(&l, p, 8); p += 8;
        if ((size_t)(end - p) < l)   { free(ptrs); free(lens); return -1; }
        ptrs[i] = (const char *)p;
        lens[i] = l;
        p += l;
    }
    *out_ptrs = ptrs;
    *out_lens = lens;
    *out_n    = n;
    return 0;
}
