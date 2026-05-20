/* qw36.c — CPU reference forward pass + sampling + engine lifecycle.
 *
 * Owner: Claude.
 *
 * This file is the *reference*. Every GPU backend must produce fp32
 * output that matches this within tolerance on the golden vectors in
 * tests/. Keep it readable over fast — readability is the whole point.
 */

#include "qw36.h"
#include "qw36_gpu.h"
#include "qw36_gguf.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QW36_VERSION_STR "qw36 0.0.1-scaffold"

const char *qw36_version(void) { return QW36_VERSION_STR; }

/* Lazy weight descriptor: a pointer into the mmap'd GGUF, plus the dtype
 * and shape so matmul/embed can dequantize per-block on the fly. We avoid
 * materializing huge tensors up-front (lm_head, expert mats, qkv, mlp)
 * because a 35B Q4_K_XL model decompresses to ~140GB fp32.
 *
 * After engine_open with backend != NULL we may flip `data` to point at a
 * materialized fp32 buffer (host-side) and set `gpu_buf` to a backend-owned
 * device view of the same memory (zero-copy on Apple unified memory). */
typedef struct {
    const void   *data;
    qw36_dtype    dtype;       /* current dtype; flips to F32 after eager dequant */
    uint32_t      ggml_type;   /* original ggml type — informational */
    uint64_t      rows;        /* GGUF dim[1] (outer) */
    uint64_t      cols;        /* GGUF dim[0] (inner) */
    uint64_t      n_extra;     /* dim[2] (used for MoE expert stacks) */
    qw36_gpu_buf *gpu_buf;     /* non-NULL after upload to GPU backend */
} qw36_lazy_w;

struct qw36_engine {
    qw36_config       cfg;
    qw36_weights      weights;
    qw36_gpu_backend *backend;     /* may be NULL → CPU-only */
    qw36_gpu_ctx     *ctx;
    qw36_gguf_file   *gguf;

    /* Owned heap allocations (materialized fp32 buffers for small tensors
     * like norms, plus qw36_lazy_w descriptors for big tensors). All freed
     * uniformly on close. */
    void             **owned;
    size_t             owned_n;
    size_t             owned_cap;

    /* Cached arch prefix ("qwen3" / "qwen3moe" / "qwen35" / "qwen35moe"). */
    char               arch[32];
};

static void *eng_own_(qw36_engine *eng, void *p) {
    if (!p) return NULL;
    if (eng->owned_n >= eng->owned_cap) {
        size_t nc = eng->owned_cap ? eng->owned_cap * 2 : 64;
        void **arr = (void **)realloc(eng->owned, nc * sizeof(void *));
        if (!arr) return p; /* will leak but won't crash; debug later */
        eng->owned = arr;
        eng->owned_cap = nc;
    }
    eng->owned[eng->owned_n++] = p;
    return p;
}

/* --------------------------------------------------------------------- */
/* dtype conversion helpers                                               */
/* --------------------------------------------------------------------- */

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign << 31; }
        else {
            /* denorm → normalize */
            int e = 1;
            while (!(mant & 0x400u)) { mant <<= 1; e--; }
            mant &= 0x3FFu;
            f = (sign << 31) | ((uint32_t)(e + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = (sign << 31) | (0xFFu << 23) | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    union { uint32_t u; float f; } u; u.u = f; return u.f;
}

static float bf16_to_f32(uint16_t b) {
    union { uint32_t u; float f; } u;
    u.u = (uint32_t)b << 16;
    return u.f;
}

/* Read element i of a tensor in storage dtype, return fp32. */
static float load_elem_f32(const void *data, qw36_dtype dt, size_t i) {
    switch (dt) {
        case QW36_DTYPE_F32:  return ((const float    *)data)[i];
        case QW36_DTYPE_F16:  return f16_to_f32(((const uint16_t *)data)[i]);
        case QW36_DTYPE_BF16: return bf16_to_f32(((const uint16_t *)data)[i]);
        default: return 0.0f; /* quantized — TODO */
    }
}

/* --------------------------------------------------------------------- */
/* GGML-style quantized block dequantizers (Q8_0 / Q4_K / Q6_K).          */
/*                                                                        */
/* Block layouts mirror llama.cpp's ggml-quants.h. Numeric output matches */
/* dequantize_row_* in ggml-quants.c so loaded weights are bit-equivalent */
/* to a fresh load through the upstream library.                         */
/* --------------------------------------------------------------------- */

#define QW36_QK_K   256
#define QW36_QK8_0  32

static void dq_q8_0(const uint8_t *blocks, float *out, size_t n) {
    /* block: fp16 d (2B) + int8 qs[32] (32B) = 34B per 32 elements. */
    const size_t nb = n / QW36_QK8_0;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t *b = blocks + i * 34;
        uint16_t dh; memcpy(&dh, b, 2);
        float d = f16_to_f32(dh);
        const int8_t *qs = (const int8_t *)(b + 2);
        for (int j = 0; j < QW36_QK8_0; j++) *out++ = d * (float)qs[j];
    }
}

/* Q4_K: 144 bytes/block. Layout:
 *   fp16 d, fp16 dmin, u8 scales[12], u8 qs[128]
 * Per 64-element chunk a sub-scale and sub-min are unpacked from scales[]. */
static void q4_K_get_scale_min(int j, const uint8_t *q,
                               uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j]     & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

static void dq_q4_K(const uint8_t *blocks, float *out, size_t n) {
    const size_t nb = n / QW36_QK_K;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t *b = blocks + i * 144;
        uint16_t dh, dmh;
        memcpy(&dh,  b,     2);
        memcpy(&dmh, b + 2, 2);
        const float d   = f16_to_f32(dh);
        const float dmn = f16_to_f32(dmh);
        const uint8_t *scales = b + 4;
        const uint8_t *qs     = b + 16;
        int is = 0;
        for (int j = 0; j < QW36_QK_K; j += 64) {
            uint8_t sc, m;
            q4_K_get_scale_min(is + 0, scales, &sc, &m);
            const float d1 = d * (float)sc, m1 = dmn * (float)m;
            q4_K_get_scale_min(is + 1, scales, &sc, &m);
            const float d2 = d * (float)sc, m2 = dmn * (float)m;
            for (int l = 0; l < 32; l++) *out++ = d1 * (float)(qs[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++) *out++ = d2 * (float)(qs[l] >>  4) - m2;
            qs += 32; is += 2;
        }
    }
}

/* Q2_K: 84 bytes/block. Layout:
 *   u8 scales[16]
 *   u8 qs[64]      (2-bit quants)
 *   fp16 d, fp16 dmin
 * Mirrors dequantize_row_q2_K in ggml-quants.c. */
static void dq_q2_K(const uint8_t *blocks, float *out, size_t n) {
    const size_t nb = n / QW36_QK_K;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t *b      = blocks + i * 84;
        const uint8_t *scales = b;
        const uint8_t *qs     = b + 16;
        uint16_t dh, dmh;
        memcpy(&dh,  b + 80, 2);
        memcpy(&dmh, b + 82, 2);
        const float d   = f16_to_f32(dh);
        const float dmn = f16_to_f32(dmh);
        int is = 0;
        for (int nblk = 0; nblk < QW36_QK_K; nblk += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t sc = scales[is++];
                float dl = d * (float)(sc & 0xF), ml = dmn * (float)(sc >> 4);
                for (int l = 0; l < 16; l++)
                    *out++ = dl * (float)((int8_t)((qs[l] >> shift) & 3)) - ml;
                sc = scales[is++];
                dl = d * (float)(sc & 0xF); ml = dmn * (float)(sc >> 4);
                for (int l = 0; l < 16; l++)
                    *out++ = dl * (float)((int8_t)((qs[l + 16] >> shift) & 3)) - ml;
                shift += 2;
            }
            qs += 32;
        }
    }
}

/* Q3_K: 110 bytes/block. Layout:
 *   u8 hmask[32]   — high bit per element
 *   u8 qs[64]      — low 2 bits per element
 *   u8 scales[12]  — 6-bit packed scales
 *   fp16 d
 * Mirrors dequantize_row_q3_K in ggml-quants.c. */
static void dq_q3_K(const uint8_t *blocks, float *out, size_t n) {
    const size_t nb = n / QW36_QK_K;
    static const uint32_t kmask1 = 0x03030303;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t *b  = blocks + i * 110;
        const uint8_t *hm = b;
        const uint8_t *qs = b + 32;
        uint16_t dh;
        memcpy(&dh, b + 32 + 64 + 12, 2);
        const float d_all = f16_to_f32(dh);

        uint32_t aux[4];
        memcpy(aux, b + 32 + 64, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0]      & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1]      & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t *scales = (const int8_t *)aux;

        uint8_t mask = 1;
        int is = 0;
        for (int nblk = 0; nblk < QW36_QK_K; nblk += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                const float dl = d_all * (float)((int)scales[is++] - 32);
                for (int l = 0; l < 32; l++) {
                    int q = ((qs[l] >> shift) & 3) - ((hm[l] & mask) ? 0 : 4);
                    *out++ = dl * (float)q;
                }
                shift += 2; mask <<= 1;
            }
            qs += 32;
        }
    }
}

/* Q5_K: 176 bytes/block. Layout:
 *   fp16 d, fp16 dmin
 *   u8 scales[12]   — same 6-bit packing as Q4_K
 *   u8 qh[32]       — 5th bit per element
 *   u8 qs[128]      — lower 4 bits
 * Mirrors dequantize_row_q5_K in ggml-quants.c. */
static void dq_q5_K(const uint8_t *blocks, float *out, size_t n) {
    const size_t nb = n / QW36_QK_K;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t *b = blocks + i * 176;
        uint16_t dh, dmh;
        memcpy(&dh,  b,     2);
        memcpy(&dmh, b + 2, 2);
        const float d   = f16_to_f32(dh);
        const float dmn = f16_to_f32(dmh);
        const uint8_t *scales = b + 4;
        const uint8_t *qh     = b + 16;
        const uint8_t *ql     = b + 48;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QW36_QK_K; j += 64) {
            uint8_t sc, m;
            #define GS_K4(jj) do { \
                if ((jj) < 4) { sc = scales[(jj)] & 63; m = scales[(jj)+4] & 63; } \
                else { \
                    sc = (scales[(jj)+4] & 0xF) | ((scales[(jj)-4] >> 6) << 4); \
                    m  = (scales[(jj)+4] >>  4) | ((scales[(jj)-0] >> 6) << 4); \
                } \
            } while (0)
            GS_K4(is + 0);
            const float d1 = d * (float)sc, m1 = dmn * (float)m;
            GS_K4(is + 1);
            const float d2 = d * (float)sc, m2 = dmn * (float)m;
            #undef GS_K4
            for (int l = 0; l < 32; l++)
                *out++ = d1 * (float)((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
            for (int l = 0; l < 32; l++)
                *out++ = d2 * (float)((ql[l] >>  4) + ((qh[l] & u2) ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

/* Q6_K: 210 bytes/block. Layout:
 *   u8 ql[128]  — lower 4 bits per quant
 *   u8 qh[64]   — upper 2 bits per quant
 *   i8 scales[16]
 *   fp16 d      — super-block scale
 */
static void dq_q6_K(const uint8_t *blocks, float *out, size_t n) {
    const size_t nb = n / QW36_QK_K;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t *b = blocks + i * 210;
        const uint8_t *ql = b;
        const uint8_t *qh = b + 128;
        const int8_t  *sc = (const int8_t *)(b + 128 + 64);
        uint16_t dh; memcpy(&dh, b + 128 + 64 + 16, 2);
        const float d = f16_to_f32(dh);
        for (int n_off = 0; n_off < QW36_QK_K; n_off += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                out[l +  0] = d * (float)sc[is + 0] * (float)q1;
                out[l + 32] = d * (float)sc[is + 2] * (float)q2;
                out[l + 64] = d * (float)sc[is + 4] * (float)q3;
                out[l + 96] = d * (float)sc[is + 6] * (float)q4;
            }
            out += 128; ql += 64; qh += 32; sc += 8;
        }
    }
}

/* Materialize an entire tensor to a freshly malloc'd fp32 buffer. Caller
 * frees. Returns NULL on unsupported dtype / oom. */
static float *materialize_f32(const void *data, qw36_dtype dt, size_t n) {
    float *out = (float *)malloc(n * sizeof(float));
    if (!out) return NULL;
    switch (dt) {
        case QW36_DTYPE_F32:
            memcpy(out, data, n * sizeof(float));
            return out;
        case QW36_DTYPE_F16: {
            const uint16_t *p = (const uint16_t *)data;
            for (size_t i = 0; i < n; i++) out[i] = f16_to_f32(p[i]);
            return out;
        }
        case QW36_DTYPE_BF16: {
            const uint16_t *p = (const uint16_t *)data;
            for (size_t i = 0; i < n; i++) out[i] = bf16_to_f32(p[i]);
            return out;
        }
        case QW36_DTYPE_Q8_0:
            dq_q8_0((const uint8_t *)data, out, n);
            return out;
        case QW36_DTYPE_Q2_K:
            dq_q2_K((const uint8_t *)data, out, n);
            return out;
        case QW36_DTYPE_Q3_K:
            dq_q3_K((const uint8_t *)data, out, n);
            return out;
        case QW36_DTYPE_Q4_K:
            dq_q4_K((const uint8_t *)data, out, n);
            return out;
        case QW36_DTYPE_Q5_K:
            dq_q5_K((const uint8_t *)data, out, n);
            return out;
        case QW36_DTYPE_Q6_K:
            dq_q6_K((const uint8_t *)data, out, n);
            return out;
        default:
            free(out);
            return NULL;
    }
}

/* Forward declarations for helpers used by MoE / lazy ops before their
 * full definitions further down. */
static inline float silu(float x);

/* --------------------------------------------------------------------- */
/* Lazy block-quantized helpers: dequantize one row at a time.            */
/* --------------------------------------------------------------------- */

static int dtype_block_geom(qw36_dtype dt, size_t *qk, size_t *bytes_per_block) {
    switch (dt) {
        case QW36_DTYPE_Q2_K: *qk = 256; *bytes_per_block =  84; return 0;
        case QW36_DTYPE_Q3_K: *qk = 256; *bytes_per_block = 110; return 0;
        case QW36_DTYPE_Q4_K: *qk = 256; *bytes_per_block = 144; return 0;
        case QW36_DTYPE_Q5_K: *qk = 256; *bytes_per_block = 176; return 0;
        case QW36_DTYPE_Q6_K: *qk = 256; *bytes_per_block = 210; return 0;
        case QW36_DTYPE_Q8_0: *qk = 32;  *bytes_per_block = 34;  return 0;
        default: *qk = 0; *bytes_per_block = 0; return -1;
    }
}

/* Decode `cols` elements of row `row_idx` from a block-quantized stack
 * where each row is laid out as cols-elements-worth of blocks. */
static int dequant_row(const qw36_lazy_w *w, size_t row_idx, float *out) {
    const size_t cols = (size_t)w->cols;
    switch (w->dtype) {
        case QW36_DTYPE_F32: {
            const float *p = (const float *)w->data + row_idx * cols;
            memcpy(out, p, cols * sizeof(float));
            return 0;
        }
        case QW36_DTYPE_F16: {
            const uint16_t *p = (const uint16_t *)w->data + row_idx * cols;
            for (size_t i = 0; i < cols; i++) out[i] = f16_to_f32(p[i]);
            return 0;
        }
        case QW36_DTYPE_BF16: {
            const uint16_t *p = (const uint16_t *)w->data + row_idx * cols;
            for (size_t i = 0; i < cols; i++) out[i] = bf16_to_f32(p[i]);
            return 0;
        }
        default: break;
    }
    size_t qk, bpb;
    if (dtype_block_geom(w->dtype, &qk, &bpb) || cols % qk != 0) return -1;
    const size_t blocks_per_row = cols / qk;
    const uint8_t *row = (const uint8_t *)w->data
                       + row_idx * blocks_per_row * bpb;
    switch (w->dtype) {
        case QW36_DTYPE_Q2_K: dq_q2_K(row, out, cols); return 0;
        case QW36_DTYPE_Q3_K: dq_q3_K(row, out, cols); return 0;
        case QW36_DTYPE_Q4_K: dq_q4_K(row, out, cols); return 0;
        case QW36_DTYPE_Q5_K: dq_q5_K(row, out, cols); return 0;
        case QW36_DTYPE_Q6_K: dq_q6_K(row, out, cols); return 0;
        case QW36_DTYPE_Q8_0: dq_q8_0(row, out, cols); return 0;
        default: return -1;
    }
}

/* Set/cleared by qw36_forward so matmul_lazy can locate the backend
 * without threading the engine pointer through every helper signature.
 * Single-threaded today; promote to __thread when batched/multi-stream. */
static qw36_engine *qw36_active_engine = NULL;

/* matmul against a lazy quantized weight: y[r] = sum_c W[r,c] * x[c].
 * Dispatches to the GPU backend when one is attached and the weight has
 * an uploaded gpu_buf; otherwise falls back to per-row CPU dequant + dot. */
static int matmul_lazy(float *y, const float *x, const qw36_lazy_w *w,
                       float *row_scratch)
{
    const size_t rows = (size_t)w->rows;
    const size_t cols = (size_t)w->cols;

    qw36_engine *eng = qw36_active_engine;
    if (eng && eng->backend && eng->backend->matmul &&
        eng->backend->upload && eng->backend->download &&
        eng->backend->alloc && eng->backend->free &&
        w->gpu_buf)
    {
        qw36_gpu_backend *be = eng->backend;
        qw36_gpu_ctx     *ctx = eng->ctx;
        qw36_gpu_buf *xb = be->upload(ctx, x, cols * sizeof(float), QW36_DTYPE_F32);
        qw36_gpu_buf *yb = be->alloc(ctx, rows * sizeof(float),   QW36_DTYPE_F32);
        if (!xb || !yb) {
            if (xb) be->free(ctx, xb);
            if (yb) be->free(ctx, yb);
            goto cpu_path;
        }
        be->matmul(ctx, yb, xb, w->gpu_buf, 1, (uint32_t)rows, (uint32_t)cols);
        be->download(ctx, yb, y, rows * sizeof(float));
        be->free(ctx, xb);
        be->free(ctx, yb);
        return 0;
    }

cpu_path:
    for (size_t r = 0; r < rows; r++) {
        if (dequant_row(w, r, row_scratch)) return -1;
        double acc = 0.0;
        for (size_t c = 0; c < cols; c++) acc += (double)row_scratch[c] * x[c];
        y[r] = (float)acc;
    }
    return 0;
}

/* Same as matmul_lazy but operates on a single slice of a 3D stack
 * [n_experts, rows, cols]. Used for MoE expert matmuls without copying. */
static int matmul_lazy_slice(float *y, const float *x,
                             const qw36_lazy_w *w, size_t slice_idx,
                             float *row_scratch)
{
    const size_t rows = (size_t)w->rows;
    const size_t cols = (size_t)w->cols;
    /* Byte offset of slice s = s * rows * blocks_per_row * bytes_per_block. */
    size_t row_bytes;
    if (w->dtype == QW36_DTYPE_F32)      row_bytes = cols * 4;
    else if (w->dtype == QW36_DTYPE_F16) row_bytes = cols * 2;
    else if (w->dtype == QW36_DTYPE_BF16) row_bytes = cols * 2;
    else {
        size_t qk, bpb;
        if (dtype_block_geom(w->dtype, &qk, &bpb) || cols % qk != 0) return -1;
        row_bytes = (cols / qk) * bpb;
    }
    const uint8_t *slice_data = (const uint8_t *)w->data
                              + slice_idx * rows * row_bytes;
    qw36_lazy_w view = *w;
    view.data = (const void *)slice_data;
    return matmul_lazy(y, x, &view, row_scratch);
}

/* Top-k selection (small k, unsorted by index). On return out_idx/out_val
 * hold the k largest values of `logits`, sorted descending by value. */
static void top_k_select(const float *logits, uint32_t n, int k,
                         uint32_t *out_idx, float *out_val)
{
    /* Initialise the heap with -inf. */
    for (int i = 0; i < k; i++) { out_idx[i] = 0; out_val[i] = -INFINITY; }
    for (uint32_t i = 0; i < n; i++) {
        if (logits[i] <= out_val[k - 1]) continue;
        /* Insertion sort: find position, shift. */
        int j = k - 1;
        while (j > 0 && logits[i] > out_val[j - 1]) {
            out_val[j] = out_val[j - 1];
            out_idx[j] = out_idx[j - 1];
            j--;
        }
        out_val[j] = logits[i];
        out_idx[j] = i;
    }
}

/* Stable softmax over the first k entries of vals (in-place). */
static void softmax_k(float *vals, int k) {
    float mx = vals[0];
    for (int i = 1; i < k; i++) if (vals[i] > mx) mx = vals[i];
    double sum = 0.0;
    for (int i = 0; i < k; i++) { vals[i] = expf(vals[i] - mx); sum += vals[i]; }
    float inv = (float)(1.0 / sum);
    for (int i = 0; i < k; i++) vals[i] *= inv;
}

/* --------------------------------------------------------------------- */
/* Gated DeltaNet (Qwen3.5 / 3.6 linear attention) decode.                */
/*                                                                        */
/* Mirrors agent-infer/crates/cuda-kernels/.../gated_delta_rule.cu        */
/* (gated_delta_rule_decode_kernel) — see comments there. Algorithm per   */
/* token, per value head v:                                               */
/*                                                                        */
/*   k_head = v * num_key_heads / num_value_heads                         */
/*   q = qkv[k_head*key_dim ..]            (already conv+SiLU'd by caller)*/
/*   k = qkv[q_dim + k_head*key_dim ..]                                   */
/*   v_in = qkv[q_dim + k_dim + v*val_dim ..]                             */
/*   q ← L2-normalize(q) / sqrt(key_dim)                                  */
/*   k ← L2-normalize(k)                                                  */
/*   x = α[v] + dt_bias[v];     softplus = log(1 + e^x) (or x if x>20)    */
/*   g  = -e^{A_log[v]} * softplus;   exp_g = e^g                         */
/*   β  = σ(b[v])                                                         */
/*   kv_mem[d] = Σ_j (S[v,j,d] * exp_g) * k[j]    (pass 1, also decays S) */
/*   δ[d]      = (v_in[d] - kv_mem[d]) * β                                */
/*   S[v,j,d] += δ[d] * k[j]                                              */
/*   y[v,d]    = Σ_j S[v,j,d] * q[j]                                      */
/* --------------------------------------------------------------------- */
static void gated_delta_decode(const float *qkv,
                               const float *b_proj_raw,
                               const float *a_proj_raw,
                               const float *dt_bias,
                               const float *a_log,
                               float       *state,    /* [n_v, key, val] */
                               float       *out,      /* [n_v * val] */
                               uint32_t n_key, uint32_t n_value,
                               uint32_t key_dim, uint32_t val_dim,
                               float *qbuf, float *kbuf)
{
    const uint32_t q_dim_total = n_key * key_dim;
    const uint32_t k_dim_total = q_dim_total;
    const float    inv_sqrt_d  = 1.0f / sqrtf((float)key_dim);

    for (uint32_t v = 0; v < n_value; v++) {
        /* We keep GGUF's native V-head order instead of materializing the
         * agent-infer/HF reorder. GGUF stores value heads tiled by
         * V-within-K-group, so raw value head v maps to key head v % n_key.
         * agent-infer reverses this order at load time and then uses
         * v * n_key / n_value. */
        const uint32_t kh = v % n_key;

        /* Copy and L2-normalize q, k for this head. */
        double qss = 0.0, kss = 0.0;
        for (uint32_t d = 0; d < key_dim; d++) {
            qbuf[d] = qkv[kh * key_dim + d];
            kbuf[d] = qkv[q_dim_total + kh * key_dim + d];
            qss += (double)qbuf[d] * qbuf[d];
            kss += (double)kbuf[d] * kbuf[d];
        }
        float q_scale = 1.0f / sqrtf((float)qss + 1e-12f);
        float k_scale = 1.0f / sqrtf((float)kss + 1e-12f);
        for (uint32_t d = 0; d < key_dim; d++) {
            qbuf[d] *= q_scale * inv_sqrt_d;
            kbuf[d] *= k_scale;
        }

        const float *vin = qkv + q_dim_total + k_dim_total + v * val_dim;

        /* Gating scalars. */
        float a_v = a_proj_raw[v] + dt_bias[v];
        float softplus = (a_v > 20.0f) ? a_v : logf(1.0f + expf(a_v));
        float g  = -expf(a_log[v]) * softplus;
        float exp_g = expf(g);
        float beta  = 1.0f / (1.0f + expf(-b_proj_raw[v]));

        float *S = state + (size_t)v * key_dim * val_dim;

        /* Pass 1: decay state in place + accumulate kv_mem = (decayed_S^T) @ k */
        float *out_v = out + (size_t)v * val_dim;
        for (uint32_t d = 0; d < val_dim; d++) out_v[d] = 0.0f;
        /* Use out_v as kv_mem accumulator first; we'll overwrite it later. */
        for (uint32_t j = 0; j < key_dim; j++) {
            float *row = S + (size_t)j * val_dim;
            const float kj = kbuf[j];
            for (uint32_t d = 0; d < val_dim; d++) {
                float s = row[d] * exp_g;
                row[d] = s;
                out_v[d] += s * kj;
            }
        }

        /* Pass 2: rank-1 update + output. */
        /* delta = (v_in - kv_mem) * beta. We need kv_mem from pass 1 (in
         * out_v), so copy it aside then zero out_v for the output sum. */
        float kv_mem_local[8192];
        for (uint32_t d = 0; d < val_dim; d++) {
            kv_mem_local[d] = out_v[d];
            out_v[d] = 0.0f;
        }
        for (uint32_t j = 0; j < key_dim; j++) {
            float *row = S + (size_t)j * val_dim;
            const float kj = kbuf[j];
            const float qj = qbuf[j];
            for (uint32_t d = 0; d < val_dim; d++) {
                float delta = (vin[d] - kv_mem_local[d]) * beta;
                row[d] += delta * kj;
                out_v[d] += row[d] * qj;
            }
        }
    }
}

/* Depthwise causal 1D conv (decode step). conv_w is [k, channels] (or its
 * transpose; we read element wt(t, c) accordingly). conv_state holds the
 * last (k-1) inputs per channel, shape [k-1, channels]. On exit:
 *   y[c] = silu( Σ_{t=0..k-1} wt(t, c) * input_at_t[c] )
 * where input at the latest position is x[c]; older positions live in
 * conv_state. State is shifted left by one and the new x appended.
 *
 * conv_w layout in GGUF is [k, channels] with k = dim[1] (i.e. innermost
 * axis is k). We read wt(t, c) as conv_w[c * k + t]. */
static void conv1d_silu_decode(const float *x, const float *conv_w,
                               float *conv_state, float *y,
                               uint32_t channels, uint32_t k)
{
    /* y[c] = silu(sum_{t=0..k-1} conv_w[c*k + t] * window[t, c])
     * where window[0..k-2] is conv_state, window[k-1] is x. */
    if (k == 0) {
        for (uint32_t c = 0; c < channels; c++) y[c] = silu(x[c]);
        return;
    }
    for (uint32_t c = 0; c < channels; c++) {
        const float *wt = conv_w + c * k;
        double acc = 0.0;
        if (k > 1) {
            const float *win = conv_state + c;        /* stride = channels */
            for (uint32_t t = 0; t < k - 1; t++)
                acc += (double)wt[t] * win[t * channels];
        }
        acc += (double)wt[k - 1] * x[c];
        y[c] = silu((float)acc);
    }
    /* Shift state: drop oldest, append new x. */
    if (k > 1) {
        for (uint32_t t = 0; t + 1 < k - 1; t++) {
            float *dst = conv_state + (size_t)t * channels;
            const float *src = conv_state + (size_t)(t + 1) * channels;
            memcpy(dst, src, channels * sizeof(float));
        }
        memcpy(conv_state + (size_t)(k - 2) * channels, x,
               channels * sizeof(float));
    }
}

/* MoE forward (per token):
 *
 *   r = router @ x                       [n_experts]
 *   pick top_k experts by r
 *   probs = softmax(r[top_k]); (optionally renormalized)
 *   y = Σ_{e ∈ top_k} probs[e] * expert_e(x)
 *     + shared_expert(x)                 (if shared weights present)
 *
 *   expert_e(x) = down_exps[e] @ (silu(gate_exps[e] @ x) * (up_exps[e] @ x))
 *
 * scratch needs: hidden + n_experts + 2*k + 2*moe_inter + hidden floats. */
static int moe_forward_f32(float *y, const float *x,
                           const qw36_lazy_w *router,
                           const qw36_lazy_w *gate_exps,
                           const qw36_lazy_w *up_exps,
                           const qw36_lazy_w *down_exps,
                           const qw36_lazy_w *shared_gate,
                           const qw36_lazy_w *shared_up,
                           const qw36_lazy_w *shared_down,
                           uint32_t hidden, uint32_t moe_inter,
                           uint32_t n_experts, int top_k,
                           uint8_t norm_topk,
                           float *scratch, float *row_scratch)
{
    float    *r_logits = scratch;                          /* n_experts */
    float    *tk_vals  = r_logits + n_experts;             /* top_k    */
    uint32_t *tk_idx   = (uint32_t *)(tk_vals + top_k);    /* top_k    */
    float    *tmp_gate = (float *)(tk_idx + top_k);        /* moe_inter*/
    float    *tmp_up   = tmp_gate + moe_inter;             /* moe_inter*/
    float    *tmp_y    = tmp_up   + moe_inter;             /* hidden   */

    /* 1. router */
    if (matmul_lazy(r_logits, x, router, row_scratch)) return -1;

    /* 2. top-k */
    top_k_select(r_logits, n_experts, top_k, tk_idx, tk_vals);

    /* 3. softmax (and optional renormalize — softmax_k already sums to 1) */
    softmax_k(tk_vals, top_k);
    (void)norm_topk; /* always renormalized — Qwen3 sets norm_topk_prob=true */

    /* 4. accumulate top-k experts */
    memset(y, 0, hidden * sizeof(float));
    for (int t = 0; t < top_k; t++) {
        const uint32_t e = tk_idx[t];
        const float    p = tk_vals[t];
        if (matmul_lazy_slice(tmp_gate, x, gate_exps, e, row_scratch)) return -1;
        if (matmul_lazy_slice(tmp_up,   x, up_exps,   e, row_scratch)) return -1;
        for (uint32_t i = 0; i < moe_inter; i++)
            tmp_gate[i] = silu(tmp_gate[i]) * tmp_up[i];
        if (matmul_lazy_slice(tmp_y, tmp_gate, down_exps, e, row_scratch)) return -1;
        for (uint32_t i = 0; i < hidden; i++) y[i] += p * tmp_y[i];
    }

    /* 5. shared expert (Qwen3-MoE: always-on extra path) */
    if (shared_gate && shared_up && shared_down) {
        if (matmul_lazy(tmp_gate, x, shared_gate, row_scratch)) return -1;
        if (matmul_lazy(tmp_up,   x, shared_up,   row_scratch)) return -1;
        for (uint32_t i = 0; i < moe_inter; i++)
            tmp_gate[i] = silu(tmp_gate[i]) * tmp_up[i];
        if (matmul_lazy(tmp_y, tmp_gate, shared_down, row_scratch)) return -1;
        for (uint32_t i = 0; i < hidden; i++) y[i] += tmp_y[i];
    }
    return 0;
}

/* Embedding lookup: write hidden=W.cols floats to out from row `token`. */
static int embed_lookup_lazy(const qw36_lazy_w *w, uint32_t token, float *out) {
    return dequant_row(w, token, out);
}

/* --------------------------------------------------------------------- */
/* CPU reference math — all fp32, scalar. The GPU backends must match.   */
/* --------------------------------------------------------------------- */

static void rmsnorm_f32(float *out, const float *x, const float *w,
                        size_t n, float eps)
{
    double ss = 0.0;
    for (size_t i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float scale = (float)(1.0 / sqrt(ss / (double)n + (double)eps));
    for (size_t i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}

/* y[r] = sum_c W[r,c] * x[c]. Row-major W, shape [rows, cols]. */
static void matmul_f32(float *y, const float *x, const float *w,
                       size_t rows, size_t cols)
{
    for (size_t r = 0; r < rows; r++) {
        double acc = 0.0;
        const float *wr = w + r * cols;
        for (size_t c = 0; c < cols; c++) acc += (double)wr[c] * x[c];
        y[r] = (float)acc;
    }
}

static inline float silu(float x) { return x / (1.0f + expf(-x)); }

static void swiglu_mlp_f32(float *y, const float *x,
                           const float *w_gate, const float *w_up,
                           const float *w_down,
                           float *scratch_gate, float *scratch_up,
                           size_t hidden, size_t inter)
{
    matmul_f32(scratch_gate, x, w_gate, inter, hidden);
    matmul_f32(scratch_up,   x, w_up,   inter, hidden);
    for (size_t i = 0; i < inter; i++)
        scratch_gate[i] = silu(scratch_gate[i]) * scratch_up[i];
    matmul_f32(y, scratch_gate, w_down, hidden, inter);
}

/* In-place RoPE on a single head. Rotates the first `rot_dim` components;
 * the tail is left untouched (partial rotary). Pair convention:
 * (x[2i], x[2i+1]) — the "interleaved" convention Qwen3 uses. */
static void rope_head(float *x, size_t pos, size_t rot_dim, float theta_base)
{
    size_t pairs = rot_dim / 2;
    for (size_t i = 0; i < pairs; i++) {
        float inv_freq = 1.0f /
            powf(theta_base, (2.0f * (float)i) / (float)rot_dim);
        float angle = (float)pos * inv_freq;
        float c = cosf(angle), s = sinf(angle);
        float x0 = x[2*i], x1 = x[2*i + 1];
        x[2*i]     = x0 * c - x1 * s;
        x[2*i + 1] = x0 * s + x1 * c;
    }
}

/* attention_f32 — single-token decode against a populated KV cache.
 *
 * Steps:
 *   1. q = x @ Wq^T, split into n_heads heads of size head_dim.
 *      k = x @ Wk^T, v = x @ Wv^T, split into n_kv heads.
 *   2. Per-head RMSNorm with q_norm / k_norm.
 *   3. RoPE on q and k at position seq_pos.
 *   4. Append k, v to k_cache[seq_pos], v_cache[seq_pos].
 *   5. For each query head h, dot it against k_cache rows 0..=seq_pos of
 *      its kv head (h mod n_kv), scaled by 1/sqrt(head_dim), softmax,
 *      then weighted sum with v_cache.
 *   6. Output concat is `y`, shape [hidden = n_heads * head_dim]. The
 *      caller multiplies by o_proj.
 *
 * `attn_scratch` must point to at least (n_heads * (seq_capacity + 1))
 * floats — workspace for the per-head score arrays.
 */
static void attention_f32(float *y,
                          const float *x,
                          const float *wq, const float *wk, const float *wv,
                          const float *q_norm, const float *k_norm,
                          float *k_cache, float *v_cache,
                          uint32_t hidden, uint32_t n_heads, uint32_t n_kv,
                          uint32_t head_dim, uint32_t seq_pos,
                          uint32_t seq_capacity,
                          float rope_theta, float partial_rotary_factor,
                          float rms_eps,
                          float *attn_scratch)
{
    const uint32_t kv_dim    = n_kv * head_dim;
    const uint32_t q_dim     = n_heads * head_dim;
    const uint32_t rot_dim   = (uint32_t)((float)head_dim * partial_rotary_factor);

    /* Project. We reuse y as scratch for q (sized hidden = q_dim). */
    float *q = y;                                 /* [q_dim]  */
    float *k_row = k_cache + (size_t)seq_pos * kv_dim; /* row in cache */
    float *v_row = v_cache + (size_t)seq_pos * kv_dim;

    matmul_f32(q,     x, wq, q_dim,  hidden);
    matmul_f32(k_row, x, wk, kv_dim, hidden);
    matmul_f32(v_row, x, wv, kv_dim, hidden);

    /* Q-norm / K-norm per head, then RoPE. */
    for (uint32_t h = 0; h < n_heads; h++) {
        float *qh = q + h * head_dim;
        rmsnorm_f32(qh, qh, q_norm, head_dim, rms_eps);
        rope_head(qh, seq_pos, rot_dim, rope_theta);
    }
    for (uint32_t h = 0; h < n_kv; h++) {
        float *kh = k_row + h * head_dim;
        rmsnorm_f32(kh, kh, k_norm, head_dim, rms_eps);
        rope_head(kh, seq_pos, rot_dim, rope_theta);
    }

    /* Attention. */
    const float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);
    for (uint32_t h = 0; h < n_heads; h++) {
        const uint32_t kvh = h * n_kv / n_heads;
        const float *qh = q + h * head_dim;
        float *scores = attn_scratch + (size_t)h * (seq_capacity + 1);

        /* Dot product with each cached k. */
        double maxv = -INFINITY;
        for (uint32_t t = 0; t <= seq_pos; t++) {
            const float *kh = k_cache + (size_t)t * kv_dim + kvh * head_dim;
            double dot = 0.0;
            for (uint32_t d = 0; d < head_dim; d++)
                dot += (double)qh[d] * kh[d];
            scores[t] = (float)dot * inv_sqrt_d;
            if (scores[t] > maxv) maxv = scores[t];
        }
        /* Softmax. */
        double sum = 0.0;
        for (uint32_t t = 0; t <= seq_pos; t++) {
            scores[t] = (float)exp((double)scores[t] - maxv);
            sum += scores[t];
        }
        float inv_sum = (float)(1.0 / sum);
        for (uint32_t t = 0; t <= seq_pos; t++) scores[t] *= inv_sum;

        /* Weighted sum of v rows → into a fresh "head_out" we keep in
         * the scratch tail (per-head, since y aliases q above). After all
         * heads are computed we write back to y. */
    }

    /* Second pass: now that all heads have scores, materialize each
     * head's output into y in place. We need the head outputs first
     * (since y was holding q above). Use the tail of attn_scratch for
     * the n_heads * head_dim staging buffer. */
    float *staging = attn_scratch + (size_t)n_heads * (seq_capacity + 1);
    for (uint32_t h = 0; h < n_heads; h++) {
        const uint32_t kvh = h * n_kv / n_heads;
        const float *scores = attn_scratch + (size_t)h * (seq_capacity + 1);
        float *out = staging + h * head_dim;
        for (uint32_t d = 0; d < head_dim; d++) {
            double acc = 0.0;
            for (uint32_t t = 0; t <= seq_pos; t++) {
                const float *vh = v_cache + (size_t)t * kv_dim + kvh * head_dim;
                acc += (double)scores[t] * vh[d];
            }
            out[d] = (float)acc;
        }
    }
    memcpy(y, staging, (size_t)q_dim * sizeof(float));
}

/* --------------------------------------------------------------------- */
/* GGUF → config + weights binding                                        */
/* --------------------------------------------------------------------- */

static int eng_get_u32(const qw36_engine *eng, const char *suffix, uint32_t *out) {
    char key[128];
    snprintf(key, sizeof(key), "%s.%s", eng->arch, suffix);
    return qw36_gguf_get_u32(eng->gguf, key, out);
}
static int eng_get_f32(const qw36_engine *eng, const char *suffix, float *out) {
    char key[128];
    snprintf(key, sizeof(key), "%s.%s", eng->arch, suffix);
    return qw36_gguf_get_f32(eng->gguf, key, out);
}

/* Materialize a small tensor (norm-sized) to fp32. Used for layernorms,
 * biases, A_log — anything small enough that the dequant pass is cheap. */
static float *bind_tensor_f32(qw36_engine *eng, const char *name) {
    qw36_gguf_tensor t;
    if (qw36_gguf_get_tensor(eng->gguf, name, &t)) return NULL;
    size_t numel = 1;
    for (uint32_t d = 0; d < t.n_dims; d++) numel *= (size_t)t.dims[d];
    float *p = materialize_f32(t.data, t.dtype, numel);
    if (!p) return NULL;
    return (float *)eng_own_(eng, p);
}

static float *bind_tensor_f32_opt(qw36_engine *eng, const char *name) {
    qw36_gguf_tensor t;
    if (qw36_gguf_get_tensor(eng->gguf, name, &t)) return NULL;
    return bind_tensor_f32(eng, name);
}

/* Materialize a lazy_w's quantized weight to a fresh fp32 host buffer
 * and flip the descriptor to point at it. Memory is registered on the
 * engine for free-on-close. Returns 0 on success, -1 on OOM/unsupported. */
static int lazy_materialize_f32(qw36_engine *eng, qw36_lazy_w *lw) {
    if (!lw || lw->dtype == QW36_DTYPE_F32) return 0;
    size_t numel = (size_t)lw->cols * (lw->rows ? lw->rows : 1);
    if (lw->n_extra) numel *= (size_t)lw->n_extra;
    float *p = materialize_f32(lw->data, lw->dtype, numel);
    if (!p) return -1;
    if (!eng_own_(eng, p)) { free(p); return -1; }
    lw->data  = p;
    lw->dtype = QW36_DTYPE_F32;
    return 0;
}

/* Lazy bind for big tensors: keep a pointer into mmap + shape + dtype,
 * to be dequantized block-by-block during matmul. */
static qw36_lazy_w *bind_tensor_lazy(qw36_engine *eng, const char *name) {
    qw36_gguf_tensor t;
    if (qw36_gguf_get_tensor(eng->gguf, name, &t)) return NULL;
    qw36_lazy_w *lw = (qw36_lazy_w *)calloc(1, sizeof(*lw));
    if (!lw) return NULL;
    lw->data      = t.data;
    lw->dtype     = t.dtype;
    lw->ggml_type = t.ggml_type;
    lw->cols      = t.n_dims >= 1 ? t.dims[0] : 0;
    lw->rows      = t.n_dims >= 2 ? t.dims[1] : 0;
    lw->n_extra   = t.n_dims >= 3 ? t.dims[2] : 0;
    return (qw36_lazy_w *)eng_own_(eng, lw);
}

/* --------------------------------------------------------------------- */
/* Engine lifecycle                                                       */
/* --------------------------------------------------------------------- */

qw36_engine *qw36_engine_open(const char *gguf_path,
                              qw36_gpu_backend *backend,
                              char *err, size_t err_cap)
{
    qw36_engine *eng = (qw36_engine *)calloc(1, sizeof(*eng));
    if (!eng) {
        if (err && err_cap) snprintf(err, err_cap, "oom");
        return NULL;
    }
    eng->backend = backend;

    eng->gguf = qw36_gguf_open(gguf_path, err, err_cap);
    if (!eng->gguf) { free(eng); return NULL; }

    /* Detect architecture. We accept any qwen3-family string. */
    const char *arch = NULL;
    if (qw36_gguf_get_str(eng->gguf, "general.architecture", &arch) || !arch) {
        if (err && err_cap) snprintf(err, err_cap, "missing general.architecture");
        qw36_engine_close(eng); return NULL;
    }
    snprintf(eng->arch, sizeof(eng->arch), "%s", arch);

    qw36_config *c = &eng->cfg;
    /* Required keys. MoE-only variants may not store feed_forward_length;
     * fall back to expert_feed_forward_length later. */
    int missing = 0;
    if (eng_get_u32(eng, "embedding_length",        &c->hidden_size))       missing |= 1;
    if (eng_get_u32(eng, "block_count",             &c->num_hidden_layers)) missing |= 2;
    if (eng_get_u32(eng, "attention.head_count",    &c->num_attention_heads)) missing |= 4;
    if (eng_get_u32(eng, "attention.head_count_kv", &c->num_key_value_heads)) missing |= 8;
    if (eng_get_u32(eng, "context_length",          &c->max_position_embeddings)) missing |= 16;
    /* Optional / fall-back: dense MLP size. MoE-only models omit this. */
    eng_get_u32(eng, "feed_forward_length", &c->intermediate_size);
    if (missing) {
        if (err && err_cap) snprintf(err, err_cap,
            "missing required %s.* config keys (mask=0x%x)", eng->arch, missing);
        qw36_engine_close(eng); return NULL;
    }
    /* head_dim — Qwen3 stores attention.key_length; fall back to hidden/heads. */
    if (eng_get_u32(eng, "attention.key_length", &c->head_dim)) {
        c->head_dim = c->hidden_size / c->num_attention_heads;
    }
    /* vocab — read from the tokens array via the gguf module. */
    {
        qw36_gguf_tensor unused;
        (void)unused;
        /* Best-effort: many qwen models also set <arch>.vocab_size. */
        if (eng_get_u32(eng, "vocab_size", &c->vocab_size) != 0) {
            /* Fallback to the embedding table row count. */
            qw36_gguf_tensor t;
            if (qw36_gguf_get_tensor(eng->gguf, "token_embd.weight", &t) == 0
                && t.n_dims >= 2) {
                c->vocab_size = (uint32_t)t.dims[1];
            }
        }
    }
    if (eng_get_f32(eng, "attention.layer_norm_rms_epsilon", &c->rms_norm_eps))
        c->rms_norm_eps = 1e-6f;
    if (eng_get_f32(eng, "rope.freq_base", &c->rope_theta))
        c->rope_theta = 1000000.0f;
    c->partial_rotary_factor = 1.0f;
    {
        float prf;
        if (eng_get_f32(eng, "rope.partial_rotary_factor", &prf) == 0 && prf > 0)
            c->partial_rotary_factor = prf;
    }
    {
        uint32_t tie = 0;
        eng_get_u32(eng, "tie_word_embeddings", &tie);
        c->tie_word_embeddings = (uint8_t)tie;
    }

    /* MoE config — optional. Picks up qwen3moe.expert_count etc. */
    eng_get_u32(eng, "expert_count",              &c->moe_num_experts);
    eng_get_u32(eng, "expert_used_count",         &c->moe_experts_per_tok);
    eng_get_u32(eng, "expert_feed_forward_length",&c->moe_intermediate_size);
    if (!c->moe_decoder_sparse_step) c->moe_decoder_sparse_step = 1;

    /* Gated DeltaNet config. Prefer the explicit GGUF SSM metadata:
     *   group_count = key heads
     *   state_size  = key/value head dim
     *   inner_size  = value_heads * value_head_dim
     * Tensor-shape fallback keeps older experimental files loadable. */
    {
        qw36_gguf_tensor t;
        uint32_t inner_size = 0;
        eng_get_u32(eng, "ssm.group_count", &c->dn_num_key_heads);
        eng_get_u32(eng, "ssm.state_size",  &c->dn_key_head_dim);
        eng_get_u32(eng, "ssm.inner_size",  &inner_size);
        if (eng_get_u32(eng, "ssm.conv_kernel", &c->dn_conv_kernel_size)) {
            if (qw36_gguf_get_tensor(eng->gguf, "blk.0.ssm_conv1d.weight", &t) == 0) {
                /* GGUF stores dims innermost-first. ssm_conv1d shape is
                 * [k, channels] in numpy -> dims[0]=k, dims[1]=channels. */
                c->dn_conv_kernel_size = (uint32_t)t.dims[0];
            }
        }
        if (inner_size && c->dn_key_head_dim) {
            c->dn_value_head_dim = c->dn_key_head_dim;
            c->dn_num_value_heads = inner_size / c->dn_value_head_dim;
        }
        if (!c->dn_num_value_heads &&
            qw36_gguf_get_tensor(eng->gguf, "blk.0.ssm_a", &t) == 0)
            c->dn_num_value_heads = (uint32_t)t.dims[0];
        if (!c->dn_value_head_dim &&
            qw36_gguf_get_tensor(eng->gguf, "blk.0.ssm_norm.weight", &t) == 0)
            c->dn_value_head_dim = (uint32_t)t.dims[0];
        if (!c->dn_key_head_dim) c->dn_key_head_dim = c->dn_value_head_dim;
        if (!c->dn_num_key_heads &&
            qw36_gguf_get_tensor(eng->gguf, "blk.0.attn_qkv.weight", &t) == 0 &&
            t.n_dims >= 2 && c->dn_key_head_dim &&
            c->dn_num_value_heads && c->dn_value_head_dim) {
            uint64_t v_dim = (uint64_t)c->dn_num_value_heads * c->dn_value_head_dim;
            if (t.dims[1] > v_dim) {
                uint64_t qk_dim = (t.dims[1] - v_dim) / 2;
                if (qk_dim && qk_dim % c->dn_key_head_dim == 0)
                    c->dn_num_key_heads = (uint32_t)(qk_dim / c->dn_key_head_dim);
            }
        }
        if (qw36_gguf_get_tensor(eng->gguf, "blk.0.ssm_conv1d.weight", &t) == 0) {
            uint64_t qkv_dim = (uint64_t)c->dn_num_key_heads * c->dn_key_head_dim * 2
                             + (uint64_t)c->dn_num_value_heads * c->dn_value_head_dim;
            if (qkv_dim && t.n_dims >= 2 && t.dims[1] != qkv_dim) {
                fprintf(stderr,
                    "qw36: warning: DeltaNet qkv channels mismatch: "
                    "config=%llu tensor=%llu\n",
                    (unsigned long long)qkv_dim,
                    (unsigned long long)t.dims[1]);
            }
        }
    }

    /* Qwen3.5/3.6 hybrid checkpoints sometimes report a head_count in
     * metadata that disagrees with the actual attn_q.weight output dim
     * (e.g. metadata says 8 but the tensor is [hidden, 16*head_dim]).
     * Trust the tensor: scan for the first vanilla layer and derive
     * num_attention_heads from its q_proj rows. Same for num_key_value_heads
     * via k_proj. */
    {
        char nm[128];
        qw36_gguf_tensor t;
        for (uint32_t l = 0; l < c->num_hidden_layers; l++) {
            snprintf(nm, sizeof(nm), "blk.%u.attn_q.weight", l);
            if (qw36_gguf_get_tensor(eng->gguf, nm, &t) == 0) {
                /* dims[1] = output dim of the projection = n_heads * head_dim */
                uint32_t real_q = (uint32_t)(t.dims[1] / c->head_dim);
                if (real_q && real_q != c->num_attention_heads) {
                    fprintf(stderr,
                        "qw36: overriding num_attention_heads %u -> %u "
                        "(from blk.%u.attn_q.weight shape)\n",
                        c->num_attention_heads, real_q, l);
                    c->num_attention_heads = real_q;
                }
                snprintf(nm, sizeof(nm), "blk.%u.attn_k.weight", l);
                if (qw36_gguf_get_tensor(eng->gguf, nm, &t) == 0) {
                    uint32_t real_kv = (uint32_t)(t.dims[1] / c->head_dim);
                    if (real_kv && real_kv != c->num_key_value_heads) {
                        fprintf(stderr,
                            "qw36: overriding num_key_value_heads %u -> %u\n",
                            c->num_key_value_heads, real_kv);
                        c->num_key_value_heads = real_kv;
                    }
                }
                break;
            }
        }
    }

    /* Bind global tensors. Small (norm) → fp32; big (embed/lm_head) → lazy. */
    qw36_weights *w = &eng->weights;
    w->dtype = QW36_DTYPE_F32;
    w->embed_tokens = bind_tensor_lazy(eng, "token_embd.weight");
    w->final_norm   = bind_tensor_f32(eng, "output_norm.weight");
    w->lm_head      = bind_tensor_lazy(eng, "output.weight");
    /* Many Qwen3 / Qwen3.5 / Qwen3.6 checkpoints omit output.weight and
     * tie it to the input embedding without setting tie_word_embeddings
     * explicitly. Fall back unconditionally — never leave lm_head NULL. */
    if (!w->lm_head) {
        w->lm_head = w->embed_tokens;
        c->tie_word_embeddings = 1;
    }
    if (!w->embed_tokens || !w->final_norm) {
        if (err && err_cap) snprintf(err, err_cap, "missing embedding or output norm");
        qw36_engine_close(eng); return NULL;
    }

    /* Bind per-layer tensors. */
    w->layers = (qw36_layer_weights *)calloc(c->num_hidden_layers,
                                             sizeof(qw36_layer_weights));
    if (!w->layers) {
        if (err && err_cap) snprintf(err, err_cap, "oom (layers)");
        qw36_engine_close(eng); return NULL;
    }
    char name[128];
    for (uint32_t l = 0; l < c->num_hidden_layers; l++) {
        qw36_layer_weights *L = &w->layers[l];
        L->dtype = QW36_DTYPE_F32;
        #define BIND_NORM(field, fmt) do { \
            snprintf(name, sizeof(name), fmt, l);             \
            L->field = bind_tensor_f32_opt(eng, name);        \
        } while (0)
        #define BIND_W(field, fmt) do { \
            snprintf(name, sizeof(name), fmt, l);             \
            L->field = bind_tensor_lazy(eng, name);           \
        } while (0)

        /* Norms are small — keep them fp32. */
        BIND_NORM(input_layernorm,     "blk.%u.attn_norm.weight");
        BIND_NORM(q_norm,              "blk.%u.attn_q_norm.weight");
        BIND_NORM(k_norm,              "blk.%u.attn_k_norm.weight");
        /* Some Qwen3.5/3.6 checkpoints name the post-attention norm
         * "post_attention_norm" (with the underscore expanded) instead of
         * "ffn_norm". Try both. */
        BIND_NORM(post_attn_layernorm, "blk.%u.ffn_norm.weight");
        if (!L->post_attn_layernorm)
            BIND_NORM(post_attn_layernorm, "blk.%u.post_attention_norm.weight");

        /* Vanilla attention projections — lazy (big). */
        BIND_W(q_proj,    "blk.%u.attn_q.weight");
        BIND_W(k_proj,    "blk.%u.attn_k.weight");
        BIND_W(v_proj,    "blk.%u.attn_v.weight");
        BIND_W(o_proj,    "blk.%u.attn_output.weight");

        /* Gated DeltaNet projections. attn_qkv / attn_gate are big (lazy);
         * the rest are small enough to materialize. conv1d is depthwise
         * (kernel × channels = a few thousand floats), keep as fp32. */
        BIND_W(dn_qkv,           "blk.%u.attn_qkv.weight");
        BIND_W(dn_gate,          "blk.%u.attn_gate.weight");
        BIND_W(dn_alpha,         "blk.%u.ssm_alpha.weight");
        BIND_W(dn_beta,          "blk.%u.ssm_beta.weight");
        BIND_NORM(dn_conv1d,     "blk.%u.ssm_conv1d.weight");
        BIND_NORM(dn_dt_bias,    "blk.%u.ssm_dt.bias");
        BIND_NORM(dn_a_log,      "blk.%u.ssm_a");
        if (L->dn_a_log && c->dn_num_value_heads) {
            float *a = (float *)L->dn_a_log;
            for (uint32_t i = 0; i < c->dn_num_value_heads; i++) {
                float v = fabsf(a[i]);
                if (v < 1.0e-10f) v = 1.0e-10f;
                a[i] = logf(v);
            }
        }
        BIND_NORM(dn_norm,       "blk.%u.ssm_norm.weight");
        BIND_W(dn_out,           "blk.%u.ssm_out.weight");

        /* Dense MLP — lazy. */
        BIND_W(gate_proj, "blk.%u.ffn_gate.weight");
        BIND_W(up_proj,   "blk.%u.ffn_up.weight");
        BIND_W(down_proj, "blk.%u.ffn_down.weight");

        /* MoE — lazy. The expert stacks are 3D [n_experts, hidden, inter]
         * and qw36_lazy_w records that via rows/cols/n_extra. */
        BIND_W(moe_router,        "blk.%u.ffn_gate_inp.weight");
        BIND_W(moe_expert_gate,   "blk.%u.ffn_gate_exps.weight");
        BIND_W(moe_expert_up,     "blk.%u.ffn_up_exps.weight");
        BIND_W(moe_expert_down,   "blk.%u.ffn_down_exps.weight");
        BIND_W(moe_shared_gate,   "blk.%u.ffn_gate_shexp.weight");
        BIND_W(moe_shared_up,     "blk.%u.ffn_up_shexp.weight");
        BIND_W(moe_shared_down,   "blk.%u.ffn_down_shexp.weight");
        #undef BIND_NORM
        #undef BIND_W
    }

    /* When a GPU backend is provided, eagerly materialize every lazy_w
     * tensor to fp32 in host memory and (on Apple Silicon / unified memory
     * backends) ask the backend to wrap that buffer as a device view.
     *
     * Memory budget: a 0.8B model materializes to ~3 GB fp32, fits easily.
     * A 35B-A3B materializes to ~140 GB — caller must use the CPU build
     * for that until streaming dequant is added. */
    if (backend) {
        eng->ctx = backend->init(err, err_cap);
        if (!eng->ctx) { qw36_engine_close(eng); return NULL; }

        qw36_lazy_w *lws[] = {
            (qw36_lazy_w *)w->embed_tokens,
            (qw36_lazy_w *)w->lm_head,
            NULL
        };
        for (size_t i = 0; lws[i]; i++) {
            if (lazy_materialize_f32(eng, lws[i])) {
                if (err && err_cap) snprintf(err, err_cap,
                    "%s: backend materialize failed (unsupported dtype %d)",
                    backend->name, lws[i]->ggml_type);
                qw36_engine_close(eng); return NULL;
            }
        }
        /* lm_head usually aliases embed_tokens — avoid double-binding. */
        if (w->lm_head == w->embed_tokens) ((qw36_lazy_w *)w->lm_head)->gpu_buf =
            ((qw36_lazy_w *)w->embed_tokens)->gpu_buf;

        #define MAT(field) do { \
            qw36_lazy_w *lw = (qw36_lazy_w *)L->field; \
            if (lw && lazy_materialize_f32(eng, lw)) { \
                if (err && err_cap) snprintf(err, err_cap, \
                    "%s: backend materialize failed at blk.%u." #field, \
                    backend->name, l); \
                qw36_engine_close(eng); return NULL; \
            } \
        } while (0)
        for (uint32_t l = 0; l < c->num_hidden_layers; l++) {
            qw36_layer_weights *L = &eng->weights.layers[l];
            MAT(q_proj); MAT(k_proj); MAT(v_proj); MAT(o_proj);
            MAT(dn_qkv); MAT(dn_gate); MAT(dn_alpha); MAT(dn_beta); MAT(dn_out);
            MAT(gate_proj); MAT(up_proj); MAT(down_proj);
            MAT(moe_router);
            MAT(moe_expert_gate); MAT(moe_expert_up); MAT(moe_expert_down);
            MAT(moe_shared_gate); MAT(moe_shared_up); MAT(moe_shared_down);
        }
        #undef MAT

        /* Now wrap every materialized buffer as a device-visible view.
         * For Apple Metal this is zero-copy via MTLBuffer NoCopy; for
         * CUDA/HIP this does an actual device copy. */
        #define UPLOAD(field) do { \
            qw36_lazy_w *lw = (qw36_lazy_w *)L->field; \
            if (lw && lw->data && !lw->gpu_buf) { \
                size_t numel = (size_t)lw->cols * (lw->rows ? lw->rows : 1); \
                if (lw->n_extra) numel *= (size_t)lw->n_extra; \
                lw->gpu_buf = backend->upload(eng->ctx, lw->data, \
                    numel * sizeof(float), QW36_DTYPE_F32); \
            } \
        } while (0)
        for (uint32_t l = 0; l < c->num_hidden_layers; l++) {
            qw36_layer_weights *L = &eng->weights.layers[l];
            UPLOAD(q_proj); UPLOAD(k_proj); UPLOAD(v_proj); UPLOAD(o_proj);
            UPLOAD(dn_qkv); UPLOAD(dn_gate); UPLOAD(dn_alpha);
            UPLOAD(dn_beta); UPLOAD(dn_out);
            UPLOAD(gate_proj); UPLOAD(up_proj); UPLOAD(down_proj);
            UPLOAD(moe_router);
            UPLOAD(moe_expert_gate); UPLOAD(moe_expert_up); UPLOAD(moe_expert_down);
            UPLOAD(moe_shared_gate); UPLOAD(moe_shared_up); UPLOAD(moe_shared_down);
        }
        #undef UPLOAD
        /* Globals. */
        if (w->embed_tokens) {
            qw36_lazy_w *lw = (qw36_lazy_w *)w->embed_tokens;
            size_t numel = (size_t)lw->cols * lw->rows;
            lw->gpu_buf = backend->upload(eng->ctx, lw->data,
                numel * sizeof(float), QW36_DTYPE_F32);
        }
        if (w->lm_head && w->lm_head != w->embed_tokens) {
            qw36_lazy_w *lw = (qw36_lazy_w *)w->lm_head;
            size_t numel = (size_t)lw->cols * lw->rows;
            lw->gpu_buf = backend->upload(eng->ctx, lw->data,
                numel * sizeof(float), QW36_DTYPE_F32);
        } else if (w->lm_head == w->embed_tokens) {
            ((qw36_lazy_w *)w->lm_head)->gpu_buf =
                ((qw36_lazy_w *)w->embed_tokens)->gpu_buf;
        }
    }
    return eng;
}

void qw36_engine_close(qw36_engine *eng)
{
    if (!eng) return;
    if (eng->backend && eng->backend->destroy && eng->ctx)
        eng->backend->destroy(eng->ctx);
    if (eng->owned) {
        for (size_t i = 0; i < eng->owned_n; i++) free(eng->owned[i]);
        free(eng->owned);
    }
    free(eng->cfg.layer_types);
    free(eng->weights.layers);
    if (eng->gguf) qw36_gguf_close(eng->gguf);
    free(eng);
}

const qw36_config  *qw36_engine_config(const qw36_engine *eng)  { return &eng->cfg; }
const qw36_weights *qw36_engine_weights(const qw36_engine *eng) { return &eng->weights; }
const struct qw36_gguf_file *qw36_engine_gguf(const qw36_engine *eng) { return eng->gguf; }

/* --------------------------------------------------------------------- */
/* State                                                                  */
/* --------------------------------------------------------------------- */

qw36_state *qw36_state_new(const qw36_engine *eng, uint32_t seq_capacity)
{
    const qw36_config *c = &eng->cfg;
    qw36_state *st = (qw36_state *)calloc(1, sizeof(*st));
    if (!st) return NULL;
    st->seq_capacity = seq_capacity;
    st->seq_pos      = 0;
    st->kv_dtype     = QW36_DTYPE_F32;
    st->num_layers   = c->num_hidden_layers;

    const size_t kv_dim   = (size_t)c->num_key_value_heads * c->head_dim;
    const size_t q_dim    = (size_t)c->num_attention_heads * c->head_dim;
    const size_t hidden   = c->hidden_size;
    const size_t inter    = c->intermediate_size ? c->intermediate_size : 1;
    const size_t vocab    = c->vocab_size;
    const size_t L        = c->num_hidden_layers;

    st->k_cache = (void **)calloc(L, sizeof(void *));
    st->v_cache = (void **)calloc(L, sizeof(void *));
    if (!st->k_cache || !st->v_cache) goto fail;
    for (size_t l = 0; l < L; l++) {
        st->k_cache[l] = calloc((size_t)seq_capacity * kv_dim, sizeof(float));
        st->v_cache[l] = calloc((size_t)seq_capacity * kv_dim, sizeof(float));
        if (!st->k_cache[l] || !st->v_cache[l]) goto fail;
    }

    /* DeltaNet per-layer state. Sized to the conv kernel size and the
     * key/val dims read from config. Zero for layers without DeltaNet —
     * we allocate symmetrically across L for simplicity. */
    if (c->dn_num_value_heads) {
        const size_t qkv_dim = (size_t)c->dn_num_key_heads   * c->dn_key_head_dim
                             + (size_t)c->dn_num_key_heads   * c->dn_key_head_dim
                             + (size_t)c->dn_num_value_heads * c->dn_value_head_dim;
        const size_t conv_window = c->dn_conv_kernel_size
                                 ? (size_t)c->dn_conv_kernel_size - 1 : 0;
        const size_t s_elems = (size_t)c->dn_num_value_heads
                             * c->dn_key_head_dim * c->dn_value_head_dim;
        st->conv_state  = (float **)calloc(L, sizeof(float *));
        st->delta_state = (float **)calloc(L, sizeof(float *));
        if (!st->conv_state || !st->delta_state) goto fail;
        for (size_t l = 0; l < L; l++) {
            st->conv_state[l]  = (float *)calloc(conv_window * qkv_dim, sizeof(float));
            st->delta_state[l] = (float *)calloc(s_elems, sizeof(float));
            if ((conv_window && !st->conv_state[l]) || !st->delta_state[l])
                goto fail;
        }
    }

    /* Scratch — big enough for attention staging (n_heads*(cap+1) +
     * n_heads*head_dim) and for MLP gate/up. */
    const size_t attn_scratch_n =
        (size_t)c->num_attention_heads * ((size_t)seq_capacity + 1)
      + (size_t)c->num_attention_heads * c->head_dim;
    st->x           = (float *)calloc(hidden,        sizeof(float));
    st->x_rms       = (float *)calloc(hidden,        sizeof(float));
    st->q           = (float *)calloc(q_dim,         sizeof(float));
    st->k           = (float *)calloc(kv_dim,        sizeof(float));
    st->v           = (float *)calloc(kv_dim,        sizeof(float));
    st->attn_scores = (float *)calloc(attn_scratch_n, sizeof(float));
    st->gate        = (float *)calloc(inter,         sizeof(float));
    st->up          = (float *)calloc(inter,         sizeof(float));
    st->logits      = (float *)calloc(vocab,         sizeof(float));
    /* row_scratch is used by matmul_lazy. Size = max(hidden, inter, vocab)
     * since matmul reads one full row of the weight at a time. */
    size_t rs_n = hidden;
    if (inter > rs_n) rs_n = inter;
    if (vocab > rs_n) rs_n = vocab;
    /* Note: q_dim/kv_dim are smaller than hidden in practice (n_heads *
     * head_dim ≤ hidden), so the existing bound suffices for QKV matmuls. */
    /* We stash the row scratch in attn_scores' tail — simpler than adding
     * a public field. Realloc attn_scores to fit. */
    free(st->attn_scores);
    st->attn_scores = (float *)calloc(attn_scratch_n + rs_n, sizeof(float));
    if (!st->x || !st->x_rms || !st->q || !st->k || !st->v ||
        !st->attn_scores || !st->gate || !st->up || !st->logits) goto fail;

    return st;

fail:
    qw36_state_free(st);
    return NULL;
}

void qw36_state_free(qw36_state *st)
{
    if (!st) return;
    if (st->k_cache) {
        for (uint32_t l = 0; l < st->num_layers; l++) free(st->k_cache[l]);
        free(st->k_cache);
    }
    if (st->v_cache) {
        for (uint32_t l = 0; l < st->num_layers; l++) free(st->v_cache[l]);
        free(st->v_cache);
    }
    if (st->conv_state) {
        for (uint32_t l = 0; l < st->num_layers; l++) free(st->conv_state[l]);
        free(st->conv_state);
    }
    if (st->delta_state) {
        for (uint32_t l = 0; l < st->num_layers; l++) free(st->delta_state[l]);
        free(st->delta_state);
    }
    free(st->x); free(st->x_rms); free(st->q); free(st->k); free(st->v);
    free(st->attn_scores); free(st->gate); free(st->up); free(st->logits);
    free(st);
}

/* --------------------------------------------------------------------- */
/* Forward — CPU reference per-layer loop.                                */
/* --------------------------------------------------------------------- */

int qw36_forward(qw36_engine *eng, qw36_state *st, uint32_t token)
{
    if (!eng || !st) return -1;
    const qw36_config  *c = &eng->cfg;
    const qw36_weights *w = &eng->weights;
    if (token >= c->vocab_size) return -1;
    if (st->seq_pos >= st->seq_capacity) return -1;

    /* Publish the engine pointer so matmul_lazy and friends can locate the
     * GPU backend without explicit parameter threading. Restored on exit. */
    qw36_engine *prev_active = qw36_active_engine;
    qw36_active_engine = eng;

    const size_t hidden = c->hidden_size;
    const size_t inter  = c->intermediate_size;

    /* Compute the row_scratch slot inside attn_scores tail. */
    const size_t attn_scratch_n =
        (size_t)c->num_attention_heads * ((size_t)st->seq_capacity + 1)
      + (size_t)c->num_attention_heads * c->head_dim;
    float *row_scratch = st->attn_scores + attn_scratch_n;

    /* Embed: dequant row `token` from embed_tokens. */
    if (embed_lookup_lazy((const qw36_lazy_w *)w->embed_tokens, token, st->x))
        return -3;

    /* Optional debug: per-layer norm trace, enabled by env var. */
    static int debug_layer = -1;
    if (debug_layer < 0) {
        const char *e = getenv("QW36_DEBUG_LAYER");
        debug_layer = e && atoi(e) ? 1 : 0;
    }
    if (debug_layer) {
        double ss = 0.0;
        for (size_t i = 0; i < hidden; i++) ss += (double)st->x[i] * st->x[i];
        fprintf(stderr, "[layer 0 in]  ||x||=%.4f x[0..3]=%.3f %.3f %.3f\n",
                sqrt(ss), st->x[0], st->x[1], st->x[2]);
    }

    for (uint32_t l = 0; l < c->num_hidden_layers; l++) {
        const qw36_layer_weights *L = &w->layers[l];
        float *x = st->x;

        if (!L->q_proj && L->dn_qkv) {
            /* === Gated DeltaNet branch === */
            const uint32_t n_v   = c->dn_num_value_heads;
            const uint32_t n_k   = c->dn_num_key_heads;
            const uint32_t kd    = c->dn_key_head_dim;
            const uint32_t vd    = c->dn_value_head_dim;
            const uint32_t q_dim = n_k * kd, k_dim_t = n_k * kd, v_dim = n_v * vd;
            const uint32_t qkv_dim = q_dim + k_dim_t + v_dim;

            rmsnorm_f32(st->x_rms, x, (const float *)L->input_layernorm,
                        hidden, c->rms_norm_eps);

            /* Projections.  Heap-allocate to keep the stack frame small. */
            const size_t need = (size_t)2 * qkv_dim   /* qkv + qkv_act */
                              + (size_t)v_dim          /* z_proj */
                              + (size_t)n_v * 2        /* alpha + beta */
                              + (size_t)v_dim          /* gout */
                              + (size_t)kd * 2;        /* qb + kb */
            float *blk = (float *)calloc(need, sizeof(float));
            if (!blk) return -3;
            float *qkv     = blk;
            float *qkv_act = qkv + qkv_dim;
            float *z_proj  = qkv_act + qkv_dim;
            float *alpha   = z_proj + v_dim;
            float *beta    = alpha + n_v;
            float *gout    = beta + n_v;
            float *qb      = gout + v_dim;
            float *kb      = qb + kd;

            int dn_rc = 0;
            dn_rc |= matmul_lazy(qkv,    st->x_rms, (const qw36_lazy_w *)L->dn_qkv,   row_scratch);
            dn_rc |= matmul_lazy(z_proj, st->x_rms, (const qw36_lazy_w *)L->dn_gate,  row_scratch);
            dn_rc |= matmul_lazy(alpha,  st->x_rms, (const qw36_lazy_w *)L->dn_alpha, row_scratch);
            dn_rc |= matmul_lazy(beta,   st->x_rms, (const qw36_lazy_w *)L->dn_beta,  row_scratch);
            if (dn_rc) {
                fprintf(stderr, "qw36: DN projection failed at layer %u "
                        "(unsupported dtype in attn_qkv/gate/alpha/beta — "
                        "ggml types %d/%d/%d/%d)\n", l,
                        ((const qw36_lazy_w *)L->dn_qkv)->ggml_type,
                        ((const qw36_lazy_w *)L->dn_gate)->ggml_type,
                        ((const qw36_lazy_w *)L->dn_alpha)->ggml_type,
                        ((const qw36_lazy_w *)L->dn_beta)->ggml_type);
                free(blk);
                return -4;
            }

            /* Conv1d + SiLU on the QKV channels (depthwise causal).
             * QW36_SKIP_CONV1D=1 forwards qkv directly (just silu, no conv).*/
            static int skip_conv = -1;
            if (skip_conv < 0) {
                const char *e = getenv("QW36_SKIP_CONV1D");
                skip_conv = e && atoi(e) ? 1 : 0;
            }
            if (skip_conv) {
                for (uint32_t i = 0; i < qkv_dim; i++) qkv_act[i] = silu(qkv[i]);
            } else {
                conv1d_silu_decode(qkv, (const float *)L->dn_conv1d,
                                   st->conv_state[l], qkv_act,
                                   qkv_dim, c->dn_conv_kernel_size);
            }

            /* Gated delta rule step. */
            gated_delta_decode(qkv_act, beta, alpha,
                               (const float *)L->dn_dt_bias,
                               (const float *)L->dn_a_log,
                               st->delta_state[l], gout,
                               n_k, n_v, kd, vd, qb, kb);

            /* Per-value-head gated RMSNorm + silu(z) gating:
             *   gout[v,d] = silu(z_proj[v,d]) * (gout[v,d] *
             *               rsqrt(mean(gout[v]²)+eps) * dn_norm.weight[d]) */
            const float *dn_norm_w = (const float *)L->dn_norm;
            for (uint32_t v = 0; v < n_v; v++) {
                float *go = gout   + (size_t)v * vd;
                float *zp = z_proj + (size_t)v * vd;
                rmsnorm_f32(go, go, dn_norm_w, vd, c->rms_norm_eps);
                for (uint32_t d = 0; d < vd; d++) go[d] = silu(zp[d]) * go[d];
            }

            /* Out projection: hidden = ssm_out @ gout. */
            if (matmul_lazy(st->x_rms, gout,
                        (const qw36_lazy_w *)L->dn_out, row_scratch)) {
                fprintf(stderr, "qw36: DN out projection failed at layer %u "
                        "(unsupported dtype in ssm_out — ggml type %d)\n",
                        l, ((const qw36_lazy_w *)L->dn_out)->ggml_type);
                free(blk);
                return -5;
            }
            /* Debug knob: QW36_SKIP_DN=1 zeroes the DN contribution to
             * isolate whether the bug is in the DeltaNet path. */
            static int skip_dn = -1;
            if (skip_dn < 0) {
                const char *e = getenv("QW36_SKIP_DN");
                skip_dn = e && atoi(e) ? 1 : 0;
            }
            if (!skip_dn) {
                for (size_t i = 0; i < hidden; i++) x[i] += st->x_rms[i];
            }
            free(blk);

            /* MLP block (shared with full-attention path below). */
            goto mlp_block;
        }

        if (!L->q_proj) {
            /* Layer is neither vanilla nor DeltaNet — unknown variant. */
            return -2;
        }

        /* --- vanilla full-attention block --- */
        rmsnorm_f32(st->x_rms, x, (const float *)L->input_layernorm,
                    hidden, c->rms_norm_eps);
        if (debug_layer) {
            double ss = 0.0;
            for (size_t i = 0; i < hidden; i++) ss += (double)st->x_rms[i] * st->x_rms[i];
            fprintf(stderr, "[L%u attn x_rms ||=%.4f] norm_w[0..2]=%.4f %.4f %.4f\n",
                    l, sqrt(ss),
                    ((const float*)L->input_layernorm)[0],
                    ((const float*)L->input_layernorm)[1],
                    ((const float*)L->input_layernorm)[2]);
        }

        const uint32_t kv_dim = c->num_key_value_heads * c->head_dim;
        const uint32_t rot_dim = (uint32_t)((float)c->head_dim *
                                            c->partial_rotary_factor);

        /* QKV projections via lazy matmul. */
        int v_rc = 0;
        v_rc |= matmul_lazy(st->q, st->x_rms,
                            (const qw36_lazy_w *)L->q_proj, row_scratch);
        float *k_row = (float *)st->k_cache[l] + (size_t)st->seq_pos * kv_dim;
        float *v_row = (float *)st->v_cache[l] + (size_t)st->seq_pos * kv_dim;
        v_rc |= matmul_lazy(k_row, st->x_rms,
                            (const qw36_lazy_w *)L->k_proj, row_scratch);
        v_rc |= matmul_lazy(v_row, st->x_rms,
                            (const qw36_lazy_w *)L->v_proj, row_scratch);
        if (v_rc) {
            fprintf(stderr, "qw36: vanilla QKV failed at layer %u "
                    "(ggml types %d/%d/%d)\n", l,
                    ((const qw36_lazy_w *)L->q_proj)->ggml_type,
                    ((const qw36_lazy_w *)L->k_proj)->ggml_type,
                    ((const qw36_lazy_w *)L->v_proj)->ggml_type);
            return -6;
        }

        /* Per-head q_norm/k_norm + RoPE. */
        for (uint32_t h = 0; h < c->num_attention_heads; h++) {
            float *qh = st->q + h * c->head_dim;
            rmsnorm_f32(qh, qh, (const float *)L->q_norm,
                        c->head_dim, c->rms_norm_eps);
            rope_head(qh, st->seq_pos, rot_dim, c->rope_theta);
        }
        for (uint32_t h = 0; h < c->num_key_value_heads; h++) {
            float *kh = k_row + h * c->head_dim;
            rmsnorm_f32(kh, kh, (const float *)L->k_norm,
                        c->head_dim, c->rms_norm_eps);
            rope_head(kh, st->seq_pos, rot_dim, c->rope_theta);
        }

        /* Attention: for each head h, score against k_cache[0..=seq_pos]
         * of kv_head (h * n_kv / n_heads, GQA group-replicate), softmax,
         * weighted sum of v_cache. */
        const float inv_sqrt_d = 1.0f / sqrtf((float)c->head_dim);
        float *staging = st->attn_scores
                       + (size_t)c->num_attention_heads * (st->seq_capacity + 1);
        for (uint32_t h = 0; h < c->num_attention_heads; h++) {
            const uint32_t kvh = h * c->num_key_value_heads
                               / c->num_attention_heads;
            const float *qh = st->q + h * c->head_dim;
            float *scores = st->attn_scores
                          + (size_t)h * (st->seq_capacity + 1);
            float maxv = -INFINITY;
            for (uint32_t t = 0; t <= st->seq_pos; t++) {
                const float *kh = (float *)st->k_cache[l]
                                + (size_t)t * kv_dim + kvh * c->head_dim;
                double dot = 0.0;
                for (uint32_t d = 0; d < c->head_dim; d++)
                    dot += (double)qh[d] * kh[d];
                scores[t] = (float)dot * inv_sqrt_d;
                if (scores[t] > maxv) maxv = scores[t];
            }
            double sum = 0.0;
            for (uint32_t t = 0; t <= st->seq_pos; t++) {
                scores[t] = expf(scores[t] - maxv);
                sum += scores[t];
            }
            float inv_sum = (float)(1.0 / sum);
            float *head_out = staging + h * c->head_dim;
            for (uint32_t d = 0; d < c->head_dim; d++) {
                double acc = 0.0;
                for (uint32_t t = 0; t <= st->seq_pos; t++) {
                    const float *vh = (float *)st->v_cache[l]
                                    + (size_t)t * kv_dim + kvh * c->head_dim;
                    acc += (double)(scores[t] * inv_sum) * vh[d];
                }
                head_out[d] = (float)acc;
            }
        }
        /* o_proj */
        matmul_lazy(st->x_rms, staging,
                    (const qw36_lazy_w *)L->o_proj, row_scratch);
        for (size_t i = 0; i < hidden; i++) x[i] += st->x_rms[i];

mlp_block:
        if (debug_layer) {
            double ss = 0.0;
            for (size_t i = 0; i < hidden; i++) ss += (double)x[i] * x[i];
            fprintf(stderr, "[L%u post-attn] ||x||=%.4f\n", l, sqrt(ss));
        }
        /* --- mlp block (shared by vanilla and DeltaNet branches) --- */
        rmsnorm_f32(st->x_rms, x, (const float *)L->post_attn_layernorm,
                    hidden, c->rms_norm_eps);

        if (L->moe_router) {
            /* MoE path. Expert intermediate may be smaller than dense
             * intermediate; we use moe_intermediate_size from config. */
            const uint32_t mi = c->moe_intermediate_size
                              ? c->moe_intermediate_size
                              : ((const qw36_lazy_w *)L->moe_expert_gate)->rows;
            /* Scratch overlay: use part of attn_scores tail past row_scratch.
             * We sized row_scratch for max(hidden, inter, vocab); for safety,
             * append a fresh allocation here. */
            const size_t need = (size_t)c->moe_num_experts
                              + (size_t)c->moe_experts_per_tok * 2  /* val+idx */
                              + (size_t)mi * 2
                              + (size_t)hidden;
            float *moe_scratch = (float *)alloca(need * sizeof(float));
            if (moe_forward_f32(st->x_rms, st->x_rms,
                                (const qw36_lazy_w *)L->moe_router,
                                (const qw36_lazy_w *)L->moe_expert_gate,
                                (const qw36_lazy_w *)L->moe_expert_up,
                                (const qw36_lazy_w *)L->moe_expert_down,
                                (const qw36_lazy_w *)L->moe_shared_gate,
                                (const qw36_lazy_w *)L->moe_shared_up,
                                (const qw36_lazy_w *)L->moe_shared_down,
                                hidden, mi,
                                c->moe_num_experts,
                                (int)c->moe_experts_per_tok,
                                c->moe_norm_topk_prob,
                                moe_scratch, row_scratch) == 0)
            {
                for (size_t i = 0; i < hidden; i++) x[i] += st->x_rms[i];
            }
        } else if (L->gate_proj && L->up_proj && L->down_proj) {
            matmul_lazy(st->gate, st->x_rms,
                        (const qw36_lazy_w *)L->gate_proj, row_scratch);
            matmul_lazy(st->up,   st->x_rms,
                        (const qw36_lazy_w *)L->up_proj,   row_scratch);
            for (size_t i = 0; i < inter; i++)
                st->gate[i] = silu(st->gate[i]) * st->up[i];
            matmul_lazy(st->x_rms, st->gate,
                        (const qw36_lazy_w *)L->down_proj, row_scratch);
            for (size_t i = 0; i < hidden; i++) x[i] += st->x_rms[i];
        }
        if (debug_layer) {
            double ss = 0.0;
            for (size_t i = 0; i < hidden; i++) ss += (double)x[i] * x[i];
            fprintf(stderr, "[L%u out]  ||x||=%.4f\n", l, sqrt(ss));
        }
    }

    /* Final norm + lm_head (lazy). */
    rmsnorm_f32(st->x_rms, st->x, (const float *)w->final_norm,
                hidden, c->rms_norm_eps);
    matmul_lazy(st->logits, st->x_rms,
                (const qw36_lazy_w *)w->lm_head, row_scratch);

    st->seq_pos++;
    qw36_active_engine = prev_active;
    return 0;
}

int qw36_prefill(qw36_engine *eng, qw36_state *st,
                 const uint32_t *tokens, size_t length)
{
    for (size_t i = 0; i < length; i++) {
        int rc = qw36_forward(eng, st, tokens[i]);
        if (rc) return rc;
    }
    return 0;
}

/* --------------------------------------------------------------------- */
/* Sampling                                                               */
/* --------------------------------------------------------------------- */

static uint64_t splitmix64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

uint32_t qw36_sample(const float *logits, uint32_t vocab, qw36_sampler *s)
{
    if (s->temperature <= 0.0f) {
        uint32_t best = 0;
        float bv = logits[0];
        for (uint32_t i = 1; i < vocab; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
        return best;
    }
    double maxv = logits[0];
    for (uint32_t i = 1; i < vocab; i++) if (logits[i] > maxv) maxv = logits[i];
    double sum = 0.0;
    double *p = (double *)malloc(sizeof(double) * vocab);
    if (!p) return 0;
    for (uint32_t i = 0; i < vocab; i++) {
        p[i] = exp(((double)logits[i] - maxv) / (double)s->temperature);
        sum += p[i];
    }
    double r = (double)(splitmix64(&s->rng_seed) >> 11) /
               (double)(1ULL << 53) * sum;
    double acc = 0.0;
    uint32_t pick = vocab - 1;
    for (uint32_t i = 0; i < vocab; i++) {
        acc += p[i];
        if (acc >= r) { pick = i; break; }
    }
    free(p);
    return pick;
}
