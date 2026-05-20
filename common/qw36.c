/* qw36.c — CPU reference forward pass + sampling + engine lifecycle.
 *
 * Owner: Claude.
 *
 * This file is the *reference*. Every GPU backend must produce fp32
 * output that matches this within tolerance on the golden vectors in
 * tests/. Keep it readable over fast — readability is the whole point.
 */

#include "qw36_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QW36_VERSION_STR "qw36 0.0.1-scaffold"

const char *qw36_version(void) { return QW36_VERSION_STR; }

void *qw36__eng_own(qw36_engine *eng, void *p) {
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

uint16_t qw36__f32_to_f16(float f) {
    union { float f; uint32_t u; } in;
    in.f = f;
    uint32_t x = in.u;
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t mant = x & 0x007fffffu;
    int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127 + 15;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x00800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) {
        if (mant == 0) return (uint16_t)(sign | 0x7c00u);
        return (uint16_t)(sign | 0x7c00u | (mant >> 13) | 1u);
    }

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x00001000u) half++;
    return (uint16_t)half;
}

size_t qw36__dtype_nbytes(qw36_dtype dtype) {
    switch (dtype) {
        case QW36_DTYPE_F32:  return 4;
        case QW36_DTYPE_F16:  return 2;
        case QW36_DTYPE_BF16: return 2;
        default: return 0;
    }
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
float *qw36__materialize_f32(const void *data, qw36_dtype dt, size_t n) {
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
float qw36__silu(float x);

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

/* Set/cleared by qw36_forward so qw36__matmul_lazy can locate the backend
 * without threading the engine pointer through every helper signature.
 * Single-threaded today; promote to __thread when batched/multi-stream. */
qw36_engine *qw36__active_engine = NULL;
int qw36__skip_logits_this_forward = 0;

static int active_backend(qw36_gpu_backend **be_out, qw36_gpu_ctx **ctx_out)
{
    qw36_engine *eng = qw36__active_engine;
    if (!eng || !eng->backend || !eng->ctx) return 0;
    qw36_gpu_backend *be = eng->backend;
    if (!be->upload || !be->download || !be->alloc || !be->free) return 0;
    if (be_out) *be_out = be;
    if (ctx_out) *ctx_out = eng->ctx;
    return 1;
}

qw36_gpu_buf *qw36__gpu_cached_upload(qw36_engine *eng, const void *host,
                                        size_t bytes, qw36_dtype dtype)
{
    if (!eng || !eng->backend || !eng->ctx || !eng->backend->upload ||
        !host || !bytes)
        return NULL;
    for (size_t i = 0; i < eng->gpu_cache_n; i++) {
        qw36_gpu_cache_entry *e = &eng->gpu_cache[i];
        if (e->host == host && e->bytes == bytes && e->dtype == dtype)
            return e->gpu_buf;
    }
    if (eng->gpu_cache_n >= eng->gpu_cache_cap) {
        size_t nc = eng->gpu_cache_cap ? eng->gpu_cache_cap * 2 : 128;
        qw36_gpu_cache_entry *arr =
            (qw36_gpu_cache_entry *)realloc(eng->gpu_cache,
                                            nc * sizeof(*arr));
        if (!arr) return NULL;
        eng->gpu_cache = arr;
        eng->gpu_cache_cap = nc;
    }
    qw36_gpu_buf *gb = eng->backend->upload(eng->ctx, host, bytes, dtype);
    if (!gb) return NULL;
    eng->gpu_cache[eng->gpu_cache_n++] = (qw36_gpu_cache_entry){
        host, bytes, dtype, gb
    };
    return gb;
}

void qw36__gpu_cache_free(qw36_engine *eng)
{
    if (!eng) return;
    if (eng->backend && eng->backend->free && eng->ctx) {
        for (size_t i = 0; i < eng->gpu_cache_n; i++)
            eng->backend->free(eng->ctx, eng->gpu_cache[i].gpu_buf);
    }
    free(eng->gpu_cache);
    eng->gpu_cache = NULL;
    eng->gpu_cache_n = eng->gpu_cache_cap = 0;
}

int qw36__state_backend(qw36_state *st, qw36_gpu_backend **be_out,
                         qw36_gpu_ctx **ctx_out)
{
    if (!st || !st->gpu_backend || !st->gpu_ctx) return 0;
    qw36_gpu_backend *be = (qw36_gpu_backend *)st->gpu_backend;
    qw36_gpu_ctx *ctx = (qw36_gpu_ctx *)st->gpu_ctx;
    if (!be->alloc || !be->free || !be->download || !be->copy_from_host)
        return 0;
    if (be_out) *be_out = be;
    if (ctx_out) *ctx_out = ctx;
    return 1;
}

int qw36__state_copy_from_host(qw36_state *st, void *dst_dev,
                                const void *src, size_t bytes)
{
    qw36_gpu_backend *be;
    qw36_gpu_ctx *ctx;
    if (!dst_dev || !src || !qw36__state_backend(st, &be, &ctx)) return -1;
    be->copy_from_host(ctx, (qw36_gpu_buf *)dst_dev, src, bytes);
    return 0;
}

int qw36__state_download_to_host(qw36_state *st, void *src_dev,
                                  void *dst, size_t bytes)
{
    qw36_gpu_backend *be;
    qw36_gpu_ctx *ctx;
    if (!src_dev || !dst || !qw36__state_backend(st, &be, &ctx)) return -1;
    be->download(ctx, (qw36_gpu_buf *)src_dev, dst, bytes);
    return 0;
}

/* matmul against a lazy quantized weight: y[r] = sum_c W[r,c] * x[c].
 * Dispatches to the GPU backend when one is attached and the weight has
 * an uploaded gpu_buf; otherwise falls back to per-row CPU dequant + dot.
 *
 * Buffer reuse: rather than alloc/free a fresh x_dev / y_dev per matmul
 * (which dominates the per-token latency at ~10 tok/s), we maintain a
 * per-engine pool of scratch device buffers indexed by byte size. The
 * largest scratch slots are sized to match max(hidden, intermediate,
 * vocab) * sizeof(float). */
int qw36__matmul_lazy(float *y, const float *x, const qw36_lazy_w *w,
                       float *row_scratch)
{
    const size_t rows = (size_t)w->rows;
    const size_t cols = (size_t)w->cols;

    qw36_engine *eng = qw36__active_engine;
    if (eng && eng->backend && eng->backend->matmul &&
        eng->backend->upload && eng->backend->download &&
        eng->backend->alloc && eng->backend->free &&
        w->gpu_buf)
    {
        qw36_gpu_backend *be = eng->backend;
        qw36_gpu_ctx     *ctx = eng->ctx;
        const size_t x_bytes = cols * sizeof(float);
        const size_t y_bytes = rows * sizeof(float);
        qw36_gpu_buf *xb = be->upload(ctx, x, x_bytes, QW36_DTYPE_F32);
        qw36_gpu_buf *yb = be->alloc(ctx, y_bytes,   QW36_DTYPE_F32);
        if (!xb || !yb) {
            if (xb) be->free(ctx, xb);
            if (yb) be->free(ctx, yb);
            goto cpu_path;
        }
        be->matmul(ctx, yb, xb, w->gpu_buf, 1, (uint32_t)rows, (uint32_t)cols);
        be->download(ctx, yb, y, y_bytes);
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

int qw36__matmul_lazy_dev(qw36_gpu_buf *y, qw36_gpu_buf *x,
                           const qw36_lazy_w *w)
{
    qw36_gpu_backend *be;
    qw36_gpu_ctx *ctx;
    if (!y || !x || !w || !w->gpu_buf || !w->rows || !w->cols ||
        w->rows > UINT32_MAX || w->cols > UINT32_MAX ||
        !active_backend(&be, &ctx) || !be->matmul)
        return -1;
    be->matmul(ctx, y, x, w->gpu_buf, 1,
               (uint32_t)w->rows, (uint32_t)w->cols);
    return 0;
}

/* Same as qw36__matmul_lazy but operates on a single slice of a 3D stack
 * [n_experts, rows, cols]. Used for MoE expert matmuls without copying. */
int qw36__matmul_lazy_slice(float *y, const float *x,
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
    /* A whole-stack gpu_buf cannot represent an expert slice until task 23
     * adds a real matmul_slice/view contract. Keep MoE slices correct by
     * using the host fp32 slice path for now. */
    view.gpu_buf = NULL;
    return qw36__matmul_lazy(y, x, &view, row_scratch);
}

/* MoE router/top-k implementation lives in qw36_moe.c. */

/* Embedding lookup: write hidden=W.cols floats to out from row `token`. */
int qw36__embed_lookup_lazy(const qw36_lazy_w *w, uint32_t token, float *out) {
    qw36_gpu_backend *be;
    qw36_gpu_ctx *ctx;
    if (w && w->gpu_buf && w->cols <= UINT32_MAX &&
        active_backend(&be, &ctx) && be->embedding_lookup)
    {
        qw36_gpu_buf *yb = be->alloc(ctx, (size_t)w->cols * sizeof(float),
                                     QW36_DTYPE_F32);
        if (yb) {
            be->embedding_lookup(ctx, yb, w->gpu_buf, token, (uint32_t)w->cols);
            be->download(ctx, yb, out, (size_t)w->cols * sizeof(float));
            be->free(ctx, yb);
            return 0;
        }
    }
    return dequant_row(w, token, out);
}

int qw36__embed_lookup_lazy_dev(qw36_gpu_buf *out, const qw36_lazy_w *w,
                                 uint32_t token)
{
    qw36_gpu_backend *be;
    qw36_gpu_ctx *ctx;
    if (!out || !w || !w->gpu_buf || w->cols > UINT32_MAX ||
        !active_backend(&be, &ctx) || !be->embedding_lookup)
        return -1;
    be->embedding_lookup(ctx, out, w->gpu_buf, token, (uint32_t)w->cols);
    return 0;
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

float qw36__silu(float x) { return x / (1.0f + expf(-x)); }

static void swiglu_mlp_f32(float *y, const float *x,
                           const float *w_gate, const float *w_up,
                           const float *w_down,
                           float *scratch_gate, float *scratch_up,
                           size_t hidden, size_t inter)
{
    matmul_f32(scratch_gate, x, w_gate, inter, hidden);
    matmul_f32(scratch_up,   x, w_up,   inter, hidden);
    for (size_t i = 0; i < inter; i++)
        scratch_gate[i] = qw36__silu(scratch_gate[i]) * scratch_up[i];
    matmul_f32(y, scratch_gate, w_down, hidden, inter);
}

/* In-place RoPE on a single head.
 *
 * If sections != NULL and n_sections > 0: applies *multi-axis* RoPE
 * (Qwen3.5 mRoPE). Pair p is in section s if it lies in [sum_{<s} sect,
 * sum_{<=s} sect). Section 0 uses the time position (`pos`); later
 * sections use axis 1/2/3 positions, which are 0 in pure-text decode and
 * therefore leave their pairs unrotated.
 *
 * Otherwise (sections == NULL): plain RoPE over all `rot_dim/2` pairs
 * using `pos`. Pair convention is the half-rotation / NEOX layout
 * (x[i], x[i + d/2]) which matches the Qwen GGUF tensors. */
void qw36__rope_head(float *x, size_t pos, size_t rot_dim, float theta_base,
                      const uint32_t *sections, uint32_t n_sections)
{
    size_t half = rot_dim / 2;
    size_t p = 0;
    while (p < half) {
        size_t axis_pos = pos;
        size_t take = half - p;
        if (n_sections) {
            /* find current section */
            size_t cum = 0; uint32_t s = 0;
            for (; s < n_sections && cum + sections[s] <= p; s++) cum += sections[s];
            if (s >= n_sections) {
                /* past last section ⇒ unrotated */
                break;
            }
            take = (size_t)sections[s] - (p - cum);
            axis_pos = (s == 0) ? pos : 0;
        }
        for (size_t i = 0; i < take; i++) {
            size_t pair_idx = p + i;
            float inv_freq = 1.0f /
                powf(theta_base, (2.0f * (float)pair_idx) / (float)rot_dim);
            float angle = (float)axis_pos * inv_freq;
            float c = cosf(angle), s_ = sinf(angle);
            float x0 = x[pair_idx], x1 = x[pair_idx + half];
            x[pair_idx]        = x0 * c - x1 * s_;
            x[pair_idx + half] = x0 * s_ + x1 * c;
        }
        p += take;
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
        qw36__rope_head(qh, seq_pos, rot_dim, rope_theta, NULL, 0);
    }
    for (uint32_t h = 0; h < n_kv; h++) {
        float *kh = k_row + h * head_dim;
        rmsnorm_f32(kh, kh, k_norm, head_dim, rms_eps);
        qw36__rope_head(kh, seq_pos, rot_dim, rope_theta, NULL, 0);
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

void qw36__rmsnorm_dispatch(float *out, const float *x, const float *w,
                             size_t n, float eps)
{
    qw36_gpu_backend *be;
    qw36_gpu_ctx *ctx;
    if (n <= UINT32_MAX && active_backend(&be, &ctx) && be->rmsnorm) {
        qw36_gpu_buf *xb = be->upload(ctx, x, n * sizeof(float), QW36_DTYPE_F32);
        qw36_gpu_buf *wb = be->upload(ctx, w, n * sizeof(float), QW36_DTYPE_F32);
        qw36_gpu_buf *yb = be->alloc(ctx, n * sizeof(float), QW36_DTYPE_F32);
        if (xb && wb && yb) {
            be->rmsnorm(ctx, yb, xb, wb, (uint32_t)n, eps);
            be->download(ctx, yb, out, n * sizeof(float));
            be->free(ctx, xb);
            be->free(ctx, wb);
            be->free(ctx, yb);
            return;
        }
        if (xb) be->free(ctx, xb);
        if (wb) be->free(ctx, wb);
        if (yb) be->free(ctx, yb);
    }
    rmsnorm_f32(out, x, w, n, eps);
}

int qw36__rmsnorm_dispatch_dev(qw36_gpu_buf *out, qw36_gpu_buf *x,
                                const float *w, size_t n, float eps)
{
    qw36_engine *eng = qw36__active_engine;
    qw36_gpu_backend *be;
    qw36_gpu_ctx *ctx;
    if (!out || !x || !w || n > UINT32_MAX ||
        !active_backend(&be, &ctx) || !be->rmsnorm)
        return -1;
    qw36_gpu_buf *wb = qw36__gpu_cached_upload(eng, w, n * sizeof(float),
                                         QW36_DTYPE_F32);
    if (!wb) return -1;
    be->rmsnorm(ctx, out, x, wb, (uint32_t)n, eps);
    return 0;
}

void qw36__residual_add_dispatch(float *x, const float *y, size_t n)
{
    qw36_gpu_backend *be;
    qw36_gpu_ctx *ctx;
    if (n <= UINT32_MAX && active_backend(&be, &ctx) && be->residual_add) {
        qw36_gpu_buf *xb = be->upload(ctx, x, n * sizeof(float), QW36_DTYPE_F32);
        qw36_gpu_buf *yb = be->upload(ctx, y, n * sizeof(float), QW36_DTYPE_F32);
        if (xb && yb) {
            be->residual_add(ctx, xb, yb, (uint32_t)n);
            be->download(ctx, xb, x, n * sizeof(float));
            be->free(ctx, xb);
            be->free(ctx, yb);
            return;
        }
        if (xb) be->free(ctx, xb);
        if (yb) be->free(ctx, yb);
    }
    for (size_t i = 0; i < n; i++) x[i] += y[i];
}

int qw36__residual_add_dispatch_dev(qw36_gpu_buf *x, qw36_gpu_buf *y,
                                     size_t n)
{
    qw36_gpu_backend *be;
    qw36_gpu_ctx *ctx;
    if (!x || !y || n > UINT32_MAX ||
        !active_backend(&be, &ctx) || !be->residual_add)
        return -1;
    be->residual_add(ctx, x, y, (uint32_t)n);
    return 0;
}

int qw36__swiglu_dispatch(float *y, const float *x,
                           const qw36_lazy_w *w_gate,
                           const qw36_lazy_w *w_up,
                           const qw36_lazy_w *w_down,
                           uint32_t hidden, uint32_t inter,
                           float *scratch_gate, float *scratch_up,
                           float *row_scratch)
{
    qw36_gpu_backend *be;
    qw36_gpu_ctx *ctx;
    if (w_gate && w_up && w_down &&
        w_gate->gpu_buf && w_up->gpu_buf && w_down->gpu_buf &&
        active_backend(&be, &ctx) && be->swiglu_mlp)
    {
        qw36_gpu_buf *xb = be->upload(ctx, x, (size_t)hidden * sizeof(float),
                                      QW36_DTYPE_F32);
        qw36_gpu_buf *yb = be->alloc(ctx, (size_t)hidden * sizeof(float),
                                     QW36_DTYPE_F32);
        if (xb && yb) {
            be->swiglu_mlp(ctx, yb, xb, w_gate->gpu_buf, w_up->gpu_buf,
                           w_down->gpu_buf, hidden, inter);
            be->download(ctx, yb, y, (size_t)hidden * sizeof(float));
            be->free(ctx, xb);
            be->free(ctx, yb);
            return 0;
        }
        if (xb) be->free(ctx, xb);
        if (yb) be->free(ctx, yb);
    }

    if (qw36__matmul_lazy(scratch_gate, x, w_gate, row_scratch)) return -1;
    if (qw36__matmul_lazy(scratch_up,   x, w_up,   row_scratch)) return -1;
    for (uint32_t i = 0; i < inter; i++)
        scratch_gate[i] = qw36__silu(scratch_gate[i]) * scratch_up[i];
    if (qw36__matmul_lazy(y, scratch_gate, w_down, row_scratch)) return -1;
    return 0;
}

int qw36__swiglu_dispatch_dev(qw36_gpu_buf *y, qw36_gpu_buf *x,
                               const qw36_lazy_w *w_gate,
                               const qw36_lazy_w *w_up,
                               const qw36_lazy_w *w_down,
                               uint32_t hidden, uint32_t inter)
{
    qw36_gpu_backend *be;
    qw36_gpu_ctx *ctx;
    if (!y || !x || !w_gate || !w_up || !w_down ||
        !w_gate->gpu_buf || !w_up->gpu_buf || !w_down->gpu_buf ||
        !active_backend(&be, &ctx) || !be->swiglu_mlp)
        return -1;
    be->swiglu_mlp(ctx, y, x, w_gate->gpu_buf, w_up->gpu_buf,
                   w_down->gpu_buf, hidden, inter);
    return 0;
}

int qw36__attention_dispatch(float *y, const float *x,
                              const qw36_layer_weights *L,
                              float *k_cache, float *v_cache,
                              const qw36_config *c,
                              uint32_t seq_pos, uint32_t seq_capacity)
{
    const qw36_lazy_w *wq = (const qw36_lazy_w *)L->q_proj;
    const qw36_lazy_w *wk = (const qw36_lazy_w *)L->k_proj;
    const qw36_lazy_w *wv = (const qw36_lazy_w *)L->v_proj;
    qw36_gpu_backend *be;
    qw36_gpu_ctx *ctx;
    if (!wq || !wk || !wv || !wq->gpu_buf || !wk->gpu_buf || !wv->gpu_buf ||
        !L->q_norm || !L->k_norm ||
        !active_backend(&be, &ctx) || !be->attention)
        return 0;

    const size_t hidden = c->hidden_size;
    const size_t q_len = (size_t)c->num_attention_heads * c->head_dim;
    const size_t kv_dim = (size_t)c->num_key_value_heads * c->head_dim;
    const size_t cache_bytes = (size_t)seq_capacity * kv_dim * sizeof(float);

    qw36_gpu_buf *xb = be->upload(ctx, x, hidden * sizeof(float), QW36_DTYPE_F32);
    qw36_gpu_buf *yb = be->alloc(ctx, q_len * sizeof(float), QW36_DTYPE_F32);
    qw36_gpu_buf *qnb = be->upload(ctx, L->q_norm,
                                   (size_t)c->head_dim * sizeof(float),
                                   QW36_DTYPE_F32);
    qw36_gpu_buf *knb = be->upload(ctx, L->k_norm,
                                   (size_t)c->head_dim * sizeof(float),
                                   QW36_DTYPE_F32);
    qw36_gpu_buf *kb = be->upload(ctx, k_cache, cache_bytes, QW36_DTYPE_F32);
    qw36_gpu_buf *vb = be->upload(ctx, v_cache, cache_bytes, QW36_DTYPE_F32);
    if (xb && yb && qnb && knb && kb && vb) {
        be->attention(ctx, yb, xb, wq->gpu_buf, wk->gpu_buf, wv->gpu_buf,
                      qnb, knb, kb, vb,
                      c->hidden_size, c->num_attention_heads,
                      c->num_key_value_heads, c->head_dim,
                      seq_pos, seq_capacity,
                      c->rope_theta, c->partial_rotary_factor);
        be->download(ctx, yb, y, q_len * sizeof(float));
        be->download(ctx, kb, k_cache, cache_bytes);
        be->download(ctx, vb, v_cache, cache_bytes);
        be->free(ctx, xb);
        be->free(ctx, yb);
        be->free(ctx, qnb);
        be->free(ctx, knb);
        be->free(ctx, kb);
        be->free(ctx, vb);
        return 1;
    }
    if (xb) be->free(ctx, xb);
    if (yb) be->free(ctx, yb);
    if (qnb) be->free(ctx, qnb);
    if (knb) be->free(ctx, knb);
    if (kb) be->free(ctx, kb);
    if (vb) be->free(ctx, vb);
    return 0;
}

int qw36__attention_dispatch_dev(qw36_gpu_buf *y, qw36_gpu_buf *x,
                                  const qw36_layer_weights *L,
                                  qw36_gpu_buf *k_cache,
                                  qw36_gpu_buf *v_cache,
                                  const qw36_config *c,
                                  uint32_t seq_pos,
                                  uint32_t seq_capacity)
{
    const qw36_lazy_w *wq = (const qw36_lazy_w *)L->q_proj;
    const qw36_lazy_w *wk = (const qw36_lazy_w *)L->k_proj;
    const qw36_lazy_w *wv = (const qw36_lazy_w *)L->v_proj;
    qw36_engine *eng = qw36__active_engine;
    qw36_gpu_backend *be;
    qw36_gpu_ctx *ctx;
    if (!y || !x || !k_cache || !v_cache ||
        !wq || !wk || !wv || !wq->gpu_buf || !wk->gpu_buf || !wv->gpu_buf ||
        !L->q_norm || !L->k_norm ||
        !active_backend(&be, &ctx) || !be->attention)
        return -1;

    qw36_gpu_buf *qnb = qw36__gpu_cached_upload(eng, L->q_norm,
        (size_t)c->head_dim * sizeof(float), QW36_DTYPE_F32);
    qw36_gpu_buf *knb = qw36__gpu_cached_upload(eng, L->k_norm,
        (size_t)c->head_dim * sizeof(float), QW36_DTYPE_F32);
    if (!qnb || !knb) return -1;

    be->attention(ctx, y, x, wq->gpu_buf, wk->gpu_buf, wv->gpu_buf,
                  qnb, knb, k_cache, v_cache,
                  c->hidden_size, c->num_attention_heads,
                  c->num_key_value_heads, c->head_dim,
                  seq_pos, seq_capacity,
                  c->rope_theta, c->partial_rotary_factor);
    return 0;
}

int qw36__deltanet_dispatch_dev(qw36_state *st,
                                 const qw36_layer_weights *L,
                                 const qw36_config *c,
                                 uint32_t layer_idx)
{
    qw36_engine *eng = qw36__active_engine;
    qw36_gpu_backend *be;
    qw36_gpu_ctx *ctx;
    if (!st || !L || !c ||
        !st->x_dev || !st->x_rms_dev || !st->dn_qkv_dev ||
        !st->dn_qkv_act_dev || !st->dn_z_dev || !st->dn_alpha_dev ||
        !st->dn_beta_dev || !st->dn_gout_dev ||
        !st->conv_state_dev || !st->delta_state_dev ||
        !st->conv_state_dev[layer_idx] || !st->delta_state_dev[layer_idx] ||
        !L->dn_qkv || !L->dn_gate || !L->dn_alpha || !L->dn_beta ||
        !L->dn_conv1d || !L->dn_dt_bias || !L->dn_a_log ||
        !L->dn_norm || !L->dn_out ||
        !active_backend(&be, &ctx) ||
        !be->matmul || !be->rmsnorm || !be->residual_add ||
        !be->dn_conv1d_silu || !be->dn_gated_delta ||
        !be->dn_gated_rmsnorm)
        return -1;

    static int skip_conv = -1, skip_dn = -1;
    if (skip_conv < 0) {
        const char *e = getenv("QW36_SKIP_CONV1D");
        skip_conv = e && atoi(e) ? 1 : 0;
    }
    if (skip_dn < 0) {
        const char *e = getenv("QW36_SKIP_DN");
        skip_dn = e && atoi(e) ? 1 : 0;
    }
    if (skip_conv || skip_dn) return -1;

    const uint32_t n_v = c->dn_num_value_heads;
    const uint32_t n_k = c->dn_num_key_heads;
    const uint32_t kd = c->dn_key_head_dim;
    const uint32_t vd = c->dn_value_head_dim;
    if (!n_v || !n_k || !kd || !vd) return -1;
    const uint32_t qkv_dim = n_k * kd * 2 + n_v * vd;
    qw36_gpu_buf *conv_w = qw36__gpu_cached_upload(eng, L->dn_conv1d,
        (size_t)qkv_dim * c->dn_conv_kernel_size * sizeof(float),
        QW36_DTYPE_F32);
    qw36_gpu_buf *dt_bias = qw36__gpu_cached_upload(eng, L->dn_dt_bias,
        (size_t)n_v * sizeof(float), QW36_DTYPE_F32);
    qw36_gpu_buf *a_log = qw36__gpu_cached_upload(eng, L->dn_a_log,
        (size_t)n_v * sizeof(float), QW36_DTYPE_F32);
    qw36_gpu_buf *dn_norm = qw36__gpu_cached_upload(eng, L->dn_norm,
        (size_t)vd * sizeof(float), QW36_DTYPE_F32);
    if (!conv_w || !dt_bias || !a_log || !dn_norm) return -1;

    int rc = 0;
    rc |= qw36__rmsnorm_dispatch_dev((qw36_gpu_buf *)st->x_rms_dev,
                               (qw36_gpu_buf *)st->x_dev,
                               (const float *)L->input_layernorm,
                               c->hidden_size, c->rms_norm_eps);
    rc |= qw36__matmul_lazy_dev((qw36_gpu_buf *)st->dn_qkv_dev,
                          (qw36_gpu_buf *)st->x_rms_dev,
                          (const qw36_lazy_w *)L->dn_qkv);
    rc |= qw36__matmul_lazy_dev((qw36_gpu_buf *)st->dn_z_dev,
                          (qw36_gpu_buf *)st->x_rms_dev,
                          (const qw36_lazy_w *)L->dn_gate);
    rc |= qw36__matmul_lazy_dev((qw36_gpu_buf *)st->dn_alpha_dev,
                          (qw36_gpu_buf *)st->x_rms_dev,
                          (const qw36_lazy_w *)L->dn_alpha);
    rc |= qw36__matmul_lazy_dev((qw36_gpu_buf *)st->dn_beta_dev,
                          (qw36_gpu_buf *)st->x_rms_dev,
                          (const qw36_lazy_w *)L->dn_beta);
    if (rc) return -1;

    be->dn_conv1d_silu(ctx, (qw36_gpu_buf *)st->dn_qkv_act_dev,
                       (qw36_gpu_buf *)st->dn_qkv_dev,
                       conv_w,
                       (qw36_gpu_buf *)st->conv_state_dev[layer_idx],
                       qkv_dim, c->dn_conv_kernel_size);
    be->dn_gated_delta(ctx, (qw36_gpu_buf *)st->dn_gout_dev,
                       (qw36_gpu_buf *)st->dn_qkv_act_dev,
                       (qw36_gpu_buf *)st->dn_beta_dev,
                       (qw36_gpu_buf *)st->dn_alpha_dev,
                       dt_bias, a_log,
                       (qw36_gpu_buf *)st->delta_state_dev[layer_idx],
                       n_k, n_v, kd, vd);
    be->dn_gated_rmsnorm(ctx, (qw36_gpu_buf *)st->dn_qkv_dev,
                         (qw36_gpu_buf *)st->dn_gout_dev,
                         (qw36_gpu_buf *)st->dn_z_dev,
                         dn_norm, n_v, vd, c->rms_norm_eps);
    rc |= qw36__matmul_lazy_dev((qw36_gpu_buf *)st->x_rms_dev,
                          (qw36_gpu_buf *)st->dn_qkv_dev,
                          (const qw36_lazy_w *)L->dn_out);
    rc |= qw36__residual_add_dispatch_dev((qw36_gpu_buf *)st->x_dev,
                                    (qw36_gpu_buf *)st->x_rms_dev,
                                    c->hidden_size);
    return rc ? -1 : 0;
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
    float *p = qw36__materialize_f32(t.data, t.dtype, numel);
    if (!p) return NULL;
    return (float *)qw36__eng_own(eng, p);
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
    float *p = qw36__materialize_f32(lw->data, lw->dtype, numel);
    if (!p) return -1;
    if (!qw36__eng_own(eng, p)) { free(p); return -1; }
    lw->data  = p;
    lw->dtype = QW36_DTYPE_F32;
    return 0;
}

static int lazy_materialize_f16(qw36_engine *eng, qw36_lazy_w *lw) {
    if (!lw || lw->dtype == QW36_DTYPE_F16) return 0;
    size_t numel = (size_t)lw->cols * (lw->rows ? lw->rows : 1);
    if (lw->n_extra) numel *= (size_t)lw->n_extra;
    float *tmp = qw36__materialize_f32(lw->data, lw->dtype, numel);
    if (!tmp) return -1;
    uint16_t *p = (uint16_t *)malloc(numel * sizeof(uint16_t));
    if (!p) { free(tmp); return -1; }
    for (size_t i = 0; i < numel; i++) p[i] = qw36__f32_to_f16(tmp[i]);
    free(tmp);
    if (!qw36__eng_own(eng, p)) { free(p); return -1; }
    lw->data  = p;
    lw->dtype = QW36_DTYPE_F16;
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
    return (qw36_lazy_w *)qw36__eng_own(eng, lw);
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
        uint32_t rotary_dim = 0;
        if (eng_get_f32(eng, "rope.partial_rotary_factor", &prf) == 0 && prf > 0) {
            c->partial_rotary_factor = prf;
        } else if (eng_get_u32(eng, "rope.dimension_count", &rotary_dim) == 0 &&
                   rotary_dim > 0 && c->head_dim > 0) {
            c->partial_rotary_factor = (float)rotary_dim / (float)c->head_dim;
        }
    }
    /* Qwen3.5/3.6 mRoPE per-axis pair counts. Stored as
     * <arch>.rope.dimension_sections — for the 0.8B model this is
     * [11, 11, 10, 0] giving 32 pairs over 4 axes. Pairs after the sum
     * (or in section 0 for text decode) use seq_pos; other sections use
     * axis-1/2/3 positions which are 0 in text mode and so leave their
     * pairs unrotated. */
    {
        char key[128];
        snprintf(key, sizeof(key), "%s.rope.dimension_sections", eng->arch);
        int n = qw36_gguf_get_u32_array(eng->gguf, key,
                                        c->rope_sections, 4);
        /* Agent-infer's MLX text-decode path calls fast::rope with
         * rotary_dim from GGUF/config (qwen35.rope.dimension_count = 64
         * for the 0.8B model), traditional=false. We use plain NEOX by
         * default; set QW36_USE_MROPE_SECTIONS=1 to re-enable per-axis
         * chopping for experiments with multimodal positions. */
        const char *mrope_env = getenv("QW36_USE_MROPE_SECTIONS");
        if (mrope_env && atoi(mrope_env) != 0) {
            c->rope_n_sections = (n > 0) ? (uint32_t)n : 0;
        } else {
            c->rope_n_sections = 0;
        }
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
        /* Opt-in fp16 weight materialization on Metal — nearly doubles
         * decode throughput (55 → 81 tok/s) but introduces ~1e-3 drift
         * that can flip the argmax later in a greedy run. Default off so
         * tests/precision_cpu_vs_metal.sh stays bit-equal; set
         * QW36_METAL_FP16_WEIGHTS=1 for the perf path. */
        const char *fp16_env = getenv("QW36_METAL_FP16_WEIGHTS");
        const int fp16_lazy_weights =
            backend->name && strcmp(backend->name, "metal") == 0 &&
            fp16_env && atoi(fp16_env) != 0;

        qw36_lazy_w *lws[] = {
            (qw36_lazy_w *)w->embed_tokens,
            (qw36_lazy_w *)w->lm_head,
            NULL
        };
        for (size_t i = 0; lws[i]; i++) {
            int mrc = fp16_lazy_weights
                ? lazy_materialize_f16(eng, lws[i])
                : lazy_materialize_f32(eng, lws[i]);
            if (mrc) {
                if (err && err_cap) snprintf(err, err_cap,
                    "%s: backend materialize failed (unsupported dtype %d)",
                    backend->name, lws[i]->ggml_type);
                qw36_engine_close(eng); return NULL;
            }
        }
        /* lm_head usually aliases embed_tokens — avoid double-binding. */
        if (w->lm_head == w->embed_tokens) ((qw36_lazy_w *)w->lm_head)->gpu_buf =
            ((qw36_lazy_w *)w->embed_tokens)->gpu_buf;

        #define MAT_AS(field, want_f16_) do { \
            qw36_lazy_w *lw = (qw36_lazy_w *)L->field; \
            int mrc = lw ? ((want_f16_) \
                ? lazy_materialize_f16(eng, lw) \
                : lazy_materialize_f32(eng, lw)) : 0; \
            if (mrc) { \
                if (err && err_cap) snprintf(err, err_cap, \
                    "%s: backend materialize failed at blk.%u." #field, \
                    backend->name, l); \
                qw36_engine_close(eng); return NULL; \
            } \
        } while (0)
        for (uint32_t l = 0; l < c->num_hidden_layers; l++) {
            qw36_layer_weights *L = &eng->weights.layers[l];
            MAT_AS(q_proj, fp16_lazy_weights);
            MAT_AS(k_proj, fp16_lazy_weights);
            MAT_AS(v_proj, fp16_lazy_weights);
            MAT_AS(o_proj, fp16_lazy_weights);
            MAT_AS(dn_qkv, fp16_lazy_weights);
            MAT_AS(dn_gate, fp16_lazy_weights);
            MAT_AS(dn_alpha, 0);
            MAT_AS(dn_beta, 0);
            MAT_AS(dn_out, fp16_lazy_weights);
            MAT_AS(gate_proj, fp16_lazy_weights);
            MAT_AS(up_proj, fp16_lazy_weights);
            MAT_AS(down_proj, fp16_lazy_weights);
            MAT_AS(moe_router, 0);
            MAT_AS(moe_expert_gate, fp16_lazy_weights);
            MAT_AS(moe_expert_up, fp16_lazy_weights);
            MAT_AS(moe_expert_down, fp16_lazy_weights);
            MAT_AS(moe_shared_gate, fp16_lazy_weights);
            MAT_AS(moe_shared_up, fp16_lazy_weights);
            MAT_AS(moe_shared_down, fp16_lazy_weights);
        }
        #undef MAT_AS

        /* Now wrap every materialized buffer as a device-visible view.
         * For Apple Metal this is zero-copy via MTLBuffer NoCopy; for
         * CUDA/HIP this does an actual device copy. */
        #define UPLOAD(field) do { \
            qw36_lazy_w *lw = (qw36_lazy_w *)L->field; \
            if (lw && lw->data && !lw->gpu_buf) { \
                size_t numel = (size_t)lw->cols * (lw->rows ? lw->rows : 1); \
                if (lw->n_extra) numel *= (size_t)lw->n_extra; \
                size_t elem = qw36__dtype_nbytes(lw->dtype); \
                if (!elem) { qw36_engine_close(eng); return NULL; } \
                lw->gpu_buf = backend->upload(eng->ctx, lw->data, \
                    numel * elem, lw->dtype); \
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
            size_t elem = qw36__dtype_nbytes(lw->dtype);
            if (!elem) { qw36_engine_close(eng); return NULL; }
            lw->gpu_buf = backend->upload(eng->ctx, lw->data,
                numel * elem, lw->dtype);
        }
        if (w->lm_head && w->lm_head != w->embed_tokens) {
            qw36_lazy_w *lw = (qw36_lazy_w *)w->lm_head;
            size_t numel = (size_t)lw->cols * lw->rows;
            size_t elem = qw36__dtype_nbytes(lw->dtype);
            if (!elem) { qw36_engine_close(eng); return NULL; }
            lw->gpu_buf = backend->upload(eng->ctx, lw->data,
                numel * elem, lw->dtype);
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
    qw36__gpu_cache_free(eng);
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
    const size_t dn_qkv_dim = c->dn_num_value_heads
        ? (size_t)c->dn_num_key_heads * c->dn_key_head_dim * 2
        + (size_t)c->dn_num_value_heads * c->dn_value_head_dim
        : 0;
    const size_t dn_v_dim = c->dn_num_value_heads
        ? (size_t)c->dn_num_value_heads * c->dn_value_head_dim
        : 0;
    const size_t dn_conv_window = (c->dn_num_value_heads &&
                                   c->dn_conv_kernel_size)
        ? (size_t)c->dn_conv_kernel_size - 1
        : 0;
    const size_t dn_s_elems = c->dn_num_value_heads
        ? (size_t)c->dn_num_value_heads * c->dn_key_head_dim
          * c->dn_value_head_dim
        : 0;

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
        st->conv_state  = (float **)calloc(L, sizeof(float *));
        st->delta_state = (float **)calloc(L, sizeof(float *));
        if (!st->conv_state || !st->delta_state) goto fail;
        for (size_t l = 0; l < L; l++) {
            st->conv_state[l] =
                (float *)calloc(dn_conv_window * dn_qkv_dim, sizeof(float));
            st->delta_state[l] = (float *)calloc(dn_s_elems, sizeof(float));
            if ((dn_conv_window && !st->conv_state[l]) || !st->delta_state[l])
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
    /* row_scratch is used by qw36__matmul_lazy. Size = max(hidden, inter, vocab)
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

    if (eng->backend && eng->ctx && eng->backend->alloc && eng->backend->free) {
        qw36_gpu_backend *be = eng->backend;
        qw36_gpu_ctx *ctx = eng->ctx;
        st->gpu_backend = (void *)be;
        st->gpu_ctx = (void *)ctx;
        st->k_cache_dev = (void **)calloc(L, sizeof(void *));
        st->v_cache_dev = (void **)calloc(L, sizeof(void *));
        if (!st->k_cache_dev || !st->v_cache_dev) goto fail;
        if (c->dn_num_value_heads) {
            st->conv_state_dev = (void **)calloc(L, sizeof(void *));
            st->delta_state_dev = (void **)calloc(L, sizeof(void *));
            if (!st->conv_state_dev || !st->delta_state_dev) goto fail;
        }

        const char *fp16_weights_env = getenv("QW36_METAL_FP16_WEIGHTS");
        const char *fp16_kv_env = getenv("QW36_METAL_FP16_KV");
        const int use_fp16_dev_kv =
            be->name && strcmp(be->name, "metal") == 0 &&
            fp16_weights_env && atoi(fp16_weights_env) != 0 &&
            (!fp16_kv_env || atoi(fp16_kv_env) != 0);
        const qw36_dtype dev_kv_dtype =
            use_fp16_dev_kv ? QW36_DTYPE_F16 : QW36_DTYPE_F32;
        const size_t dev_kv_elem_bytes =
            use_fp16_dev_kv ? sizeof(uint16_t) : sizeof(float);
        const size_t cache_bytes =
            (size_t)seq_capacity * kv_dim * dev_kv_elem_bytes;
        for (size_t l = 0; l < L; l++) {
            st->k_cache_dev[l] = be->alloc(ctx, cache_bytes, dev_kv_dtype);
            st->v_cache_dev[l] = be->alloc(ctx, cache_bytes, dev_kv_dtype);
            if (!st->k_cache_dev[l] || !st->v_cache_dev[l]) goto fail;
            if (c->dn_num_value_heads) {
                st->conv_state_dev[l] = be->alloc(ctx,
                    dn_conv_window * dn_qkv_dim * sizeof(float), QW36_DTYPE_F32);
                st->delta_state_dev[l] = be->alloc(ctx,
                    dn_s_elems * sizeof(float), QW36_DTYPE_F32);
                if (!st->conv_state_dev[l] || !st->delta_state_dev[l])
                    goto fail;
                if (be->copy_from_host) {
                    be->copy_from_host(ctx, (qw36_gpu_buf *)st->conv_state_dev[l],
                        st->conv_state[l],
                        dn_conv_window * dn_qkv_dim * sizeof(float));
                    be->copy_from_host(ctx, (qw36_gpu_buf *)st->delta_state_dev[l],
                        st->delta_state[l], dn_s_elems * sizeof(float));
                }
            }
        }

        st->x_dev = be->alloc(ctx, hidden * sizeof(float), QW36_DTYPE_F32);
        st->x_rms_dev = be->alloc(ctx, hidden * sizeof(float), QW36_DTYPE_F32);
        st->q_dev = be->alloc(ctx, q_dim * sizeof(float), QW36_DTYPE_F32);
        st->k_dev = be->alloc(ctx, kv_dim * sizeof(float), QW36_DTYPE_F32);
        st->v_dev = be->alloc(ctx, kv_dim * sizeof(float), QW36_DTYPE_F32);
        st->attn_scores_dev = be->alloc(ctx,
            (attn_scratch_n + rs_n) * sizeof(float), QW36_DTYPE_F32);
        st->gate_dev = be->alloc(ctx, inter * sizeof(float), QW36_DTYPE_F32);
        st->up_dev = be->alloc(ctx, inter * sizeof(float), QW36_DTYPE_F32);
        st->logits_dev = be->alloc(ctx, vocab * sizeof(float), QW36_DTYPE_F32);
        if (c->dn_num_value_heads) {
            st->dn_qkv_dev = be->alloc(ctx, dn_qkv_dim * sizeof(float),
                                       QW36_DTYPE_F32);
            st->dn_qkv_act_dev = be->alloc(ctx, dn_qkv_dim * sizeof(float),
                                           QW36_DTYPE_F32);
            st->dn_z_dev = be->alloc(ctx, dn_v_dim * sizeof(float),
                                     QW36_DTYPE_F32);
            st->dn_alpha_dev = be->alloc(ctx,
                (size_t)c->dn_num_value_heads * sizeof(float), QW36_DTYPE_F32);
            st->dn_beta_dev = be->alloc(ctx,
                (size_t)c->dn_num_value_heads * sizeof(float), QW36_DTYPE_F32);
            st->dn_gout_dev = be->alloc(ctx, dn_v_dim * sizeof(float),
                                        QW36_DTYPE_F32);
        }
        if (!st->x_dev || !st->x_rms_dev || !st->q_dev ||
            !st->k_dev || !st->v_dev || !st->attn_scores_dev ||
            !st->gate_dev || !st->up_dev || !st->logits_dev)
            goto fail;
        if (c->dn_num_value_heads &&
            (!st->dn_qkv_dev || !st->dn_qkv_act_dev || !st->dn_z_dev ||
             !st->dn_alpha_dev || !st->dn_beta_dev || !st->dn_gout_dev))
            goto fail;
    }

    return st;

fail:
    qw36_state_free(st);
    return NULL;
}

void qw36_state_free(qw36_state *st)
{
    if (!st) return;
    qw36_gpu_backend *be = NULL;
    qw36_gpu_ctx *ctx = NULL;
    (void)qw36__state_backend(st, &be, &ctx);
    if (be && ctx && be->free) {
        if (st->k_cache_dev) {
            for (uint32_t l = 0; l < st->num_layers; l++)
                be->free(ctx, (qw36_gpu_buf *)st->k_cache_dev[l]);
        }
        if (st->v_cache_dev) {
            for (uint32_t l = 0; l < st->num_layers; l++)
                be->free(ctx, (qw36_gpu_buf *)st->v_cache_dev[l]);
        }
        if (st->conv_state_dev) {
            for (uint32_t l = 0; l < st->num_layers; l++)
                be->free(ctx, (qw36_gpu_buf *)st->conv_state_dev[l]);
        }
        if (st->delta_state_dev) {
            for (uint32_t l = 0; l < st->num_layers; l++)
                be->free(ctx, (qw36_gpu_buf *)st->delta_state_dev[l]);
        }
        be->free(ctx, (qw36_gpu_buf *)st->x_dev);
        be->free(ctx, (qw36_gpu_buf *)st->x_rms_dev);
        be->free(ctx, (qw36_gpu_buf *)st->q_dev);
        be->free(ctx, (qw36_gpu_buf *)st->k_dev);
        be->free(ctx, (qw36_gpu_buf *)st->v_dev);
        be->free(ctx, (qw36_gpu_buf *)st->attn_scores_dev);
        be->free(ctx, (qw36_gpu_buf *)st->gate_dev);
        be->free(ctx, (qw36_gpu_buf *)st->up_dev);
        be->free(ctx, (qw36_gpu_buf *)st->logits_dev);
        be->free(ctx, (qw36_gpu_buf *)st->dn_qkv_dev);
        be->free(ctx, (qw36_gpu_buf *)st->dn_qkv_act_dev);
        be->free(ctx, (qw36_gpu_buf *)st->dn_z_dev);
        be->free(ctx, (qw36_gpu_buf *)st->dn_alpha_dev);
        be->free(ctx, (qw36_gpu_buf *)st->dn_beta_dev);
        be->free(ctx, (qw36_gpu_buf *)st->dn_gout_dev);
    }
    free(st->k_cache_dev);
    free(st->v_cache_dev);
    free(st->conv_state_dev);
    free(st->delta_state_dev);
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
    const int skip_logits = qw36__skip_logits_this_forward;
    qw36__skip_logits_this_forward = 0;
    if (!eng || !st) return -1;
    const qw36_config  *c = &eng->cfg;
    const qw36_weights *w = &eng->weights;
    if (token >= c->vocab_size) return -1;
    if (st->seq_pos >= st->seq_capacity) return -1;

    qw36_engine *prev_active = qw36__active_engine;
    qw36__active_engine = eng;
    int forward_batch_active = 0;
    if (eng->backend && eng->backend->begin_batch && eng->ctx) {
        eng->backend->begin_batch(eng->ctx);
        forward_batch_active = 1;
    }
#define QW36_FORWARD_END_BATCH() do { \
        if (forward_batch_active && eng->backend && eng->backend->end_batch && eng->ctx) { \
            eng->backend->end_batch(eng->ctx); \
            forward_batch_active = 0; \
        } \
    } while (0)
#define QW36_FORWARD_RETURN(code_) do { \
        QW36_FORWARD_END_BATCH(); \
        qw36__active_engine = prev_active; \
        return (code_); \
    } while (0)

    const size_t hidden = c->hidden_size;
    const size_t inter = c->intermediate_size;
    const size_t attn_scratch_n =
        (size_t)c->num_attention_heads * ((size_t)st->seq_capacity + 1)
      + (size_t)c->num_attention_heads * c->head_dim;
    float *row_scratch = st->attn_scores + attn_scratch_n;

    qw36_gpu_backend *state_be = NULL;
    qw36_gpu_ctx *state_ctx = NULL;
    int gpu_state = qw36__state_backend(st, &state_be, &state_ctx) &&
                    state_be == eng->backend && state_ctx == eng->ctx &&
                    st->x_dev && st->x_rms_dev && st->q_dev && st->logits_dev;
    int x_dev_valid = 0;
    int x_host_valid = 0;

    qw36_forward_ctx fc = {
        .eng = eng,
        .st = st,
        .cfg = c,
        .weights = w,
        .hidden = hidden,
        .inter = inter,
        .row_scratch = row_scratch,
        .gpu_state = gpu_state,
        .x_dev_valid = &x_dev_valid,
        .x_host_valid = &x_host_valid,
        .debug_layer = 0,
    };

    if (gpu_state &&
        qw36__embed_lookup_lazy_dev((qw36_gpu_buf *)st->x_dev,
                                    (const qw36_lazy_w *)w->embed_tokens,
                                    token) == 0) {
        x_dev_valid = 1;
    } else {
        if (qw36__embed_lookup_lazy((const qw36_lazy_w *)w->embed_tokens,
                                    token, st->x))
            QW36_FORWARD_RETURN(-3);
        x_host_valid = 1;
        if (gpu_state && qw36__ensure_x_dev(&fc)) QW36_FORWARD_RETURN(-8);
    }

    static int debug_layer = -1;
    if (debug_layer < 0) {
        const char *e = getenv("QW36_DEBUG_LAYER");
        debug_layer = e && atoi(e) ? 1 : 0;
    }
    fc.debug_layer = debug_layer;
    if (debug_layer) {
        if (qw36__ensure_x_host(&fc)) QW36_FORWARD_RETURN(-8);
        double ss = 0.0;
        for (size_t i = 0; i < hidden; i++) ss += (double)st->x[i] * st->x[i];
        fprintf(stderr, "[layer 0 in]  ||x||=%.4f x[0..3]=%.3f %.3f %.3f\n",
                sqrt(ss), st->x[0], st->x[1], st->x[2]);
    }

    static int bypass_layers = -1;
    static int max_layers = -1;
    if (bypass_layers < 0) {
        const char *e = getenv("QW36_BYPASS_LAYERS");
        bypass_layers = e && atoi(e) ? 1 : 0;
    }
    if (max_layers < 0) {
        const char *e = getenv("QW36_MAX_LAYERS");
        max_layers = e ? atoi(e) : (int)c->num_hidden_layers;
        if (max_layers < 0) max_layers = (int)c->num_hidden_layers;
    }

    for (uint32_t l = 0; l < c->num_hidden_layers; l++) {
        if (bypass_layers) break;
        if ((int)l >= max_layers) break;
        const qw36_layer_weights *L = &w->layers[l];

        int rc = 0;
        if (!L->q_proj && L->dn_qkv) {
            rc = qw36__attn_deltanet_forward(&fc, L, l);
        } else if (L->q_proj) {
            rc = qw36__attn_vanilla_forward(&fc, L, l);
        } else {
            QW36_FORWARD_RETURN(-2);
        }
        if (rc) QW36_FORWARD_RETURN(rc);

        if (debug_layer) {
            if (qw36__ensure_x_host(&fc)) QW36_FORWARD_RETURN(-8);
            double ss = 0.0;
            for (size_t i = 0; i < hidden; i++) ss += (double)st->x[i] * st->x[i];
            fprintf(stderr, "[L%u post-attn] ||x||=%.4f\n", l, sqrt(ss));
        }

        rc = qw36__mlp_forward(&fc, L, l);
        if (rc) QW36_FORWARD_RETURN(rc);

        if (debug_layer) {
            if (qw36__ensure_x_host(&fc)) QW36_FORWARD_RETURN(-8);
            double ss = 0.0;
            for (size_t i = 0; i < hidden; i++) ss += (double)st->x[i] * st->x[i];
            fprintf(stderr, "[L%u out]  ||x||=%.4f\n", l, sqrt(ss));
        }
    }

    if (skip_logits) {
        st->seq_pos++;
        QW36_FORWARD_RETURN(0);
    }

    if (gpu_state && eng->backend && eng->backend->rmsnorm && eng->backend->matmul) {
        if (qw36__ensure_x_dev(&fc)) QW36_FORWARD_RETURN(-8);
        int grc = 0;
        grc |= qw36__rmsnorm_dispatch_dev((qw36_gpu_buf *)st->x_rms_dev,
                                          (qw36_gpu_buf *)st->x_dev,
                                          (const float *)w->final_norm,
                                          hidden, c->rms_norm_eps);
        grc |= qw36__matmul_lazy_dev((qw36_gpu_buf *)st->logits_dev,
                                     (qw36_gpu_buf *)st->x_rms_dev,
                                     (const qw36_lazy_w *)w->lm_head);
        if (grc == 0) {
            QW36_FORWARD_END_BATCH();
            if (qw36__state_download_to_host(st, st->logits_dev, st->logits,
                                             (size_t)c->vocab_size * sizeof(float)))
                QW36_FORWARD_RETURN(-8);
            st->seq_pos++;
            QW36_FORWARD_RETURN(0);
        }
    }

    if (qw36__ensure_x_host(&fc)) QW36_FORWARD_RETURN(-8);
    qw36__rmsnorm_dispatch(st->x_rms, st->x, (const float *)w->final_norm,
                           hidden, c->rms_norm_eps);
    qw36__matmul_lazy(st->logits, st->x_rms,
                      (const qw36_lazy_w *)w->lm_head, row_scratch);

    st->seq_pos++;
    QW36_FORWARD_RETURN(0);
#undef QW36_FORWARD_END_BATCH
#undef QW36_FORWARD_RETURN
}

int qw36_prefill(qw36_engine *eng, qw36_state *st,
                 const uint32_t *tokens, size_t length)
{
    for (size_t i = 0; i < length; i++) {
        qw36__skip_logits_this_forward = (i + 1 < length);
        int rc = qw36_forward(eng, st, tokens[i]);
        qw36__skip_logits_this_forward = 0;
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
