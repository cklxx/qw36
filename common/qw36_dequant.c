/* qw36_dequant.c - GGUF dtype conversion and block dequantization.
 *
 * This module owns dense fp16/bf16 conversion, GGML block dequantizers,
 * whole-tensor materialization to fp32, and row-at-a-time lazy dequant for
 * matrix/vector kernels. All entry points are package-private and are used
 * by engine binding plus common ops; public API signatures remain unchanged.
 */

#include "qw36_internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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

float qw36__f16_to_f32(uint16_t h) { return f16_to_f32(h); }

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

/* Total byte count for an n-element tensor in `dtype`. Handles dense and
 * block-quantised dtypes uniformly: dense returns numel*sizeof(elem); quant
 * returns (numel / qk) * bytes_per_block. */
size_t qw36__tensor_bytes(qw36_dtype dtype, size_t numel) {
    size_t elem = qw36__dtype_nbytes(dtype);
    if (elem) return numel * elem;
    switch (dtype) {
        case QW36_DTYPE_Q2_K: return (numel / 256u) *  84u;
        case QW36_DTYPE_Q3_K: return (numel / 256u) * 110u;
        case QW36_DTYPE_Q4_K: return (numel / 256u) * 144u;
        case QW36_DTYPE_Q4K_AFFINE32: return (numel / 256u) * 160u;
        case QW36_DTYPE_Q5_K: return (numel / 256u) * 176u;
        case QW36_DTYPE_Q5K_AFFINE32: return (numel / 256u) * 192u;
        case QW36_DTYPE_Q6_K: return (numel / 256u) * 210u;
        case QW36_DTYPE_Q6K_SCALE16: return (numel / 256u) * 224u;
        case QW36_DTYPE_Q8_0: return (numel /  32u) *  34u;
        default: return 0;
    }
}

int qw36__dtype_is_native_gpu_quant(qw36_dtype dt) {
    /* Quantised dtypes for which the Metal backend has a native on-device
     * matmul kernel. CUDA/AMD don't yet, so callers gate on backend name. */
    return dt == QW36_DTYPE_Q4_K || dt == QW36_DTYPE_Q5_K ||
           dt == QW36_DTYPE_Q6_K || dt == QW36_DTYPE_Q8_0 ||
           dt == QW36_DTYPE_Q4K_AFFINE32 ||
           dt == QW36_DTYPE_Q5K_AFFINE32 ||
           dt == QW36_DTYPE_Q6K_SCALE16;
}

int qw36__lazy_w_shape(const qw36_lazy_w *w, size_t *rows_out,
                       size_t *cols_out, size_t *numel_out)
{
    if (!w || !w->cols) return -1;
    uint64_t rows64 = w->rows ? w->rows : 1u;
    if (w->n_extra) {
        if (rows64 > UINT64_MAX / w->n_extra) return -1;
        rows64 *= w->n_extra;
    }
    if (rows64 > (uint64_t)SIZE_MAX || w->cols > (uint64_t)SIZE_MAX)
        return -1;
    size_t rows = (size_t)rows64;
    size_t cols = (size_t)w->cols;
    if (rows && cols > SIZE_MAX / rows) return -1;
    if (rows_out) *rows_out = rows;
    if (cols_out) *cols_out = cols;
    if (numel_out) *numel_out = rows * cols;
    return 0;
}

/* --------------------------------------------------------------------- */
/* GGML-style quantized block dequantizers.                               */
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

/* Q4K_AFFINE32: internal runtime layout produced from GGUF Q4_K.
 * Per 256-element block:
 *   fp16 scale[8]      (16B)  scale = d * sc[sub]
 *   fp16 bias[8]       (16B)  bias  = -dmin * mn[sub]
 *   u8 q[8][16]       (128B)  each 32-element sub-block packed low/high
 * Total: 160 bytes. */
static void dq_q4k_affine32(const uint8_t *blocks, float *out, size_t n) {
    const size_t nb = n / QW36_QK_K;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t *b = blocks + i * 160;
        const uint16_t *scales = (const uint16_t *)b;
        const uint16_t *biases = (const uint16_t *)(b + 16);
        const uint8_t *qs = b + 32;
        for (int sub = 0; sub < 8; sub++) {
            const float scale = f16_to_f32(scales[sub]);
            const float bias = f16_to_f32(biases[sub]);
            const uint8_t *qg = qs + sub * 16;
            for (int j = 0; j < 16; j++) {
                uint8_t byte = qg[j];
                *out++ = scale * (float)(byte & 0x0F) + bias;
                *out++ = scale * (float)(byte >> 4) + bias;
            }
        }
    }
}

static uint8_t q4_K_nibble(const uint8_t *block, int sub, int j) {
    const uint8_t *qs = block + 16;
    int iter = sub >> 1;
    int hi = sub & 1;
    uint8_t byte = qs[iter * 32 + j];
    return hi ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0x0F);
}

static void pack_5bit8(uint8_t *dst, const uint8_t q[8]) {
    uint64_t bits = 0;
    for (int i = 0; i < 8; i++)
        bits |= ((uint64_t)(q[i] & 31u)) << (5 * i);
    for (int i = 0; i < 5; i++)
        dst[i] = (uint8_t)((bits >> (8 * i)) & 0xFFu);
}

static uint8_t unpack_5bit8(const uint8_t *src, int idx) {
    uint64_t bits = 0;
    for (int i = 0; i < 5; i++)
        bits |= ((uint64_t)src[i]) << (8 * i);
    return (uint8_t)((bits >> (5 * idx)) & 31u);
}

static void pack_6bit16(uint8_t *dst, const uint8_t q[16]) {
    memset(dst, 0, 12);
    for (int i = 0; i < 16; i++) {
        uint16_t v = (uint16_t)(q[i] & 63u);
        int bit = 6 * i;
        int byte = bit >> 3;
        int shift = bit & 7;
        uint16_t x = (uint16_t)(v << shift);
        dst[byte] |= (uint8_t)(x & 0xFFu);
        if (byte + 1 < 12)
            dst[byte + 1] |= (uint8_t)(x >> 8);
    }
}

static uint8_t unpack_6bit16(const uint8_t *src, int idx) {
    int bit = 6 * idx;
    int byte = bit >> 3;
    int shift = bit & 7;
    uint16_t x = (uint16_t)src[byte];
    if (byte + 1 < 12)
        x |= (uint16_t)src[byte + 1] << 8;
    return (uint8_t)((x >> shift) & 63u);
}

int qw36__repack_q4k_affine32(const qw36_lazy_w *w, void *dst_void) {
    if (!w || !dst_void || w->dtype != QW36_DTYPE_Q4_K ||
        !w->cols || (w->cols % QW36_QK_K) != 0)
        return -1;
    const size_t rows = (size_t)w->rows * (w->n_extra ? (size_t)w->n_extra : 1u);
    const size_t blocks_per_row = (size_t)w->cols / QW36_QK_K;
    const uint8_t *src = (const uint8_t *)w->data;
    uint8_t *dst = (uint8_t *)dst_void;
    const size_t src_row_bytes = blocks_per_row * 144u;
    const size_t dst_row_bytes = blocks_per_row * 160u;

    for (size_t r = 0; r < rows; r++) {
        const uint8_t *src_row = src + r * src_row_bytes;
        uint8_t *dst_row = dst + r * dst_row_bytes;
        for (size_t bi = 0; bi < blocks_per_row; bi++) {
            const uint8_t *b = src_row + bi * 144u;
            uint8_t *o = dst_row + bi * 160u;
            uint16_t dh, dmh;
            memcpy(&dh,  b,     2);
            memcpy(&dmh, b + 2, 2);
            const float d = f16_to_f32(dh);
            const float dmin = f16_to_f32(dmh);
            const uint8_t *packed_scales = b + 4;
            uint16_t *out_scales = (uint16_t *)o;
            uint16_t *out_biases = (uint16_t *)(o + 16);
            uint8_t *out_q = o + 32;

            for (int sub = 0; sub < 8; sub++) {
                uint8_t sc, mn;
                q4_K_get_scale_min(sub, packed_scales, &sc, &mn);
                out_scales[sub] = qw36__f32_to_f16(d * (float)sc);
                out_biases[sub] = qw36__f32_to_f16(-dmin * (float)mn);
                uint8_t *qg = out_q + sub * 16;
                for (int j = 0; j < 16; j++) {
                    uint8_t q0 = q4_K_nibble(b, sub, j * 2);
                    uint8_t q1 = q4_K_nibble(b, sub, j * 2 + 1);
                    qg[j] = (uint8_t)(q0 | (q1 << 4));
                }
            }
        }
    }
    return 0;
}

/* Q5K_AFFINE32: internal runtime layout produced from GGUF Q5_K.
 * Per 256-element block:
 *   fp16 scale[8]      (16B)  scale = d * sc[sub]
 *   fp16 bias[8]       (16B)  bias  = -dmin * mn[sub]
 *   u8 q[8][20]       (160B)  each 32-element sub-block packed 5-bit
 * Total: 192 bytes. */
static void dq_q5k_affine32(const uint8_t *blocks, float *out, size_t n) {
    const size_t nb = n / QW36_QK_K;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t *b = blocks + i * 192;
        const uint16_t *scales = (const uint16_t *)b;
        const uint16_t *biases = (const uint16_t *)(b + 16);
        const uint8_t *qs = b + 32;
        for (int sub = 0; sub < 8; sub++) {
            const float scale = f16_to_f32(scales[sub]);
            const float bias = f16_to_f32(biases[sub]);
            const uint8_t *qg = qs + sub * 20;
            for (int p = 0; p < 4; p++) {
                const uint8_t *pack = qg + p * 5;
                for (int j = 0; j < 8; j++)
                    *out++ = scale * (float)unpack_5bit8(pack, j) + bias;
            }
        }
    }
}

static uint8_t q5_K_quant(const uint8_t *block, int sub, int j) {
    const uint8_t *qh = block + 16;
    const uint8_t *ql = block + 48;
    int iter = sub >> 1;
    int hi_nibble = sub & 1;
    uint8_t byte = ql[iter * 32 + j];
    uint8_t lo = hi_nibble ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0x0F);
    uint8_t hi = (uint8_t)((qh[j] >> sub) & 1u);
    return (uint8_t)(lo | (hi << 4));
}

int qw36__repack_q5k_affine32(const qw36_lazy_w *w, void *dst_void) {
    if (!w || !dst_void || w->dtype != QW36_DTYPE_Q5_K ||
        !w->cols || (w->cols % QW36_QK_K) != 0)
        return -1;
    const size_t rows = (size_t)w->rows * (w->n_extra ? (size_t)w->n_extra : 1u);
    const size_t blocks_per_row = (size_t)w->cols / QW36_QK_K;
    const uint8_t *src = (const uint8_t *)w->data;
    uint8_t *dst = (uint8_t *)dst_void;
    const size_t src_row_bytes = blocks_per_row * 176u;
    const size_t dst_row_bytes = blocks_per_row * 192u;

    for (size_t r = 0; r < rows; r++) {
        const uint8_t *src_row = src + r * src_row_bytes;
        uint8_t *dst_row = dst + r * dst_row_bytes;
        for (size_t bi = 0; bi < blocks_per_row; bi++) {
            const uint8_t *b = src_row + bi * 176u;
            uint8_t *o = dst_row + bi * 192u;
            uint16_t dh, dmh;
            memcpy(&dh,  b,     2);
            memcpy(&dmh, b + 2, 2);
            const float d = f16_to_f32(dh);
            const float dmin = f16_to_f32(dmh);
            const uint8_t *packed_scales = b + 4;
            uint16_t *out_scales = (uint16_t *)o;
            uint16_t *out_biases = (uint16_t *)(o + 16);
            uint8_t *out_q = o + 32;

            for (int sub = 0; sub < 8; sub++) {
                uint8_t sc, mn;
                q4_K_get_scale_min(sub, packed_scales, &sc, &mn);
                out_scales[sub] = qw36__f32_to_f16(d * (float)sc);
                out_biases[sub] = qw36__f32_to_f16(-dmin * (float)mn);
                uint8_t *qg = out_q + sub * 20;
                for (int p = 0; p < 4; p++) {
                    uint8_t q[8];
                    for (int j = 0; j < 8; j++)
                        q[j] = q5_K_quant(b, sub, p * 8 + j);
                    pack_5bit8(qg + p * 5, q);
                }
            }
        }
    }
    return 0;
}

/* Q6K_SCALE16: internal runtime layout produced from GGUF Q6_K.
 * Per 256-element block:
 *   fp16 scale[16]    (32B)  scale = d * int8_sc[group]
 *   u8 q[16][12]     (192B)  each 16-element scale group packed 6-bit
 * Total: 224 bytes. */
static void dq_q6k_scale16(const uint8_t *blocks, float *out, size_t n) {
    const size_t nb = n / QW36_QK_K;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t *b = blocks + i * 224;
        const uint16_t *scales = (const uint16_t *)b;
        const uint8_t *qs = b + 32;
        for (int group = 0; group < 16; group++) {
            const float scale = f16_to_f32(scales[group]);
            const uint8_t *qg = qs + group * 12;
            for (int j = 0; j < 16; j++)
                *out++ = scale * ((float)unpack_6bit16(qg, j) - 32.0f);
        }
    }
}

static uint8_t q6_K_quant(const uint8_t *block, int local) {
    const uint8_t *ql_all = block;
    const uint8_t *qh_all = block + 128;
    int half = local >> 7;
    int rem = local & 127;
    int lane = rem & 31;
    const uint8_t *ql = ql_all + half * 64;
    const uint8_t *qh = qh_all + half * 32;
    if (rem < 32)
        return (uint8_t)((ql[lane] & 0x0F) | ((qh[lane] & 0x03) << 4));
    if (rem < 64)
        return (uint8_t)((ql[lane + 32] & 0x0F) | (((qh[lane] >> 2) & 0x03) << 4));
    if (rem < 96)
        return (uint8_t)((ql[lane] >> 4) | (((qh[lane] >> 4) & 0x03) << 4));
    return (uint8_t)((ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 0x03) << 4));
}

int qw36__repack_q6k_scale16(const qw36_lazy_w *w, void *dst_void) {
    if (!w || !dst_void || w->dtype != QW36_DTYPE_Q6_K ||
        !w->cols || (w->cols % QW36_QK_K) != 0)
        return -1;
    const size_t rows = (size_t)w->rows * (w->n_extra ? (size_t)w->n_extra : 1u);
    const size_t blocks_per_row = (size_t)w->cols / QW36_QK_K;
    const uint8_t *src = (const uint8_t *)w->data;
    uint8_t *dst = (uint8_t *)dst_void;
    const size_t src_row_bytes = blocks_per_row * 210u;
    const size_t dst_row_bytes = blocks_per_row * 224u;

    for (size_t r = 0; r < rows; r++) {
        const uint8_t *src_row = src + r * src_row_bytes;
        uint8_t *dst_row = dst + r * dst_row_bytes;
        for (size_t bi = 0; bi < blocks_per_row; bi++) {
            const uint8_t *b = src_row + bi * 210u;
            uint8_t *o = dst_row + bi * 224u;
            uint16_t dh;
            memcpy(&dh, b + 208, 2);
            const float d = f16_to_f32(dh);
            const int8_t *src_scales = (const int8_t *)(b + 192);
            uint16_t *out_scales = (uint16_t *)o;
            uint8_t *out_q = o + 32;

            for (int group = 0; group < 16; group++) {
                out_scales[group] = qw36__f32_to_f16(d * (float)src_scales[group]);
                uint8_t q[16];
                for (int j = 0; j < 16; j++)
                    q[j] = q6_K_quant(b, group * 16 + j);
                pack_6bit16(out_q + group * 12, q);
            }
        }
    }
    return 0;
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
        case QW36_DTYPE_Q4K_AFFINE32:
            dq_q4k_affine32((const uint8_t *)data, out, n);
            return out;
        case QW36_DTYPE_Q5_K:
            dq_q5_K((const uint8_t *)data, out, n);
            return out;
        case QW36_DTYPE_Q5K_AFFINE32:
            dq_q5k_affine32((const uint8_t *)data, out, n);
            return out;
        case QW36_DTYPE_Q6_K:
            dq_q6_K((const uint8_t *)data, out, n);
            return out;
        case QW36_DTYPE_Q6K_SCALE16:
            dq_q6k_scale16((const uint8_t *)data, out, n);
            return out;
        default:
            free(out);
            return NULL;
    }
}

/* --------------------------------------------------------------------- */
/* Lazy block-quantized helpers: dequantize one row at a time.            */
/* --------------------------------------------------------------------- */

int qw36__dtype_block_geom(qw36_dtype dt, size_t *qk, size_t *bytes_per_block) {
    switch (dt) {
        case QW36_DTYPE_Q2_K: *qk = 256; *bytes_per_block =  84; return 0;
        case QW36_DTYPE_Q3_K: *qk = 256; *bytes_per_block = 110; return 0;
        case QW36_DTYPE_Q4_K: *qk = 256; *bytes_per_block = 144; return 0;
        case QW36_DTYPE_Q4K_AFFINE32: *qk = 256; *bytes_per_block = 160; return 0;
        case QW36_DTYPE_Q5_K: *qk = 256; *bytes_per_block = 176; return 0;
        case QW36_DTYPE_Q5K_AFFINE32: *qk = 256; *bytes_per_block = 192; return 0;
        case QW36_DTYPE_Q6_K: *qk = 256; *bytes_per_block = 210; return 0;
        case QW36_DTYPE_Q6K_SCALE16: *qk = 256; *bytes_per_block = 224; return 0;
        case QW36_DTYPE_Q8_0: *qk = 32;  *bytes_per_block = 34;  return 0;
        default: *qk = 0; *bytes_per_block = 0; return -1;
    }
}

/* Decode `cols` elements of row `row_idx` from a block-quantized stack
 * where each row is laid out as cols-elements-worth of blocks. */
int qw36__dequant_row(const qw36_lazy_w *w, size_t row_idx, float *out) {
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
    if (qw36__dtype_block_geom(w->dtype, &qk, &bpb) || cols % qk != 0)
        return -1;
    const size_t blocks_per_row = cols / qk;
    const uint8_t *row = (const uint8_t *)w->data
                       + row_idx * blocks_per_row * bpb;
    switch (w->dtype) {
        case QW36_DTYPE_Q2_K: dq_q2_K(row, out, cols); return 0;
        case QW36_DTYPE_Q3_K: dq_q3_K(row, out, cols); return 0;
        case QW36_DTYPE_Q4_K: dq_q4_K(row, out, cols); return 0;
        case QW36_DTYPE_Q4K_AFFINE32: dq_q4k_affine32(row, out, cols); return 0;
        case QW36_DTYPE_Q5_K: dq_q5_K(row, out, cols); return 0;
        case QW36_DTYPE_Q5K_AFFINE32: dq_q5k_affine32(row, out, cols); return 0;
        case QW36_DTYPE_Q6_K: dq_q6_K(row, out, cols); return 0;
        case QW36_DTYPE_Q6K_SCALE16: dq_q6k_scale16(row, out, cols); return 0;
        case QW36_DTYPE_Q8_0: dq_q8_0(row, out, cols); return 0;
        default: return -1;
    }
}

uint16_t *qw36__materialize_f16_rows(const qw36_lazy_w *w)
{
    size_t rows = 0, cols = 0, numel = 0;
    if (qw36__lazy_w_shape(w, &rows, &cols, &numel)) return NULL;
    if (numel > SIZE_MAX / sizeof(uint16_t) ||
        cols > SIZE_MAX / sizeof(float))
        return NULL;
    uint16_t *p = (uint16_t *)malloc(numel * sizeof(uint16_t));
    float *row = (float *)malloc(cols * sizeof(float));
    if (!p || !row) {
        free(p);
        free(row);
        return NULL;
    }
    for (size_t r = 0; r < rows; r++) {
        if (qw36__dequant_row(w, r, row)) {
            free(p);
            free(row);
            return NULL;
        }
        uint16_t *dst = p + r * cols;
        for (size_t c = 0; c < cols; c++)
            dst[c] = qw36__f32_to_f16(row[c]);
    }
    free(row);
    return p;
}
