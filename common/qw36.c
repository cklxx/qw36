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
 * because a 35B Q4_K_XL model decompresses to ~140GB fp32. */
typedef struct {
    const void *data;
    qw36_dtype  dtype;
    uint32_t    ggml_type;   /* informational */
    uint64_t    rows;        /* GGUF dim[1] (outer) */
    uint64_t    cols;        /* GGUF dim[0] (inner) */
    uint64_t    n_extra;     /* dim[2] (used for MoE expert stacks) */
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
        case QW36_DTYPE_Q4_K:
            dq_q4_K((const uint8_t *)data, out, n);
            return out;
        case QW36_DTYPE_Q6_K:
            dq_q6_K((const uint8_t *)data, out, n);
            return out;
        default:
            free(out);
            return NULL;
    }
}

/* --------------------------------------------------------------------- */
/* Lazy block-quantized helpers: dequantize one row at a time.            */
/* --------------------------------------------------------------------- */

static int dtype_block_geom(qw36_dtype dt, size_t *qk, size_t *bytes_per_block) {
    switch (dt) {
        case QW36_DTYPE_Q4_K: *qk = 256; *bytes_per_block = 144; return 0;
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
        case QW36_DTYPE_Q4_K: dq_q4_K(row, out, cols); return 0;
        case QW36_DTYPE_Q6_K: dq_q6_K(row, out, cols); return 0;
        case QW36_DTYPE_Q8_0: dq_q8_0(row, out, cols); return 0;
        default: return -1;
    }
}

/* matmul against a lazy quantized weight: y[r] = sum_c W[r,c] * x[c]. */
static int matmul_lazy(float *y, const float *x, const qw36_lazy_w *w,
                       float *row_scratch)
{
    const size_t rows = (size_t)w->rows;
    const size_t cols = (size_t)w->cols;
    for (size_t r = 0; r < rows; r++) {
        if (dequant_row(w, r, row_scratch)) return -1;
        double acc = 0.0;
        for (size_t c = 0; c < cols; c++) acc += (double)row_scratch[c] * x[c];
        y[r] = (float)acc;
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
        const uint32_t kvh = h % n_kv;
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
        const uint32_t kvh = h % n_kv;
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

    /* Bind global tensors. Small (norm) → fp32; big (embed/lm_head) → lazy. */
    qw36_weights *w = &eng->weights;
    w->dtype = QW36_DTYPE_F32;
    w->embed_tokens = bind_tensor_lazy(eng, "token_embd.weight");
    w->final_norm   = bind_tensor_f32(eng, "output_norm.weight");
    w->lm_head      = bind_tensor_lazy(eng, "output.weight");
    if (!w->lm_head && c->tie_word_embeddings) w->lm_head = w->embed_tokens;
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
        BIND_NORM(post_attn_layernorm, "blk.%u.ffn_norm.weight");

        /* Vanilla attention projections — lazy (big). */
        BIND_W(q_proj,    "blk.%u.attn_q.weight");
        BIND_W(k_proj,    "blk.%u.attn_k.weight");
        BIND_W(v_proj,    "blk.%u.attn_v.weight");
        BIND_W(o_proj,    "blk.%u.attn_output.weight");

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

    /* TODO(Claude): if backend is non-NULL, upload weights and switch the
     * forward path to use the backend vtable. For v0 we always run CPU. */
    (void)backend;
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

    for (uint32_t l = 0; l < c->num_hidden_layers; l++) {
        const qw36_layer_weights *L = &w->layers[l];
        float *x = st->x;
        if (!L->q_proj) {
            /* Gated DeltaNet (Qwen3.5/3.6 linear attention) — TODO. */
            return -2;
        }

        /* --- attention block --- */
        rmsnorm_f32(st->x_rms, x, (const float *)L->input_layernorm,
                    hidden, c->rms_norm_eps);

        const uint32_t kv_dim = c->num_key_value_heads * c->head_dim;
        const uint32_t rot_dim = (uint32_t)((float)c->head_dim *
                                            c->partial_rotary_factor);

        /* QKV projections via lazy matmul. */
        matmul_lazy(st->q, st->x_rms, (const qw36_lazy_w *)L->q_proj, row_scratch);
        float *k_row = (float *)st->k_cache[l] + (size_t)st->seq_pos * kv_dim;
        float *v_row = (float *)st->v_cache[l] + (size_t)st->seq_pos * kv_dim;
        matmul_lazy(k_row, st->x_rms, (const qw36_lazy_w *)L->k_proj, row_scratch);
        matmul_lazy(v_row, st->x_rms, (const qw36_lazy_w *)L->v_proj, row_scratch);

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
         * of kv_head (h mod n_kv), softmax, weighted sum of v_cache. */
        const float inv_sqrt_d = 1.0f / sqrtf((float)c->head_dim);
        float *staging = st->attn_scores
                       + (size_t)c->num_attention_heads * (st->seq_capacity + 1);
        for (uint32_t h = 0; h < c->num_attention_heads; h++) {
            const uint32_t kvh = h % c->num_key_value_heads;
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

        /* --- mlp block --- */
        rmsnorm_f32(st->x_rms, x, (const float *)L->post_attn_layernorm,
                    hidden, c->rms_norm_eps);

        if (L->moe_router) {
            /* MoE — TODO. Stub: skip MLP contribution. */
            (void)inter;
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
    }

    /* Final norm + lm_head (lazy). */
    rmsnorm_f32(st->x_rms, st->x, (const float *)w->final_norm,
                hidden, c->rms_norm_eps);
    matmul_lazy(st->logits, st->x_rms,
                (const qw36_lazy_w *)w->lm_head, row_scratch);

    st->seq_pos++;
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
