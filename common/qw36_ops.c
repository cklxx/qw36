/* qw36_ops.c - shared scalar/GPU forward primitives.
 *
 * This file owns the package-private glue that keeps host and device copies
 * of the residual stream coherent while the forward pass is split across
 * attention, DeltaNet, and MLP modules. It deliberately does not allocate
 * model state; callers pass a qw36_forward_ctx built by qw36_forward.
 *
 * Return convention: 0 means the requested copy or dispatch completed, and
 * negative values are forwarded by qw36_forward as engine errors.
 */

#include "qw36_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int qw36__ensure_x_host(qw36_forward_ctx *fc)
{
    if (!fc || !fc->st || !fc->x_host_valid || !fc->x_dev_valid)
        return -8;
    if (*fc->x_host_valid || !*fc->x_dev_valid) return 0;
    const size_t n = fc->hidden;
    if (fc->st->dev_x_dtype == QW36_DTYPE_F16) {
        /* Download fp16 dev → f32 host with on-the-fly conversion. */
        uint16_t *tmp = (uint16_t *)malloc(n * sizeof(uint16_t));
        if (!tmp) return -8;
        if (qw36__state_download_to_host(fc->st, fc->st->x_dev, tmp,
                                         n * sizeof(uint16_t))) {
            free(tmp);
            return -8;
        }
        float *xh = fc->st->x;
        for (size_t i = 0; i < n; i++) xh[i] = qw36__f16_to_f32(tmp[i]);
        free(tmp);
    } else {
        if (qw36__state_download_to_host(fc->st, fc->st->x_dev, fc->st->x,
                                         n * sizeof(float)))
            return -8;
    }
    *fc->x_host_valid = 1;
    return 0;
}

int qw36__ensure_x_dev(qw36_forward_ctx *fc)
{
    if (!fc || !fc->st || !fc->x_host_valid || !fc->x_dev_valid)
        return -8;
    if (*fc->x_dev_valid || !*fc->x_host_valid || !fc->gpu_state) return 0;
    const size_t n = fc->hidden;
    if (fc->st->dev_x_dtype == QW36_DTYPE_F16) {
        /* Convert f32 host → fp16 then upload. */
        uint16_t *tmp = (uint16_t *)malloc(n * sizeof(uint16_t));
        if (!tmp) return -8;
        const float *xh = fc->st->x;
        for (size_t i = 0; i < n; i++) tmp[i] = qw36__f32_to_f16(xh[i]);
        if (qw36__state_copy_from_host(fc->st, fc->st->x_dev, tmp,
                                       n * sizeof(uint16_t))) {
            free(tmp);
            return -8;
        }
        free(tmp);
    } else {
        if (qw36__state_copy_from_host(fc->st, fc->st->x_dev, fc->st->x,
                                       n * sizeof(float)))
            return -8;
    }
    *fc->x_dev_valid = 1;
    return 0;
}

/* --------------------------------------------------------------------- */
/* Backend lookup + lazy matmul / scalar reference ops.                  */
/* --------------------------------------------------------------------- */

int qw36__active_backend(qw36_gpu_backend **be_out, qw36_gpu_ctx **ctx_out)
{
    qw36_engine *eng = qw36__active_engine;
    if (!eng || !eng->backend || !eng->ctx) return 0;
    qw36_gpu_backend *be = eng->backend;
    if (!be->upload || !be->download || !be->alloc || !be->free) return 0;
    if (be_out) *be_out = be;
    if (ctx_out) *ctx_out = eng->ctx;
    return 1;
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
        if (qw36__dequant_row(w, r, row_scratch)) return -1;
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
        !qw36__active_backend(&be, &ctx) || !be->matmul)
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
        if (qw36__dtype_block_geom(w->dtype, &qk, &bpb) || cols % qk != 0) return -1;
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
        qw36__active_backend(&be, &ctx) && be->embedding_lookup)
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
    return qw36__dequant_row(w, token, out);
}

int qw36__embed_lookup_lazy_dev(qw36_gpu_buf *out, const qw36_lazy_w *w,
                                 uint32_t token)
{
    qw36_gpu_backend *be;
    qw36_gpu_ctx *ctx;
    if (!out || !w || !w->gpu_buf || w->cols > UINT32_MAX ||
        !qw36__active_backend(&be, &ctx) || !be->embedding_lookup)
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
    if (n <= UINT32_MAX && qw36__active_backend(&be, &ctx) && be->rmsnorm) {
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
        !qw36__active_backend(&be, &ctx) || !be->rmsnorm)
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
    if (n <= UINT32_MAX && qw36__active_backend(&be, &ctx) && be->residual_add) {
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
        !qw36__active_backend(&be, &ctx) || !be->residual_add)
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
    if (w_gate && w_down &&
        w_gate->gpu_buf && (!w_up || w_up->gpu_buf) && w_down->gpu_buf &&
        qw36__active_backend(&be, &ctx) && be->swiglu_mlp)
    {
        qw36_gpu_buf *xb = be->upload(ctx, x, (size_t)hidden * sizeof(float),
                                      QW36_DTYPE_F32);
        qw36_gpu_buf *yb = be->alloc(ctx, (size_t)hidden * sizeof(float),
                                     QW36_DTYPE_F32);
        if (xb && yb) {
            be->swiglu_mlp(ctx, yb, xb, w_gate->gpu_buf,
                           w_up ? w_up->gpu_buf : NULL,
                           w_down->gpu_buf, hidden, inter);
            be->download(ctx, yb, y, (size_t)hidden * sizeof(float));
            be->free(ctx, xb);
            be->free(ctx, yb);
            return 0;
        }
        if (xb) be->free(ctx, xb);
        if (yb) be->free(ctx, yb);
    }

    qw36_lazy_w gate_view, up_view;
    const qw36_lazy_w *gate_w = w_gate;
    const qw36_lazy_w *up_w = w_up;
    if (!up_w) {
        if (!w_gate || w_gate->cols != hidden || w_gate->rows != (uint64_t)inter * 2 ||
            (w_gate->dtype != QW36_DTYPE_F32 && w_gate->dtype != QW36_DTYPE_F16))
            return -1;
        const size_t elem = qw36__dtype_nbytes(w_gate->dtype);
        const size_t half_bytes = (size_t)inter * (size_t)hidden * elem;
        gate_view = *w_gate;
        gate_view.rows = inter;
        gate_view.gpu_buf = NULL;
        up_view = *w_gate;
        up_view.data = (const uint8_t *)w_gate->data + half_bytes;
        up_view.rows = inter;
        up_view.gpu_buf = NULL;
        gate_w = &gate_view;
        up_w = &up_view;
    }

    if (qw36__matmul_lazy(scratch_gate, x, gate_w, row_scratch)) return -1;
    if (qw36__matmul_lazy(scratch_up,   x, up_w,   row_scratch)) return -1;
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
    if (!y || !x || !w_gate || !w_down ||
        !w_gate->gpu_buf || (w_up && !w_up->gpu_buf) || !w_down->gpu_buf ||
        !qw36__active_backend(&be, &ctx) || !be->swiglu_mlp)
        return -1;
    if (!w_up && w_gate->rows != (uint64_t)inter * 2) return -1;
    be->swiglu_mlp(ctx, y, x, w_gate->gpu_buf,
                   w_up ? w_up->gpu_buf : NULL,
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
        !qw36__active_backend(&be, &ctx) || !be->attention)
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
        !qw36__active_backend(&be, &ctx) || !be->attention)
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
