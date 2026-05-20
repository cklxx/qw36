/* qw36_attn_vanilla.c - vanilla GQA decode for Qwen-family layers.
 *
 * This module owns the full-attention branch of a single-token forward step:
 * input RMSNorm, Q/K/V projection, per-head q/k RMSNorm, RoPE, KV cache
 * append, causal GQA score/softmax/combine, optional Q-gate application,
 * output projection, and residual add.
 *
 * Backends may run the no-gate path through their fused attention vtable.
 * Qwen3.5/3.6 Q-gate layers intentionally fall back to the CPU reference
 * path until the backend attention kernels grow an explicit gate input.
 */

#include "qw36_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float sigmoid_f32(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

typedef struct {
    int init;
    int layer;
    int pos;
    int take;
    int finished;
    int wrote_any;
    uint32_t active_layer;
    uint32_t active_pos;
    const char *out_path;
    FILE *fp;
} qw36_trace_state;

static qw36_trace_state *trace_begin(uint32_t layer_idx, uint32_t seq_pos)
{
    static qw36_trace_state ts;
    if (!ts.init) {
        const char *l = getenv("QW36_TRACE_LAYER");
        const char *p = getenv("QW36_TRACE_POS");
        const char *t = getenv("QW36_TRACE_TAKE");
        ts.layer = l ? atoi(l) : -1;
        ts.pos = p ? atoi(p) : -1;
        ts.take = t ? atoi(t) : 32;
        if (ts.take <= 0) ts.take = 32;
        ts.out_path = getenv("QW36_TRACE_OUT");
        ts.init = 1;
    }
    if (ts.finished || ts.fp || ts.layer != (int)layer_idx)
        return NULL;
    if (ts.pos >= 0 && ts.pos != (int)seq_pos)
        return NULL;
    if (!ts.out_path || !ts.out_path[0])
        return NULL;

    ts.fp = fopen(ts.out_path, "w");
    if (!ts.fp) {
        fprintf(stderr, "qw36: failed to open layer trace output: %s\n",
                ts.out_path);
        ts.finished = 1;
        return NULL;
    }
    ts.active_layer = layer_idx;
    ts.active_pos = seq_pos;
    ts.wrote_any = 0;
    fprintf(ts.fp,
            "{\n"
            "  \"layer\": %u,\n"
            "  \"seq_pos\": %u,\n"
            "  \"captured\": {\n",
            layer_idx, seq_pos);
    return &ts;
}

static void trace_tensor(qw36_trace_state *ts, const char *name,
                         const float *x, size_t n,
                         const size_t *shape, size_t ndim)
{
    if (!ts || !ts->fp || !name || !x) return;
    size_t take = n < (size_t)ts->take ? n : (size_t)ts->take;
    if (ts->wrote_any) fprintf(ts->fp, ",\n");
    fprintf(ts->fp, "    \"%s\": {\"shape\": [", name);
    for (size_t i = 0; i < ndim; i++) {
        if (i) fprintf(ts->fp, ", ");
        fprintf(ts->fp, "%zu", shape[i]);
    }
    fprintf(ts->fp, "], \"first_n\": [");
    for (size_t i = 0; i < take; i++) {
        if (i) fprintf(ts->fp, ", ");
        fprintf(ts->fp, "%.9g", x[i]);
    }
    fprintf(ts->fp, "]}");
    ts->wrote_any = 1;
}

static void trace_finish(qw36_trace_state *ts)
{
    if (!ts || !ts->fp) return;
    fprintf(ts->fp, "\n  }\n}\n");
    fclose(ts->fp);
    ts->fp = NULL;
    ts->finished = 1;
    fprintf(stderr, "qw36: wrote layer trace L%u pos %u to %s\n",
            ts->active_layer, ts->active_pos, ts->out_path);
}

int qw36__attn_vanilla_forward(qw36_forward_ctx *fc,
                               const qw36_layer_weights *L,
                               uint32_t layer_idx)
{
    if (!fc || !L || !fc->st || !fc->cfg || !fc->eng) return -1;
    if (!L->q_proj) return -2;

    qw36_engine *eng = fc->eng;
    qw36_state *st = fc->st;
    const qw36_config *c = fc->cfg;
    const size_t hidden = fc->hidden;
    float *x = st->x;

    /* GPU fast-path. Allowed when there is no Q-gate, OR when there is a
     * Q-gate but all of q/k/v are fp16 (the fused decode kernel handles
     * the gate; the slower prep+score+combine fallback does not). */
    const qw36_lazy_w *wq_lw = (const qw36_lazy_w *)L->q_proj;
    const qw36_lazy_w *wk_lw = (const qw36_lazy_w *)L->k_proj;
    const qw36_lazy_w *wv_lw = (const qw36_lazy_w *)L->v_proj;
    /* The fused decode kernel reads q/k/v as fp32; the matmul writing them
     * may be either MPS (when weights are F16) or the native quantised
     * dequant+gemv kernel (when weights stay Q4_K/Q5_K/Q6_K/Q8_0 on GPU).
     * Either path produces fp32 q/k/v, so allow the fused fast-path in both
     * cases — the slower prep+score+combine fallback still cannot consume
     * the gated q layout, so non-fused matmul backends fall back to CPU. */
    #define QW36_FUSED_DTYPE_OK(d_) ((d_) == QW36_DTYPE_F16 || \
        (d_) == QW36_DTYPE_Q4_K || (d_) == QW36_DTYPE_Q5_K || \
        (d_) == QW36_DTYPE_Q6_K || (d_) == QW36_DTYPE_Q8_0)
    const int dtype_ok = wq_lw && wk_lw && wv_lw &&
        QW36_FUSED_DTYPE_OK(wq_lw->dtype) &&
        QW36_FUSED_DTYPE_OK(wk_lw->dtype) &&
        QW36_FUSED_DTYPE_OK(wv_lw->dtype);
    #undef QW36_FUSED_DTYPE_OK
    const int qgate_gpu_ok = !c->has_q_gate || dtype_ok;
    if (qgate_gpu_ok &&
        fc->gpu_state && eng->backend && eng->backend->rmsnorm &&
        eng->backend->matmul && eng->backend->attention &&
        eng->backend->residual_add &&
        L->q_proj && L->k_proj && L->v_proj && L->o_proj &&
        L->q_norm && L->k_norm &&
        st->k_cache_dev && st->v_cache_dev &&
        st->k_cache_dev[layer_idx] && st->v_cache_dev[layer_idx])
    {
        int erc = qw36__ensure_x_dev(fc);
        if (erc) return erc;
        int grc = 0;
        grc |= qw36__rmsnorm_dispatch_dev((qw36_gpu_buf *)st->x_rms_dev,
                                          (qw36_gpu_buf *)st->x_dev,
                                          (const float *)L->input_layernorm,
                                          hidden, c->rms_norm_eps);
        grc |= qw36__attention_dispatch_dev((qw36_gpu_buf *)st->q_dev,
                                            (qw36_gpu_buf *)st->x_rms_dev, L,
                                            (qw36_gpu_buf *)st->k_cache_dev[layer_idx],
                                            (qw36_gpu_buf *)st->v_cache_dev[layer_idx],
                                            c, st->seq_pos, st->seq_capacity);
        grc |= qw36__matmul_lazy_dev((qw36_gpu_buf *)st->x_rms_dev,
                                     (qw36_gpu_buf *)st->q_dev,
                                     (const qw36_lazy_w *)L->o_proj);
        /* If the backend has the fused residual+rmsnorm kernel, do it
         * here and mark the post-attn rmsnorm done so qw36__mlp_forward
         * skips its own rmsnorm dispatch. Saves one dispatch per layer. */
        const int try_fused = eng->backend && eng->backend->residual_rmsnorm &&
                              L->post_attn_layernorm;
        if (try_fused) {
            int rrc = qw36__residual_rmsnorm_dispatch_dev(
                (qw36_gpu_buf *)st->x_dev,
                (qw36_gpu_buf *)st->x_rms_dev,
                (qw36_gpu_buf *)st->x_rms_dev,
                (const float *)L->post_attn_layernorm,
                hidden, c->rms_norm_eps);
            if (rrc == 0) {
                fc->post_attn_rmsnorm_done = 1;
            } else {
                grc |= qw36__residual_add_dispatch_dev((qw36_gpu_buf *)st->x_dev,
                                                       (qw36_gpu_buf *)st->x_rms_dev,
                                                       hidden);
            }
        } else {
            grc |= qw36__residual_add_dispatch_dev((qw36_gpu_buf *)st->x_dev,
                                                   (qw36_gpu_buf *)st->x_rms_dev,
                                                   hidden);
        }
        if (grc == 0) {
            *fc->x_dev_valid = 1;
            *fc->x_host_valid = 0;
            return 0;
        }
        return -7;
    }

    int erc = qw36__ensure_x_host(fc);
    if (erc) return erc;

    qw36__rmsnorm_dispatch(st->x_rms, x, (const float *)L->input_layernorm,
                           hidden, c->rms_norm_eps);
    if (fc->debug_layer) {
        double ss = 0.0;
        for (size_t i = 0; i < hidden; i++) ss += (double)st->x_rms[i] * st->x_rms[i];
        fprintf(stderr, "[L%u attn x_rms ||=%.4f] norm_w[0..2]=%.4f %.4f %.4f\n",
                layer_idx, sqrt(ss),
                ((const float *)L->input_layernorm)[0],
                ((const float *)L->input_layernorm)[1],
                ((const float *)L->input_layernorm)[2]);
    }

    const uint32_t q_dim = c->num_attention_heads * c->head_dim;
    const uint32_t kv_dim = c->num_key_value_heads * c->head_dim;
    const uint32_t rot_dim = (uint32_t)((float)c->head_dim *
                                        c->partial_rotary_factor);
    float *staging = st->attn_scores
                   + (size_t)c->num_attention_heads * (st->seq_capacity + 1);
    qw36_trace_state *trace = trace_begin(layer_idx, st->seq_pos);
    size_t shape_hidden[3] = {1, 1, hidden};
    size_t shape_q_raw[3] = {1, 1, c->has_q_gate ? (size_t)q_dim * 2 : q_dim};
    size_t shape_q_split[4] = {1, 1, c->num_attention_heads, c->head_dim};
    size_t shape_q_heads[4] = {1, c->num_attention_heads, 1, c->head_dim};
    size_t shape_kv_raw[3] = {1, 1, kv_dim};
    size_t shape_kv_heads[4] = {1, c->num_key_value_heads, 1, c->head_dim};
    size_t shape_flat_q[3] = {1, 1, q_dim};
    trace_tensor(trace, "input_layernorm_out", st->x_rms, hidden,
                 shape_hidden, 3);

    /* The host attention_dispatch routes through metal_attention which
     * handles Q-gate via its fused fp16 path; only enable when the same
     * fp16 condition is met so the prep+score+combine fallback isn't
     * taken (it cannot consume the gated q layout). */
    int used_backend_attention = 0;
    if (qgate_gpu_ok) {
        used_backend_attention = qw36__attention_dispatch(staging, st->x_rms, L,
            (float *)st->k_cache[layer_idx], (float *)st->v_cache[layer_idx],
            c, st->seq_pos, st->seq_capacity);
    }

    if (!used_backend_attention) {
        int v_rc = 0;
        if (c->has_q_gate) {
            if (!st->q_full || !st->q_gate) return -3;
            v_rc |= qw36__matmul_lazy(st->q_full, st->x_rms,
                                      (const qw36_lazy_w *)L->q_proj,
                                      fc->row_scratch);
            trace_tensor(trace, "q_proj_raw", st->q_full, (size_t)q_dim * 2,
                         shape_q_raw, 3);
            for (uint32_t h = 0; h < c->num_attention_heads; h++) {
                const float *src = st->q_full + (size_t)h * 2 * c->head_dim;
                float *qd = st->q + (size_t)h * c->head_dim;
                float *gd = st->q_gate + (size_t)h * c->head_dim;
                memcpy(qd, src, c->head_dim * sizeof(float));
                memcpy(gd, src + c->head_dim, c->head_dim * sizeof(float));
            }
            trace_tensor(trace, "q_split_pre_norm", st->q, q_dim,
                         shape_q_split, 4);
            trace_tensor(trace, "gate_split", st->q_gate, q_dim,
                         shape_q_split, 4);
        } else {
            v_rc |= qw36__matmul_lazy(st->q, st->x_rms,
                                      (const qw36_lazy_w *)L->q_proj,
                                      fc->row_scratch);
            trace_tensor(trace, "q_proj_raw", st->q, q_dim, shape_q_raw, 3);
        }

        float *k_row = (float *)st->k_cache[layer_idx]
                     + (size_t)st->seq_pos * kv_dim;
        float *v_row = (float *)st->v_cache[layer_idx]
                     + (size_t)st->seq_pos * kv_dim;
        v_rc |= qw36__matmul_lazy(k_row, st->x_rms,
                                  (const qw36_lazy_w *)L->k_proj,
                                  fc->row_scratch);
        v_rc |= qw36__matmul_lazy(v_row, st->x_rms,
                                  (const qw36_lazy_w *)L->v_proj,
                                  fc->row_scratch);
        trace_tensor(trace, "k_proj_raw", k_row, kv_dim, shape_kv_raw, 3);
        trace_tensor(trace, "v_proj_raw", v_row, kv_dim, shape_kv_raw, 3);
        if (v_rc) {
            fprintf(stderr, "qw36: vanilla QKV failed at layer %u "
                    "(ggml types %d/%d/%d)\n", layer_idx,
                    ((const qw36_lazy_w *)L->q_proj)->ggml_type,
                    ((const qw36_lazy_w *)L->k_proj)->ggml_type,
                    ((const qw36_lazy_w *)L->v_proj)->ggml_type);
            return -6;
        }

        for (uint32_t h = 0; h < c->num_attention_heads; h++) {
            float *qh = st->q + (size_t)h * c->head_dim;
            qw36__rmsnorm_dispatch(qh, qh, (const float *)L->q_norm,
                                   c->head_dim, c->rms_norm_eps);
        }
        trace_tensor(trace, "q_post_norm", st->q, q_dim, shape_q_heads, 4);
        for (uint32_t h = 0; h < c->num_attention_heads; h++) {
            float *qh = st->q + (size_t)h * c->head_dim;
            qw36__rope_head(qh, st->seq_pos, rot_dim, c->rope_theta,
                            c->rope_n_sections ? c->rope_sections : NULL,
                            c->rope_n_sections);
        }
        for (uint32_t h = 0; h < c->num_key_value_heads; h++) {
            float *kh = k_row + (size_t)h * c->head_dim;
            qw36__rmsnorm_dispatch(kh, kh, (const float *)L->k_norm,
                                   c->head_dim, c->rms_norm_eps);
        }
        trace_tensor(trace, "k_post_norm", k_row, kv_dim, shape_kv_heads, 4);
        for (uint32_t h = 0; h < c->num_key_value_heads; h++) {
            float *kh = k_row + (size_t)h * c->head_dim;
            qw36__rope_head(kh, st->seq_pos, rot_dim, c->rope_theta,
                            c->rope_n_sections ? c->rope_sections : NULL,
                            c->rope_n_sections);
        }
        trace_tensor(trace, "q_post_rope", st->q, q_dim, shape_q_heads, 4);
        trace_tensor(trace, "k_post_rope", k_row, kv_dim, shape_kv_heads, 4);

        const float inv_sqrt_d = 1.0f / sqrtf((float)c->head_dim);
        for (uint32_t h = 0; h < c->num_attention_heads; h++) {
            const uint32_t kvh = h * c->num_key_value_heads
                               / c->num_attention_heads;
            const float *qh = st->q + (size_t)h * c->head_dim;
            float *scores = st->attn_scores
                          + (size_t)h * (st->seq_capacity + 1);
            float maxv = -INFINITY;
            for (uint32_t t = 0; t <= st->seq_pos; t++) {
                const float *kh = (float *)st->k_cache[layer_idx]
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
            float *head_out = staging + (size_t)h * c->head_dim;
            for (uint32_t d = 0; d < c->head_dim; d++) {
                double acc = 0.0;
                for (uint32_t t = 0; t <= st->seq_pos; t++) {
                    const float *vh = (float *)st->v_cache[layer_idx]
                                    + (size_t)t * kv_dim + kvh * c->head_dim;
                    acc += (double)(scores[t] * inv_sum) * vh[d];
                }
                head_out[d] = (float)acc;
            }
        }
    }

    trace_tensor(trace, "attn_out_pre_gate", staging, q_dim,
                 shape_q_heads, 4);
    if (c->has_q_gate) {
        for (uint32_t i = 0; i < q_dim; i++)
            staging[i] *= sigmoid_f32(st->q_gate[i]);
        trace_tensor(trace, "attn_out_gated", staging, q_dim,
                     shape_flat_q, 3);
    }

    if (qw36__matmul_lazy(st->x_rms, staging,
                          (const qw36_lazy_w *)L->o_proj,
                          fc->row_scratch))
        return -6;
    trace_tensor(trace, "o_proj_out", st->x_rms, hidden, shape_hidden, 3);
    trace_finish(trace);
    qw36__residual_add_dispatch(x, st->x_rms, hidden);

    *fc->x_host_valid = 1;
    *fc->x_dev_valid = 0;
    return 0;
}
