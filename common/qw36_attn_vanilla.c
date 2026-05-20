/* qw36_attn_vanilla.c - vanilla GQA decode for Qwen-family layers.
 *
 * This module owns the full-attention branch of a single-token forward step:
 * input RMSNorm, Q/K/V projection, per-head q/k RMSNorm, RoPE, KV cache
 * append, causal GQA score/softmax/combine, output projection, and residual
 * add.
 *
 * Backends may run the path through their fused attention vtable. The CPU
 * fallback remains the reference implementation used for backend parity and
 * debugging.
 */

#include "qw36_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

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

    if (fc->gpu_state && eng->backend && eng->backend->rmsnorm &&
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
        grc |= qw36__residual_add_dispatch_dev((qw36_gpu_buf *)st->x_dev,
                                               (qw36_gpu_buf *)st->x_rms_dev,
                                               hidden);
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

    int used_backend_attention = qw36__attention_dispatch(staging, st->x_rms, L,
        (float *)st->k_cache[layer_idx], (float *)st->v_cache[layer_idx],
        c, st->seq_pos, st->seq_capacity);

    if (!used_backend_attention) {
        int v_rc = 0;
        v_rc |= qw36__matmul_lazy(st->q, st->x_rms,
                                  (const qw36_lazy_w *)L->q_proj,
                                  fc->row_scratch);

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
        for (uint32_t h = 0; h < c->num_key_value_heads; h++) {
            float *kh = k_row + (size_t)h * c->head_dim;
            qw36__rope_head(kh, st->seq_pos, rot_dim, c->rope_theta,
                            c->rope_n_sections ? c->rope_sections : NULL,
                            c->rope_n_sections);
        }

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

    if (qw36__matmul_lazy(st->x_rms, staging,
                          (const qw36_lazy_w *)L->o_proj,
                          fc->row_scratch))
        return -6;
    qw36__residual_add_dispatch(x, st->x_rms, hidden);

    *fc->x_host_valid = 1;
    *fc->x_dev_valid = 0;
    return 0;
}
