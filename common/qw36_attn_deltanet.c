/* qw36_attn_deltanet.c - Qwen3.5/3.6 Gated DeltaNet decode path.
 *
 * This module implements the per-token DeltaNet branch used by hybrid Qwen
 * checkpoints. It owns the host fallback for qkv/gate/alpha/beta projection,
 * depthwise causal conv1d + SiLU, gated delta-rule state update, gated
 * RMSNorm, output projection, and residual add.
 *
 * The GPU fast path is dispatched first when a backend advertises the fused
 * DeltaNet kernels. If that path is unavailable, the function falls back to
 * the scalar reference implementation and updates qw36_forward_ctx validity
 * bits so the main scheduler knows whether x is resident on host or device.
 */

#include "qw36_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
void qw36__gated_delta_decode(const float *qkv,
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

    /* Qwen3.5 / 3.6 GGUF attn_qkv follows the reference split
     * [Q all heads | K all heads | V all heads]. An earlier bisection
     * treated it as [Qh,Kh,Vh] interleaved, which made the first vanilla
     * layer receive a corrupted residual and produced multilingual noise.
     * Keep the correct block layout by default; set QW36_QKV_INTERLEAVE=1
     * only for experimental checkpoints that really store per-head blocks. */
    static int qkv_interleave = -1;
    if (qkv_interleave < 0) {
        const char *e = getenv("QW36_QKV_INTERLEAVE");
        qkv_interleave = (e && atoi(e) != 0) ? 1 : 0;
    }

    for (uint32_t v = 0; v < n_value; v++) {
        uint32_t group = (n_key && n_value % n_key == 0) ? n_value / n_key : 1;
        if (group == 0) group = 1;
        const uint32_t kh = (n_value >= n_key && n_value % n_key == 0)
            ? v / group
            : v % n_key;
        const uint32_t head_stride = 2 * key_dim + val_dim;  /* per-head: q,k,v */

        /* Copy and L2-normalize q, k for this head. */
        double qss = 0.0, kss = 0.0;
        for (uint32_t d = 0; d < key_dim; d++) {
            if (qkv_interleave) {
                qbuf[d] = qkv[kh * head_stride + d];
                kbuf[d] = qkv[kh * head_stride + key_dim + d];
            } else {
                qbuf[d] = qkv[kh * key_dim + d];
                kbuf[d] = qkv[q_dim_total + kh * key_dim + d];
            }
            qss += (double)qbuf[d] * qbuf[d];
            kss += (double)kbuf[d] * kbuf[d];
        }
        float q_scale = 1.0f / sqrtf((float)qss + 1e-12f);
        float k_scale = 1.0f / sqrtf((float)kss + 1e-12f);
        for (uint32_t d = 0; d < key_dim; d++) {
            qbuf[d] *= q_scale * inv_sqrt_d;
            kbuf[d] *= k_scale;
        }

        const float *vin = qkv_interleave
            ? qkv + v * head_stride + 2 * key_dim
            : qkv + q_dim_total + k_dim_total + v * val_dim;

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
 *   y[c] = qw36__silu( Σ_{t=0..k-1} wt(t, c) * input_at_t[c] )
 * where input at the latest position is x[c]; older positions live in
 * conv_state. State is shifted left by one and the new x appended.
 *
 * conv_w layout in GGUF is [k, channels] with k = dim[1] (i.e. innermost
 * axis is k). We read wt(t, c) as conv_w[c * k + t]. */
void qw36__conv1d_silu_decode(const float *x, const float *conv_w,
                               float *conv_state, float *y,
                               uint32_t channels, uint32_t k)
{
    /* y[c] = qw36__silu(sum_{t=0..k-1} conv_w[c*k + t] * window[t, c])
     * where window[0..k-2] is conv_state, window[k-1] is x. */
    if (k == 0) {
        for (uint32_t c = 0; c < channels; c++) y[c] = qw36__silu(x[c]);
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
        y[c] = qw36__silu((float)acc);
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


int qw36__attn_deltanet_forward(qw36_forward_ctx *fc,
                                const qw36_layer_weights *L,
                                uint32_t layer_idx)
{
    if (!fc || !L || !fc->st || !fc->cfg) return -1;
    if (L->q_proj || !L->dn_qkv) return 0;

    const qw36_config *c = fc->cfg;
    qw36_state *st = fc->st;
    const size_t hidden = fc->hidden;
    float *x = st->x;

    if (fc->gpu_state) {
        int erc = qw36__ensure_x_dev(fc);
        if (erc) return erc;
        if (qw36__deltanet_dispatch_dev(st, L, c, layer_idx) == 0) {
            *fc->x_dev_valid = 1;
            *fc->x_host_valid = 0;
            return 0;
        }
    }

    int erc = qw36__ensure_x_host(fc);
    if (erc) return erc;

    const uint32_t n_v = c->dn_num_value_heads;
    const uint32_t n_k = c->dn_num_key_heads;
    const uint32_t kd = c->dn_key_head_dim;
    const uint32_t vd = c->dn_value_head_dim;
    const uint32_t q_dim = n_k * kd;
    const uint32_t k_dim_t = n_k * kd;
    const uint32_t v_dim = n_v * vd;
    const uint32_t qkv_dim = q_dim + k_dim_t + v_dim;

    qw36__rmsnorm_dispatch(st->x_rms, x, (const float *)L->input_layernorm,
                           hidden, c->rms_norm_eps);

    const size_t need = (size_t)2 * qkv_dim
                      + (size_t)v_dim
                      + (size_t)n_v * 2
                      + (size_t)v_dim
                      + (size_t)kd * 2;
    float *blk = (float *)calloc(need, sizeof(float));
    if (!blk) return -3;
    float *qkv = blk;
    float *qkv_act = qkv + qkv_dim;
    float *z_proj = qkv_act + qkv_dim;
    float *alpha = z_proj + v_dim;
    float *beta = alpha + n_v;
    float *gout = beta + n_v;
    float *qb = gout + v_dim;
    float *kb = qb + kd;

    int dn_rc = 0;
    dn_rc |= qw36__matmul_lazy(qkv, st->x_rms,
                               (const qw36_lazy_w *)L->dn_qkv,
                               fc->row_scratch);
    dn_rc |= qw36__matmul_lazy(z_proj, st->x_rms,
                               (const qw36_lazy_w *)L->dn_gate,
                               fc->row_scratch);
    dn_rc |= qw36__matmul_lazy(alpha, st->x_rms,
                               (const qw36_lazy_w *)L->dn_alpha,
                               fc->row_scratch);
    dn_rc |= qw36__matmul_lazy(beta, st->x_rms,
                               (const qw36_lazy_w *)L->dn_beta,
                               fc->row_scratch);
    if (dn_rc) {
        fprintf(stderr, "qw36: DN projection failed at layer %u "
                "(unsupported dtype in attn_qkv/gate/alpha/beta - "
                "ggml types %d/%d/%d/%d)\n", layer_idx,
                ((const qw36_lazy_w *)L->dn_qkv)->ggml_type,
                ((const qw36_lazy_w *)L->dn_gate)->ggml_type,
                ((const qw36_lazy_w *)L->dn_alpha)->ggml_type,
                ((const qw36_lazy_w *)L->dn_beta)->ggml_type);
        free(blk);
        return -4;
    }

    static int skip_conv = -1;
    if (skip_conv < 0) {
        const char *e = getenv("QW36_SKIP_CONV1D");
        skip_conv = e && atoi(e) ? 1 : 0;
    }
    if (skip_conv) {
        for (uint32_t i = 0; i < qkv_dim; i++) qkv_act[i] = qw36__silu(qkv[i]);
    } else {
        qw36__conv1d_silu_decode(qkv, (const float *)L->dn_conv1d,
                                 st->conv_state[layer_idx], qkv_act,
                                 qkv_dim, c->dn_conv_kernel_size);
    }

    qw36__gated_delta_decode(qkv_act, beta, alpha,
                             (const float *)L->dn_dt_bias,
                             (const float *)L->dn_a_log,
                             st->delta_state[layer_idx], gout,
                             n_k, n_v, kd, vd, qb, kb);

    const float *dn_norm_w = (const float *)L->dn_norm;
    for (uint32_t v = 0; v < n_v; v++) {
        float *go = gout + (size_t)v * vd;
        float *zp = z_proj + (size_t)v * vd;
        qw36__rmsnorm_dispatch(go, go, dn_norm_w, vd, c->rms_norm_eps);
        for (uint32_t d = 0; d < vd; d++) go[d] = qw36__silu(zp[d]) * go[d];
    }

    if (qw36__matmul_lazy(st->x_rms, gout,
                          (const qw36_lazy_w *)L->dn_out,
                          fc->row_scratch)) {
        fprintf(stderr, "qw36: DN out projection failed at layer %u "
                "(unsupported dtype in ssm_out - ggml type %d)\n",
                layer_idx, ((const qw36_lazy_w *)L->dn_out)->ggml_type);
        free(blk);
        return -5;
    }

    static int skip_dn = -1;
    if (skip_dn < 0) {
        const char *e = getenv("QW36_SKIP_DN");
        skip_dn = e && atoi(e) ? 1 : 0;
    }
    if (!skip_dn) qw36__residual_add_dispatch(x, st->x_rms, hidden);
    free(blk);

    *fc->x_host_valid = 1;
    *fc->x_dev_valid = 0;
    return 0;
}
