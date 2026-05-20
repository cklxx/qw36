/* qw36_metal.metal — Metal Shading Language kernels for Qwen 3.6.
 *
 * Owner: codex.
 *
 * Convention: every kernel takes (uint tid, uint tg_size) for the 1D grid
 * and reads/writes fp32 buffers. Weights may be fp16 or bf16 — pass dtype
 * as a constant and branch in the host code over kernel variants.
 *
 * Build: xcrun -sdk macosx metal -c qw36_metal.metal -o qw36_metal.air &&
 *        xcrun -sdk macosx metallib qw36_metal.air -o default.metallib
 */

#include <metal_stdlib>
using namespace metal;

constant uint QW36_DTYPE_F32  = 0;
constant uint QW36_DTYPE_F16  = 1;
constant uint QW36_DTYPE_BF16 = 30;

static inline float qw36_bf16_to_f32(ushort v)
{
    uint bits = uint(v) << 16;
    return as_type<float>(bits);
}

static inline float qw36_load_scalar(device const uchar *ptr, uint dtype, uint i)
{
    if (dtype == QW36_DTYPE_F32) {
        return ((device const float *)ptr)[i];
    } else if (dtype == QW36_DTYPE_F16) {
        return float(((device const half *)ptr)[i]);
    } else if (dtype == QW36_DTYPE_BF16) {
        return qw36_bf16_to_f32(((device const ushort *)ptr)[i]);
    }
    return 0.0f;
}

/* ---------------------------------------------------------------- */
/* RMSNorm: out[i] = x[i] * rsqrt(mean(x^2) + eps) * w[i]            */
/* ---------------------------------------------------------------- */
kernel void qw36_rmsnorm_f32(
    device float       *out      [[buffer(0)]],
    device const uchar *x        [[buffer(1)]],
    device const uchar *w        [[buffer(2)]],
    constant uint      &hidden   [[buffer(3)]],
    constant float     &eps      [[buffer(4)]],
    constant uint      &x_dtype  [[buffer(5)]],
    constant uint      &w_dtype  [[buffer(6)]],
    uint                tid      [[thread_position_in_grid]])
{
    if (tid >= hidden) return;
    float ss = 0.0f;
    for (uint i = 0; i < hidden; ++i) {
        float v = qw36_load_scalar(x, x_dtype, i);
        ss += v * v;
    }
    float scale = rsqrt(ss / float(hidden) + eps);
    out[tid] = qw36_load_scalar(x, x_dtype, tid) * scale *
               qw36_load_scalar(w, w_dtype, tid);
}

/* ---------------------------------------------------------------- */
/* Matmul: y[r] = sum_c x[c] * W[r,c]                                */
/* ---------------------------------------------------------------- */
kernel void qw36_matmul_f32(
    device float       *y        [[buffer(0)]],
    device const uchar *x        [[buffer(1)]],
    device const uchar *w        [[buffer(2)]],
    constant uint      &rows     [[buffer(3)]],
    constant uint      &cols     [[buffer(4)]],
    constant uint      &batch    [[buffer(5)]],
    constant uint      &x_dtype  [[buffer(6)]],
    constant uint      &w_dtype  [[buffer(7)]],
    uint                gid      [[thread_position_in_grid]])
{
    uint total = rows * batch;
    if (gid >= total) return;
    uint b = gid / rows;
    uint r = gid - b * rows;
    uint xb = b * cols;
    uint wr = r * cols;
    float acc = 0.0f;
    for (uint c = 0; c < cols; ++c)
        acc += qw36_load_scalar(x, x_dtype, xb + c) *
               qw36_load_scalar(w, w_dtype, wr + c);
    y[gid] = acc;
}

/* ---------------------------------------------------------------- */
/* SwiGLU MLP composition stays in host code (three matmuls + silu); */
/* a dedicated fused kernel can come later.                          */
/* ---------------------------------------------------------------- */
kernel void qw36_silu_mul_f32(
    device float       *gate_io  [[buffer(0)]], /* in: gate, out: silu(gate)*up */
    device const float *up       [[buffer(1)]],
    constant uint      &n        [[buffer(2)]],
    uint                tid      [[thread_position_in_grid]])
{
    if (tid >= n) return;
    float g = gate_io[tid];
    gate_io[tid] = (g / (1.0f + exp(-g))) * up[tid];
}

/* ---------------------------------------------------------------- */
/* DeltaNet decode helpers. State layout follows common/qw36.c:      */
/* conv_state [kernel-1, channels], delta_state [Hv, Dk, Dv].        */
/* ---------------------------------------------------------------- */
kernel void qw36_dn_conv1d_silu_f32(
    device float       *y           [[buffer(0)]],
    device const float *x           [[buffer(1)]],
    device const float *conv_w      [[buffer(2)]],
    device float       *conv_state  [[buffer(3)]],
    constant uint      &channels    [[buffer(4)]],
    constant uint      &kernel_size [[buffer(5)]],
    uint                c           [[thread_position_in_grid]])
{
    if (c >= channels) return;
    if (kernel_size == 0) {
        float v = x[c];
        y[c] = v / (1.0f + exp(-v));
        return;
    }

    device const float *wt = conv_w + c * kernel_size;
    float acc = 0.0f;
    if (kernel_size > 1) {
        for (uint t = 0; t + 1 < kernel_size; ++t)
            acc += wt[t] * conv_state[t * channels + c];
    }
    acc += wt[kernel_size - 1] * x[c];
    y[c] = acc / (1.0f + exp(-acc));

    if (kernel_size > 1) {
        for (uint t = 0; t + 2 < kernel_size; ++t)
            conv_state[t * channels + c] = conv_state[(t + 1) * channels + c];
        conv_state[(kernel_size - 2) * channels + c] = x[c];
    }
}

kernel void qw36_dn_gated_delta_f32(
    device float       *y          [[buffer(0)]],
    device const float *qkv        [[buffer(1)]],
    device const float *beta_raw   [[buffer(2)]],
    device const float *alpha_raw  [[buffer(3)]],
    device const float *dt_bias    [[buffer(4)]],
    device const float *a_log      [[buffer(5)]],
    device float       *state      [[buffer(6)]],
    constant uint      &n_key      [[buffer(7)]],
    constant uint      &n_value    [[buffer(8)]],
    constant uint      &key_dim    [[buffer(9)]],
    constant uint      &val_dim    [[buffer(10)]],
    uint                gid        [[thread_position_in_grid]])
{
    uint total = n_value * val_dim;
    if (gid >= total || n_key == 0 || key_dim == 0 || val_dim == 0) return;

    uint v = gid / val_dim;
    uint d = gid - v * val_dim;
    uint kh = v % n_key;
    uint q_dim_total = n_key * key_dim;
    uint k_dim_total = q_dim_total;

    float qss = 0.0f;
    float kss = 0.0f;
    device const float *q_head = qkv + kh * key_dim;
    device const float *k_head = qkv + q_dim_total + kh * key_dim;
    for (uint j = 0; j < key_dim; ++j) {
        float qv = q_head[j];
        float kv = k_head[j];
        qss += qv * qv;
        kss += kv * kv;
    }
    float q_scale = rsqrt(qss + 1.0e-12f) / sqrt(float(key_dim));
    float k_scale = rsqrt(kss + 1.0e-12f);

    float av = alpha_raw[v] + dt_bias[v];
    float softplus = av > 20.0f ? av : log(1.0f + exp(av));
    float exp_g = exp(-exp(a_log[v]) * softplus);
    float beta = 1.0f / (1.0f + exp(-beta_raw[v]));

    device float *S = state + (v * key_dim * val_dim);
    float kv_mem = 0.0f;
    for (uint j = 0; j < key_dim; ++j) {
        float kj = k_head[j] * k_scale;
        uint off = j * val_dim + d;
        float s = S[off] * exp_g;
        S[off] = s;
        kv_mem += s * kj;
    }

    device const float *vin = qkv + q_dim_total + k_dim_total + v * val_dim;
    float delta = (vin[d] - kv_mem) * beta;
    float out = 0.0f;
    for (uint j = 0; j < key_dim; ++j) {
        float kj = k_head[j] * k_scale;
        float qj = q_head[j] * q_scale;
        uint off = j * val_dim + d;
        float s = S[off] + delta * kj;
        S[off] = s;
        out += s * qj;
    }
    y[gid] = out;
}

kernel void qw36_dn_gated_rmsnorm_f32(
    device float       *y        [[buffer(0)]],
    device const float *x        [[buffer(1)]],
    device const float *z        [[buffer(2)]],
    device const float *w        [[buffer(3)]],
    constant uint      &n_value  [[buffer(4)]],
    constant uint      &val_dim  [[buffer(5)]],
    constant float     &eps      [[buffer(6)]],
    uint                gid      [[thread_position_in_grid]])
{
    uint total = n_value * val_dim;
    if (gid >= total || val_dim == 0) return;
    uint v = gid / val_dim;
    uint d = gid - v * val_dim;
    device const float *xh = x + v * val_dim;
    float ss = 0.0f;
    for (uint i = 0; i < val_dim; ++i) ss += xh[i] * xh[i];
    float scale = rsqrt(ss / float(val_dim) + eps);
    float zg = z[gid];
    float gate = zg / (1.0f + exp(-zg));
    y[gid] = gate * x[gid] * scale * w[d];
}

kernel void qw36_compute_g_beta_norm_qk(
    device float       *q_out     [[buffer(0)]],
    device float       *k_out     [[buffer(1)]],
    device float       *v_out     [[buffer(2)]],
    device float       *g_out     [[buffer(3)]],
    device float       *beta_out  [[buffer(4)]],
    device const float *qkv       [[buffer(5)]],
    device const float *alpha_raw [[buffer(6)]],
    device const float *beta_raw  [[buffer(7)]],
    device const float *dt_bias   [[buffer(8)]],
    device const float *a_log     [[buffer(9)]],
    constant uint      &Hk        [[buffer(10)]],
    constant uint      &Hv        [[buffer(11)]],
    constant uint      &Dk        [[buffer(12)]],
    constant uint      &Dv        [[buffer(13)]],
    uint                gid       [[thread_position_in_grid]])
{
    uint q_total = Hk * Dk;
    uint v_total = Hv * Dv;
    uint group = (Hk == 0) ? 1 : (Hv / Hk);
    if (group == 0) group = 1;

    if (gid < q_total) {
        uint h = gid / Dk;
        uint d = gid - h * Dk;
        device const float *qh = qkv + h * Dk;
        device const float *kh = qkv + q_total + h * Dk;
        float qss = 0.0f;
        float kss = 0.0f;
        for (uint i = 0; i < Dk; ++i) {
            float qv = qh[i];
            float kv = kh[i];
            qss += qv * qv;
            kss += kv * kv;
        }
        q_out[gid] = qh[d] * rsqrt(qss + 1.0e-12f) / sqrt(float(Dk));
        k_out[gid] = kh[d] * rsqrt(kss + 1.0e-12f);
    }

    if (gid < v_total) {
        uint raw_v = gid / Dv;
        uint d = gid - raw_v * Dv;
        uint grouped_v = (Hv % Hk == 0) ? ((raw_v % Hk) * group + raw_v / Hk) : raw_v;
        v_out[grouped_v * Dv + d] = qkv[q_total * 2 + raw_v * Dv + d];
    }

    if (gid < Hv) {
        uint raw_v = gid;
        uint grouped_v = (Hv % Hk == 0) ? ((raw_v % Hk) * group + raw_v / Hk) : raw_v;
        float av = alpha_raw[raw_v] + dt_bias[raw_v];
        float softplus = av > 20.0f ? av : log(1.0f + exp(av));
        g_out[grouped_v] = exp(-exp(a_log[raw_v]) * softplus);
        beta_out[grouped_v] = 1.0f / (1.0f + exp(-beta_raw[raw_v]));
    }
}

kernel void qw36_dn_reorder_grouped_y_to_raw_f32(
    device float       *y_raw     [[buffer(0)]],
    device const float *y_grouped [[buffer(1)]],
    constant uint      &Hk        [[buffer(2)]],
    constant uint      &Hv        [[buffer(3)]],
    constant uint      &Dv        [[buffer(4)]],
    uint                gid       [[thread_position_in_grid]])
{
    uint total = Hv * Dv;
    if (gid >= total) return;
    uint raw_v = gid / Dv;
    uint d = gid - raw_v * Dv;
    uint group = (Hk == 0) ? 1 : (Hv / Hk);
    if (group == 0) group = 1;
    uint grouped_v = (Hv % Hk == 0) ? ((raw_v % Hk) * group + raw_v / Hk) : raw_v;
    y_raw[gid] = y_grouped[grouped_v * Dv + d];
}

/* ---------------------------------------------------------------- */
/* Residual add                                                      */
/* ---------------------------------------------------------------- */
kernel void qw36_residual_add_f32(
    device float       *x        [[buffer(0)]],
    device const float *y        [[buffer(1)]],
    constant uint      &n        [[buffer(2)]],
    uint                tid      [[thread_position_in_grid]])
{
    if (tid < n) x[tid] += y[tid];
}

/* ---------------------------------------------------------------- */
/* Embedding lookup                                                  */
/* ---------------------------------------------------------------- */
kernel void qw36_embedding_lookup_f32(
    device float       *y        [[buffer(0)]],
    device const uchar *embed    [[buffer(1)]],
    constant uint      &token    [[buffer(2)]],
    constant uint      &hidden   [[buffer(3)]],
    constant uint      &dtype    [[buffer(4)]],
    uint                tid      [[thread_position_in_grid]])
{
    if (tid < hidden)
        y[tid] = qw36_load_scalar(embed, dtype, token * hidden + tid);
}

/* ---------------------------------------------------------------- */
/* Per-head RMSNorm + RoPE, in-place.                               */
/* ---------------------------------------------------------------- */
kernel void qw36_head_norm_rope_f32(
    device float       *x        [[buffer(0)]],
    device const uchar *w        [[buffer(1)]],
    constant uint      &heads    [[buffer(2)]],
    constant uint      &head_dim [[buffer(3)]],
    constant uint      &seq_pos  [[buffer(4)]],
    constant float     &theta    [[buffer(5)]],
    constant float     &partial  [[buffer(6)]],
    constant float     &eps      [[buffer(7)]],
    constant uint      &w_dtype  [[buffer(8)]],
    uint                h        [[thread_position_in_grid]])
{
    if (h >= heads) return;
    device float *xh = x + h * head_dim;
    float ss = 0.0f;
    for (uint d = 0; d < head_dim; ++d) ss += xh[d] * xh[d];
    float scale = rsqrt(ss / float(head_dim) + eps);

    uint rot_dim = uint(float(head_dim) * partial);
    if (rot_dim > head_dim) rot_dim = head_dim;
    rot_dim &= ~1u;

    uint half_dim = rot_dim / 2;
    for (uint pair = 0; pair < half_dim; ++pair) {
        uint d0 = pair;
        uint d1 = pair + half_dim;
        float x0 = xh[d0] * scale * qw36_load_scalar(w, w_dtype, d0);
        float x1 = xh[d1] * scale * qw36_load_scalar(w, w_dtype, d1);
        float inv_freq = 1.0f / pow(theta, (2.0f * float(pair)) / float(rot_dim));
        float angle = float(seq_pos) * inv_freq;
        float c = cos(angle);
        float s = sin(angle);
        xh[d0] = x0 * c - x1 * s;
        xh[d1] = x0 * s + x1 * c;
    }
    for (uint d = rot_dim; d < head_dim; ++d)
        xh[d] *= scale * qw36_load_scalar(w, w_dtype, d);
}

/* ---------------------------------------------------------------- */
/* KV cache append: cache[seq_pos] = current k/v.                    */
/* ---------------------------------------------------------------- */
kernel void qw36_kv_append_f32(
    device float       *k_cache      [[buffer(0)]],
    device float       *v_cache      [[buffer(1)]],
    device const float *k            [[buffer(2)]],
    device const float *v            [[buffer(3)]],
    constant uint      &seq_pos      [[buffer(4)]],
    constant uint      &seq_capacity [[buffer(5)]],
    constant uint      &kv_len       [[buffer(6)]],
    uint                tid          [[thread_position_in_grid]])
{
    if (tid >= kv_len || seq_pos >= seq_capacity) return;
    uint off = seq_pos * kv_len + tid;
    k_cache[off] = k[tid];
    v_cache[off] = v[tid];
}

/* ---------------------------------------------------------------- */
/* Attention score: q_h dot k_cache[pos, kv_h] / sqrt(head_dim).     */
/* ---------------------------------------------------------------- */
kernel void qw36_attn_scores_f32(
    device float       *scores       [[buffer(0)]],
    device const float *q            [[buffer(1)]],
    device const float *k_cache      [[buffer(2)]],
    constant uint      &n_heads      [[buffer(3)]],
    constant uint      &n_kv         [[buffer(4)]],
    constant uint      &head_dim     [[buffer(5)]],
    constant uint      &seq_pos      [[buffer(6)]],
    constant uint      &seq_capacity [[buffer(7)]],
    uint                gid          [[thread_position_in_grid]])
{
    uint count = seq_pos + 1;
    uint total = n_heads * count;
    if (gid >= total || count > seq_capacity) return;
    uint h = gid / count;
    uint pos = gid - h * count;
    uint kv_h = h * n_kv / n_heads;
    uint kv_len = n_kv * head_dim;
    device const float *qh = q + h * head_dim;
    device const float *kh = k_cache + pos * kv_len + kv_h * head_dim;
    float acc = 0.0f;
    for (uint d = 0; d < head_dim; ++d) acc += qh[d] * kh[d];
    scores[gid] = acc / sqrt(float(head_dim));
}

/* ---------------------------------------------------------------- */
/* Stable softmax per head, in-place over positions 0..seq_pos.      */
/* ---------------------------------------------------------------- */
kernel void qw36_attn_softmax_f32(
    device float  *scores  [[buffer(0)]],
    constant uint &n_heads [[buffer(1)]],
    constant uint &seq_pos [[buffer(2)]],
    uint           h       [[thread_position_in_grid]])
{
    if (h >= n_heads) return;
    uint count = seq_pos + 1;
    device float *row = scores + h * count;
    float maxv = row[0];
    for (uint i = 1; i < count; ++i) maxv = max(maxv, row[i]);
    float sum = 0.0f;
    for (uint i = 0; i < count; ++i) {
        float e = exp(row[i] - maxv);
        row[i] = e;
        sum += e;
    }
    float inv = 1.0f / sum;
    for (uint i = 0; i < count; ++i) row[i] *= inv;
}

/* ---------------------------------------------------------------- */
/* Attention combine: y_h = softmax(scores_h) @ v_cache[kv_h].       */
/* ---------------------------------------------------------------- */
kernel void qw36_attn_combine_f32(
    device float       *y            [[buffer(0)]],
    device const float *scores       [[buffer(1)]],
    device const float *v_cache      [[buffer(2)]],
    constant uint      &n_heads      [[buffer(3)]],
    constant uint      &n_kv         [[buffer(4)]],
    constant uint      &head_dim     [[buffer(5)]],
    constant uint      &seq_pos      [[buffer(6)]],
    constant uint      &seq_capacity [[buffer(7)]],
    uint                gid          [[thread_position_in_grid]])
{
    uint hidden = n_heads * head_dim;
    if (gid >= hidden || seq_pos >= seq_capacity) return;
    uint h = gid / head_dim;
    uint d = gid - h * head_dim;
    uint kv_h = h * n_kv / n_heads;
    uint count = seq_pos + 1;
    uint kv_len = n_kv * head_dim;
    device const float *score_row = scores + h * count;
    float acc = 0.0f;
    for (uint pos = 0; pos < count; ++pos) {
        uint off = pos * kv_len + kv_h * head_dim + d;
        acc += score_row[pos] * v_cache[off];
    }
    y[gid] = acc;
}

/* ---------------------------------------------------------------- */
/* Attention is split across q/k/v matmul, head_norm_rope, kv_append, */
/* attn_scores, attn_softmax, and attn_combine.                       */
/* ---------------------------------------------------------------- */

/* =============================================================================
 * Gated Delta Rule step — transliterated from agent-infer's
 *   ../agent-infer/crates/mlx-sys/src/mlx_qwen35_model.cpp:203-275
 *   (fast::metal_kernel("gated_delta_step", ...))
 * Adapted to fp32 buffers and runtime-shape constants.
 *
 * Inputs:
 *   q         [B, T, Hk, Dk]
 *   k         [B, T, Hk, Dk]
 *   v         [B, T, Hv, Dv]
 *   g         [B, T, Hv]
 *   beta      [B, T, Hv]
 *   state_in  [B, Hv, Dv, Dk]
 * Outputs:
 *   y         [B, T, Hv, Dv]
 *   state_out [B, Hv, Dv, Dk]
 *
 * Threadgroup layout:
 *   grid.x = 32 (one simdgroup along Dk; each lane handles n_per_t entries)
 *   grid.y = Dv
 *   grid.z = B * Hv
 *   threadgroup = (32, 1, 1)
 * ============================================================================= */
kernel void qw36_gated_delta_step_f32(
    device const float *q          [[buffer(0)]],
    device const float *k          [[buffer(1)]],
    device const float *v          [[buffer(2)]],
    device const float *g          [[buffer(3)]],
    device const float *beta       [[buffer(4)]],
    device const float *state_in   [[buffer(5)]],
    device       float *y          [[buffer(6)]],
    device       float *state_out  [[buffer(7)]],
    constant uint &T   [[buffer(8)]],
    constant uint &Hk  [[buffer(9)]],
    constant uint &Hv  [[buffer(10)]],
    constant uint &Dk  [[buffer(11)]],
    constant uint &Dv  [[buffer(12)]],
    uint3 tpig  [[thread_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint  tsim  [[thread_index_in_simdgroup]])
{
    uint n      = tpig.z;
    uint b_idx  = n / Hv;
    uint hv_idx = n % Hv;
    uint hk_idx = hv_idx / (Hv / Hk);
    uint n_per_t = Dk / 32u;

    device const float *q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
    device const float *k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
    device const float *v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
    device       float *y_ = y + b_idx * T * Hv * Dv + hv_idx * Dv;

    uint dk_idx = tpitg.x;
    uint dv_idx = tpig.y;

    device const float *g_    = g    + b_idx * T * Hv;
    device const float *beta_ = beta + b_idx * T * Hv;

    device const float *i_state = state_in  + (n * Dv + dv_idx) * Dk;
    device       float *o_state = state_out + (n * Dv + dv_idx) * Dk;

    float state[8];      /* Dk <= 256 → n_per_t <= 8 */
    for (uint i = 0; i < n_per_t; ++i)
        state[i] = i_state[n_per_t * dk_idx + i];

    for (uint t = 0; t < T; ++t) {
        float g_val = g_[hv_idx];
        float kv_mem = 0.0f;
        for (uint i = 0; i < n_per_t; ++i) {
            uint s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_val;
            kv_mem  += state[i] * k_[s_idx];
        }
        kv_mem = simd_sum(kv_mem);

        float delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

        float out = 0.0f;
        for (uint i = 0; i < n_per_t; ++i) {
            uint s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
            out     += state[i] * q_[s_idx];
        }
        out = simd_sum(out);
        if (tsim == 0) y_[dv_idx] = out;

        q_    += Hk * Dk;
        k_    += Hk * Dk;
        v_    += Hv * Dv;
        y_    += Hv * Dv;
        g_    += Hv;
        beta_ += Hv;
    }
    for (uint i = 0; i < n_per_t; ++i)
        o_state[n_per_t * dk_idx + i] = state[i];
}
