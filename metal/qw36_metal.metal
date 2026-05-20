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

static inline void qw36_store_scalar(device uchar *ptr, uint dtype, uint i, float v)
{
    if (dtype == QW36_DTYPE_F32) {
        ((device float *)ptr)[i] = v;
    } else if (dtype == QW36_DTYPE_F16) {
        ((device half *)ptr)[i] = half(v);
    } else if (dtype == QW36_DTYPE_BF16) {
        uint bits = as_type<uint>(v);
        ((device ushort *)ptr)[i] = ushort(bits >> 16);
    }
}

/* ---------------------------------------------------------------- */
/* RMSNorm: out[i] = x[i] * rsqrt(mean(x^2) + eps) * w[i]            */
/* ---------------------------------------------------------------- */
kernel void qw36_rmsnorm_f32(
    device uchar       *out      [[buffer(0)]],
    device const uchar *x        [[buffer(1)]],
    device const uchar *w        [[buffer(2)]],
    constant uint      &hidden   [[buffer(3)]],
    constant float     &eps      [[buffer(4)]],
    constant uint      &x_dtype  [[buffer(5)]],
    constant uint      &w_dtype  [[buffer(6)]],
    constant uint      &out_dtype [[buffer(7)]],
    uint                tid      [[thread_position_in_grid]])
{
    if (tid >= hidden) return;
    float ss = 0.0f;
    for (uint i = 0; i < hidden; ++i) {
        float v = qw36_load_scalar(x, x_dtype, i);
        ss += v * v;
    }
    float scale = rsqrt(ss / float(hidden) + eps);
    float val = qw36_load_scalar(x, x_dtype, tid) * scale *
                qw36_load_scalar(w, w_dtype, tid);
    qw36_store_scalar(out, out_dtype, tid, val);
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

kernel void qw36_f32_to_f16_f32(
    device half        *out [[buffer(0)]],
    device const float *in  [[buffer(1)]],
    constant uint      &n   [[buffer(2)]],
    uint                tid [[thread_position_in_grid]])
{
    if (tid < n) out[tid] = half(in[tid]);
}

kernel void qw36_f16_to_f32_f32(
    device float      *out [[buffer(0)]],
    device const half *in  [[buffer(1)]],
    constant uint     &n   [[buffer(2)]],
    uint               tid [[thread_position_in_grid]])
{
    if (tid < n) out[tid] = float(in[tid]);
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

kernel void qw36_moe_route_f32(
    device const float *router_logits [[buffer(0)]],
    device float       *top_probs     [[buffer(1)]],
    device uint        *top_idx       [[buffer(2)]],
    constant uint      &num_experts   [[buffer(3)]],
    constant uint      &top_k         [[buffer(4)]],
    uint                tid           [[thread_position_in_grid]])
{
    if (tid != 0 || top_k == 0) return;
    for (uint k = 0; k < top_k; ++k) {
        float best = -INFINITY;
        uint best_i = 0;
        for (uint e = 0; e < num_experts; ++e) {
            float v = router_logits[e];
            bool used = false;
            for (uint j = 0; j < k; ++j) used = used || (top_idx[j] == e);
            if (!used && v > best) {
                best = v;
                best_i = e;
            }
        }
        top_idx[k] = best_i;
        top_probs[k] = best;
    }

    float maxv = top_probs[0];
    for (uint k = 1; k < top_k; ++k) maxv = max(maxv, top_probs[k]);
    float sum = 0.0f;
    for (uint k = 0; k < top_k; ++k) {
        float p = exp(top_probs[k] - maxv);
        top_probs[k] = p;
        sum += p;
    }
    float inv_sum = 1.0f / sum;
    for (uint k = 0; k < top_k; ++k) top_probs[k] *= inv_sum;
}

kernel void qw36_moe_gate_up_f32(
    device float       *act          [[buffer(0)]],
    device const float *x            [[buffer(1)]],
    device const uchar *expert_gate  [[buffer(2)]],
    device const uchar *expert_up    [[buffer(3)]],
    device const uint  *top_idx      [[buffer(4)]],
    constant uint      &hidden       [[buffer(5)]],
    constant uint      &inter        [[buffer(6)]],
    constant uint      &top_k        [[buffer(7)]],
    constant uint      &gate_dtype   [[buffer(8)]],
    constant uint      &up_dtype     [[buffer(9)]],
    uint                gid          [[thread_position_in_grid]])
{
    uint total = top_k * inter;
    if (gid >= total) return;
    uint t = gid / inter;
    uint i = gid - t * inter;
    uint e = top_idx[t];
    ulong base = (ulong)e * (ulong)inter * (ulong)hidden + (ulong)i * (ulong)hidden;

    float g = 0.0f;
    float u = 0.0f;
    for (uint c = 0; c < hidden; ++c) {
        float xv = x[c];
        g += xv * qw36_load_scalar(expert_gate, gate_dtype, uint(base + c));
        u += xv * qw36_load_scalar(expert_up, up_dtype, uint(base + c));
    }
    act[gid] = (g / (1.0f + exp(-g))) * u;
}

kernel void qw36_moe_down_combine_f32(
    device float       *y             [[buffer(0)]],
    device const float *act           [[buffer(1)]],
    device const uchar *expert_down   [[buffer(2)]],
    device const float *top_probs     [[buffer(3)]],
    device const uint  *top_idx       [[buffer(4)]],
    constant uint      &hidden        [[buffer(5)]],
    constant uint      &inter         [[buffer(6)]],
    constant uint      &top_k         [[buffer(7)]],
    constant uint      &down_dtype    [[buffer(8)]],
    uint                d             [[thread_position_in_grid]])
{
    if (d >= hidden) return;
    float out = 0.0f;
    for (uint t = 0; t < top_k; ++t) {
        uint e = top_idx[t];
        ulong base = (ulong)e * (ulong)hidden * (ulong)inter + (ulong)d * (ulong)inter;
        device const float *a = act + (ulong)t * inter;
        float acc = 0.0f;
        for (uint i = 0; i < inter; ++i)
            acc += a[i] * qw36_load_scalar(expert_down, down_dtype, uint(base + i));
        out += top_probs[t] * acc;
    }
    y[d] = out;
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
    uint k_total = q_total;
    uint v_total = Hv * Dv;

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
        uint v = gid / Dv;
        uint d = gid - v * Dv;
        v_out[v * Dv + d] = qkv[q_total + k_total + v * Dv + d];
    }

    if (gid < Hv) {
        uint v = gid;
        float av = alpha_raw[v] + dt_bias[v];
        float softplus = av > 20.0f ? av : log(1.0f + exp(av));
        g_out[v] = exp(-exp(a_log[v]) * softplus);
        beta_out[v] = 1.0f / (1.0f + exp(-beta_raw[v]));
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
    uint grouped_v = raw_v;
    if (Hk != 0 && Hv % Hk == 0)
        grouped_v = (raw_v % Hk) * group + raw_v / Hk;
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
    device uchar       *k_cache      [[buffer(0)]],
    device uchar       *v_cache      [[buffer(1)]],
    device const float *k            [[buffer(2)]],
    device const float *v            [[buffer(3)]],
    constant uint      &seq_pos      [[buffer(4)]],
    constant uint      &seq_capacity [[buffer(5)]],
    constant uint      &kv_len       [[buffer(6)]],
    constant uint      &k_cache_dtype [[buffer(7)]],
    constant uint      &v_cache_dtype [[buffer(8)]],
    uint                tid          [[thread_position_in_grid]])
{
    if (tid >= kv_len || seq_pos >= seq_capacity) return;
    uint off = seq_pos * kv_len + tid;
    qw36_store_scalar(k_cache, k_cache_dtype, off, k[tid]);
    qw36_store_scalar(v_cache, v_cache_dtype, off, v[tid]);
}

/* ---------------------------------------------------------------- */
/* Attention score: q_h dot k_cache[pos, kv_h] / sqrt(head_dim).     */
/* ---------------------------------------------------------------- */
kernel void qw36_attn_scores_f32(
    device float       *scores       [[buffer(0)]],
    device const float *q            [[buffer(1)]],
    device const uchar *k_cache      [[buffer(2)]],
    constant uint      &n_heads      [[buffer(3)]],
    constant uint      &n_kv         [[buffer(4)]],
    constant uint      &head_dim     [[buffer(5)]],
    constant uint      &seq_pos      [[buffer(6)]],
    constant uint      &seq_capacity [[buffer(7)]],
    constant uint      &k_cache_dtype [[buffer(8)]],
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
    uint k_base = pos * kv_len + kv_h * head_dim;
    float acc = 0.0f;
    for (uint d = 0; d < head_dim; ++d)
        acc += qh[d] * qw36_load_scalar(k_cache, k_cache_dtype, k_base + d);
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
    device const uchar *v_cache      [[buffer(2)]],
    constant uint      &n_heads      [[buffer(3)]],
    constant uint      &n_kv         [[buffer(4)]],
    constant uint      &head_dim     [[buffer(5)]],
    constant uint      &seq_pos      [[buffer(6)]],
    constant uint      &seq_capacity [[buffer(7)]],
    constant uint      &v_cache_dtype [[buffer(8)]],
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
        acc += score_row[pos] * qw36_load_scalar(v_cache, v_cache_dtype, off);
    }
    y[gid] = acc;
}

/* ---------------------------------------------------------------- */
/* Decode attention fast path: after q/k/v matmul, reduce the six tiny */
/* post-projection dispatches to two kernels.                          */
/* ---------------------------------------------------------------- */
kernel void qw36_attn_prep_decode_f32(
    device float       *q            [[buffer(0)]],
    device float       *k            [[buffer(1)]],
    device const float *v            [[buffer(2)]],
    device uchar       *k_cache      [[buffer(3)]],
    device uchar       *v_cache      [[buffer(4)]],
    device const uchar *q_norm       [[buffer(5)]],
    device const uchar *k_norm       [[buffer(6)]],
    constant uint      &n_heads      [[buffer(7)]],
    constant uint      &n_kv         [[buffer(8)]],
    constant uint      &head_dim     [[buffer(9)]],
    constant uint      &seq_pos      [[buffer(10)]],
    constant uint      &seq_capacity [[buffer(11)]],
    constant float     &theta        [[buffer(12)]],
    constant float     &partial      [[buffer(13)]],
    constant float     &eps          [[buffer(14)]],
    constant uint      &q_w_dtype    [[buffer(15)]],
    constant uint      &k_w_dtype    [[buffer(16)]],
    constant uint      &k_cache_dtype [[buffer(17)]],
    constant uint      &v_cache_dtype [[buffer(18)]],
    uint                gid          [[thread_position_in_grid]])
{
    uint rot_dim = uint(float(head_dim) * partial);
    if (rot_dim > head_dim) rot_dim = head_dim;
    rot_dim &= ~1u;
    uint half_dim = rot_dim / 2;

    if (gid < n_heads) {
        device float *qh = q + gid * head_dim;
        float ss = 0.0f;
        for (uint d = 0; d < head_dim; ++d) ss += qh[d] * qh[d];
        float scale = rsqrt(ss / float(head_dim) + eps);
        for (uint pair = 0; pair < half_dim; ++pair) {
            uint d0 = pair;
            uint d1 = pair + half_dim;
            float x0 = qh[d0] * scale * qw36_load_scalar(q_norm, q_w_dtype, d0);
            float x1 = qh[d1] * scale * qw36_load_scalar(q_norm, q_w_dtype, d1);
            float inv_freq = 1.0f / pow(theta, (2.0f * float(pair)) / float(rot_dim));
            float angle = float(seq_pos) * inv_freq;
            float c = cos(angle);
            float s = sin(angle);
            qh[d0] = x0 * c - x1 * s;
            qh[d1] = x0 * s + x1 * c;
        }
        for (uint d = rot_dim; d < head_dim; ++d)
            qh[d] *= scale * qw36_load_scalar(q_norm, q_w_dtype, d);
        return;
    }

    uint kvh = gid - n_heads;
    if (kvh >= n_kv) return;
    device float *kh = k + kvh * head_dim;
    float ss = 0.0f;
    for (uint d = 0; d < head_dim; ++d) ss += kh[d] * kh[d];
    float scale = rsqrt(ss / float(head_dim) + eps);
    for (uint pair = 0; pair < half_dim; ++pair) {
        uint d0 = pair;
        uint d1 = pair + half_dim;
        float x0 = kh[d0] * scale * qw36_load_scalar(k_norm, k_w_dtype, d0);
        float x1 = kh[d1] * scale * qw36_load_scalar(k_norm, k_w_dtype, d1);
        float inv_freq = 1.0f / pow(theta, (2.0f * float(pair)) / float(rot_dim));
        float angle = float(seq_pos) * inv_freq;
        float c = cos(angle);
        float s = sin(angle);
        kh[d0] = x0 * c - x1 * s;
        kh[d1] = x0 * s + x1 * c;
    }
    for (uint d = rot_dim; d < head_dim; ++d)
        kh[d] *= scale * qw36_load_scalar(k_norm, k_w_dtype, d);

    if (seq_pos >= seq_capacity) return;
    uint kv_len = n_kv * head_dim;
    uint cache_base = seq_pos * kv_len + kvh * head_dim;
    device const float *vh = v + kvh * head_dim;
    for (uint d = 0; d < head_dim; ++d) {
        qw36_store_scalar(k_cache, k_cache_dtype, cache_base + d, kh[d]);
        qw36_store_scalar(v_cache, v_cache_dtype, cache_base + d, vh[d]);
    }
}

kernel void qw36_attn_score_combine_tg_f32(
    device float       *y            [[buffer(0)]],
    device const float *q            [[buffer(1)]],
    device const uchar *k_cache      [[buffer(2)]],
    device const uchar *v_cache      [[buffer(3)]],
    constant uint      &n_heads      [[buffer(4)]],
    constant uint      &n_kv         [[buffer(5)]],
    constant uint      &head_dim     [[buffer(6)]],
    constant uint      &seq_pos      [[buffer(7)]],
    constant uint      &seq_capacity [[buffer(8)]],
    constant uint      &tg_size      [[buffer(9)]],
    constant uint      &k_cache_dtype [[buffer(10)]],
    constant uint      &v_cache_dtype [[buffer(11)]],
    threadgroup float  *scratch      [[threadgroup(0)]],
    uint                lane         [[thread_index_in_threadgroup]],
    uint3               tg_pos       [[threadgroup_position_in_grid]])
{
    uint h = tg_pos.x;
    if (h >= n_heads || seq_pos >= seq_capacity) return;

    uint count = seq_pos + 1;
    uint kvh = h * n_kv / n_heads;
    uint kv_len = n_kv * head_dim;
    device const float *qh = q + h * head_dim;
    threadgroup float *partials = scratch;
    threadgroup float *scores = scratch + tg_size;
    float inv_sqrt_d = rsqrt(float(head_dim));

    for (uint t = 0; t < count; ++t) {
        uint k_base = t * kv_len + kvh * head_dim;
        partials[lane] = (lane < head_dim)
            ? qh[lane] * qw36_load_scalar(k_cache, k_cache_dtype, k_base + lane)
            : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tg_size >> 1; stride > 0; stride >>= 1) {
            if (lane < stride) partials[lane] += partials[lane + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) scores[t] = partials[0] * inv_sqrt_d;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        float maxv = scores[0];
        for (uint t = 1; t < count; ++t) maxv = max(maxv, scores[t]);
        float sum = 0.0f;
        for (uint t = 0; t < count; ++t) {
            float e = exp(scores[t] - maxv);
            scores[t] = e;
            sum += e;
        }
        float inv_sum = 1.0f / sum;
        for (uint t = 0; t < count; ++t) scores[t] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane < head_dim) {
        float acc = 0.0f;
        for (uint t = 0; t < count; ++t) {
            uint off = t * kv_len + kvh * head_dim + lane;
            acc += scores[t] * qw36_load_scalar(v_cache, v_cache_dtype, off);
        }
        y[h * head_dim + lane] = acc;
    }
}

kernel void qw36_attn_decode_fused_f32(
    device float       *y            [[buffer(0)]],
    device const float *q_raw        [[buffer(1)]],
    device const float *k_raw        [[buffer(2)]],
    device const float *v_raw        [[buffer(3)]],
    device uchar       *k_cache      [[buffer(4)]],
    device uchar       *v_cache      [[buffer(5)]],
    device const uchar *q_norm       [[buffer(6)]],
    device const uchar *k_norm       [[buffer(7)]],
    constant uint      &n_heads      [[buffer(8)]],
    constant uint      &n_kv         [[buffer(9)]],
    constant uint      &head_dim     [[buffer(10)]],
    constant uint      &seq_pos      [[buffer(11)]],
    constant uint      &seq_capacity [[buffer(12)]],
    constant float     &theta        [[buffer(13)]],
    constant float     &partial      [[buffer(14)]],
    constant float     &eps          [[buffer(15)]],
    constant uint      &q_w_dtype    [[buffer(16)]],
    constant uint      &k_w_dtype    [[buffer(17)]],
    constant uint      &tg_size      [[buffer(18)]],
    constant uint      &k_cache_dtype [[buffer(19)]],
    constant uint      &v_cache_dtype [[buffer(20)]],
    constant uint      &q_has_gate    [[buffer(21)]],
    threadgroup float  *scratch      [[threadgroup(0)]],
    uint                lane         [[thread_index_in_threadgroup]],
    uint3               tg_pos       [[threadgroup_position_in_grid]])
{
    uint h = tg_pos.x;
    if (h >= n_heads || seq_pos >= seq_capacity) return;

    uint kvh = h * n_kv / n_heads;
    uint kv_len = n_kv * head_dim;
    uint count = seq_pos + 1;
    uint rot_dim = uint(float(head_dim) * partial);
    if (rot_dim > head_dim) rot_dim = head_dim;
    rot_dim &= ~1u;
    uint half_dim = rot_dim / 2;

    threadgroup float *partials = scratch;
    threadgroup float *scores = scratch + tg_size;
    uint simd_lane = lane & 31u;
    uint simd_id = lane >> 5;
    uint simd_count = (tg_size + 31u) >> 5;

    uint q_head_stride = q_has_gate ? (2u * head_dim) : head_dim;
    device const float *qh_base = q_raw + h * q_head_stride;
    device const float *gate_base = q_raw + h * 2u * head_dim + head_dim;
    device const float *kh_base = k_raw + kvh * head_dim;

    float q_in = (lane < head_dim) ? qh_base[lane] : 0.0f;
    float k_in = (lane < head_dim) ? kh_base[lane] : 0.0f;
    float q_ss = simd_sum((lane < head_dim) ? q_in * q_in : 0.0f);
    if (simd_lane == 0) partials[simd_id] = q_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    q_ss = simd_sum((lane < simd_count) ? partials[lane] : 0.0f);
    if (lane == 0) partials[0] = q_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_scale = rsqrt(partials[0] / float(head_dim) + eps);

    float k_ss = simd_sum((lane < head_dim) ? k_in * k_in : 0.0f);
    if (simd_lane == 0) partials[simd_id] = k_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    k_ss = simd_sum((lane < simd_count) ? partials[lane] : 0.0f);
    if (lane == 0) partials[0] = k_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float k_scale = rsqrt(partials[0] / float(head_dim) + eps);

    float qv = 0.0f;
    float kv = 0.0f;
    if (lane < head_dim) {
        qv = q_in * q_scale * qw36_load_scalar(q_norm, q_w_dtype, lane);
        kv = k_in * k_scale * qw36_load_scalar(k_norm, k_w_dtype, lane);
        if (lane < rot_dim) {
            bool upper = lane >= half_dim;
            uint pair = upper ? (lane - half_dim) : lane;
            uint d0 = pair;
            uint d1 = pair + half_dim;
            float q0 = qh_base[d0] * q_scale * qw36_load_scalar(q_norm, q_w_dtype, d0);
            float q1 = qh_base[d1] * q_scale * qw36_load_scalar(q_norm, q_w_dtype, d1);
            float k0 = kh_base[d0] * k_scale * qw36_load_scalar(k_norm, k_w_dtype, d0);
            float k1 = kh_base[d1] * k_scale * qw36_load_scalar(k_norm, k_w_dtype, d1);
            float inv_freq = 1.0f / pow(theta, (2.0f * float(pair)) / float(rot_dim));
            float angle = float(seq_pos) * inv_freq;
            float c = cos(angle);
            float s = sin(angle);
            qv = upper ? (q0 * s + q1 * c) : (q0 * c - q1 * s);
            kv = upper ? (k0 * s + k1 * c) : (k0 * c - k1 * s);
        }
    }

    if (lane < head_dim && h == kvh * n_heads / n_kv) {
        uint off = seq_pos * kv_len + kvh * head_dim + lane;
        qw36_store_scalar(k_cache, k_cache_dtype, off, kv);
        qw36_store_scalar(v_cache, v_cache_dtype, off, v_raw[kvh * head_dim + lane]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_sqrt_d = rsqrt(float(head_dim));
    for (uint t = 0; t < count; ++t) {
        float kval = 0.0f;
        if (lane < head_dim) {
            kval = (t == seq_pos)
                ? kv
                : qw36_load_scalar(k_cache, k_cache_dtype,
                                   t * kv_len + kvh * head_dim + lane);
        }
        float dot = simd_sum((lane < head_dim) ? qv * kval : 0.0f);
        if (simd_lane == 0) partials[simd_id] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dot = simd_sum((lane < simd_count) ? partials[lane] : 0.0f);
        if (lane == 0) scores[t] = dot * inv_sqrt_d;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        float maxv = scores[0];
        for (uint t = 1; t < count; ++t) maxv = max(maxv, scores[t]);
        float sum = 0.0f;
        for (uint t = 0; t < count; ++t) {
            float e = exp(scores[t] - maxv);
            scores[t] = e;
            sum += e;
        }
        float inv_sum = 1.0f / sum;
        for (uint t = 0; t < count; ++t) scores[t] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane < head_dim) {
        float acc = 0.0f;
        for (uint t = 0; t < count; ++t) {
            float vv = (t == seq_pos)
                ? v_raw[kvh * head_dim + lane]
                : qw36_load_scalar(v_cache, v_cache_dtype,
                                   t * kv_len + kvh * head_dim + lane);
            acc += scores[t] * vv;
        }
        if (q_has_gate) {
            float g = gate_base[lane];
            acc *= 1.0f / (1.0f + exp(-g));
        }
        y[h * head_dim + lane] = acc;
    }
}

kernel void qw36_attn_decode_fused_f16kv_f32(
    device float       *y            [[buffer(0)]],
    device const float *q_raw        [[buffer(1)]],
    device const float *k_raw        [[buffer(2)]],
    device const float *v_raw        [[buffer(3)]],
    device half        *k_cache      [[buffer(4)]],
    device half        *v_cache      [[buffer(5)]],
    device const uchar *q_norm       [[buffer(6)]],
    device const uchar *k_norm       [[buffer(7)]],
    constant uint      &n_heads      [[buffer(8)]],
    constant uint      &n_kv         [[buffer(9)]],
    constant uint      &head_dim     [[buffer(10)]],
    constant uint      &seq_pos      [[buffer(11)]],
    constant uint      &seq_capacity [[buffer(12)]],
    constant float     &theta        [[buffer(13)]],
    constant float     &partial      [[buffer(14)]],
    constant float     &eps          [[buffer(15)]],
    constant uint      &q_w_dtype    [[buffer(16)]],
    constant uint      &k_w_dtype    [[buffer(17)]],
    constant uint      &tg_size      [[buffer(18)]],
    constant uint      &q_has_gate   [[buffer(19)]],
    threadgroup float  *scratch      [[threadgroup(0)]],
    uint                lane         [[thread_index_in_threadgroup]],
    uint3               tg_pos       [[threadgroup_position_in_grid]])
{
    uint h = tg_pos.x;
    if (h >= n_heads || seq_pos >= seq_capacity) return;

    uint kvh = h * n_kv / n_heads;
    uint kv_len = n_kv * head_dim;
    uint count = seq_pos + 1;
    uint rot_dim = uint(float(head_dim) * partial);
    if (rot_dim > head_dim) rot_dim = head_dim;
    rot_dim &= ~1u;
    uint half_dim = rot_dim / 2;

    threadgroup float *partials = scratch;
    threadgroup float *scores = scratch + tg_size;
    uint simd_lane = lane & 31u;
    uint simd_id = lane >> 5;
    uint simd_count = (tg_size + 31u) >> 5;

    uint q_head_stride = q_has_gate ? (2u * head_dim) : head_dim;
    device const float *qh_base = q_raw + h * q_head_stride;
    device const float *gate_base = q_raw + h * 2u * head_dim + head_dim;
    device const float *kh_base = k_raw + kvh * head_dim;

    float q_in = (lane < head_dim) ? qh_base[lane] : 0.0f;
    float k_in = (lane < head_dim) ? kh_base[lane] : 0.0f;
    float q_ss = simd_sum((lane < head_dim) ? q_in * q_in : 0.0f);
    if (simd_lane == 0) partials[simd_id] = q_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    q_ss = simd_sum((lane < simd_count) ? partials[lane] : 0.0f);
    if (lane == 0) partials[0] = q_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_scale = rsqrt(partials[0] / float(head_dim) + eps);

    float k_ss = simd_sum((lane < head_dim) ? k_in * k_in : 0.0f);
    if (simd_lane == 0) partials[simd_id] = k_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    k_ss = simd_sum((lane < simd_count) ? partials[lane] : 0.0f);
    if (lane == 0) partials[0] = k_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float k_scale = rsqrt(partials[0] / float(head_dim) + eps);

    float qv = 0.0f;
    float kv = 0.0f;
    if (lane < head_dim) {
        qv = q_in * q_scale * qw36_load_scalar(q_norm, q_w_dtype, lane);
        kv = k_in * k_scale * qw36_load_scalar(k_norm, k_w_dtype, lane);
        if (lane < rot_dim) {
            bool upper = lane >= half_dim;
            uint pair = upper ? (lane - half_dim) : lane;
            uint d0 = pair;
            uint d1 = pair + half_dim;
            float q0 = qh_base[d0] * q_scale * qw36_load_scalar(q_norm, q_w_dtype, d0);
            float q1 = qh_base[d1] * q_scale * qw36_load_scalar(q_norm, q_w_dtype, d1);
            float k0 = kh_base[d0] * k_scale * qw36_load_scalar(k_norm, k_w_dtype, d0);
            float k1 = kh_base[d1] * k_scale * qw36_load_scalar(k_norm, k_w_dtype, d1);
            float inv_freq = 1.0f / pow(theta, (2.0f * float(pair)) / float(rot_dim));
            float angle = float(seq_pos) * inv_freq;
            float c = cos(angle);
            float s = sin(angle);
            qv = upper ? (q0 * s + q1 * c) : (q0 * c - q1 * s);
            kv = upper ? (k0 * s + k1 * c) : (k0 * c - k1 * s);
        }
    }

    if (lane < head_dim && h == kvh * n_heads / n_kv) {
        uint off = seq_pos * kv_len + kvh * head_dim + lane;
        k_cache[off] = half(kv);
        v_cache[off] = half(v_raw[kvh * head_dim + lane]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_sqrt_d = rsqrt(float(head_dim));
    for (uint t = 0; t < count; ++t) {
        float kval = 0.0f;
        if (lane < head_dim) {
            kval = (t == seq_pos)
                ? kv
                : float(k_cache[t * kv_len + kvh * head_dim + lane]);
        }
        float dot = simd_sum((lane < head_dim) ? qv * kval : 0.0f);
        if (simd_lane == 0) partials[simd_id] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dot = simd_sum((lane < simd_count) ? partials[lane] : 0.0f);
        if (lane == 0) scores[t] = dot * inv_sqrt_d;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        float maxv = scores[0];
        for (uint t = 1; t < count; ++t) maxv = max(maxv, scores[t]);
        float sum = 0.0f;
        for (uint t = 0; t < count; ++t) {
            float e = exp(scores[t] - maxv);
            scores[t] = e;
            sum += e;
        }
        float inv_sum = 1.0f / sum;
        for (uint t = 0; t < count; ++t) scores[t] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane < head_dim) {
        float acc = 0.0f;
        for (uint t = 0; t < count; ++t) {
            float vv = (t == seq_pos)
                ? v_raw[kvh * head_dim + lane]
                : float(v_cache[t * kv_len + kvh * head_dim + lane]);
            acc += scores[t] * vv;
        }
        if (q_has_gate) {
            float g = gate_base[lane];
            acc *= 1.0f / (1.0f + exp(-g));
        }
        y[h * head_dim + lane] = acc;
    }
}

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

/* =============================================================================
 * GGUF-native quantized matmul. Ported from agent-infer's
 *   crates/mlx-sys/src/mlx_bridge.cpp:620-743, 748-812
 * Operates directly on packed Q4_K/Q5_K/Q6_K/Q8_0/Q3_K row bytes — no fp16
 * materialise step, no MPS-GEMV round-trip. One threadgroup per output row,
 * 256 threads collaborate via threadgroup reduction.
 * ========================================================================== */

static inline float qw36_gguf_f16(device const uchar *p)
{
    ushort bits = ushort(p[0]) | (ushort(p[1]) << 8);
    return float(as_type<half>(bits));
}

static inline int qw36_gguf_i8(uchar v)
{
    int x = int(v);
    return x >= 128 ? x - 256 : x;
}

static inline void qw36_gguf_q4_scales(device const uchar *s,
                                       thread uchar sc[8], thread uchar mn[8])
{
    for (int i = 0; i < 4; ++i) {
        sc[i] = s[i] & 0x3f;
        mn[i] = s[i + 4] & 0x3f;
    }
    for (int i = 0; i < 4; ++i) {
        sc[4 + i] = (s[8 + i] & 0x0f) | ((s[i] >> 6) << 4);
        mn[4 + i] = (s[8 + i] >> 4) | ((s[i + 4] >> 6) << 4);
    }
}

static inline float qw36_gguf_q4_k_value(device const uchar *row, int k)
{
    int sb = k >> 8;
    int local = k & 255;
    device const uchar *p = row + sb * 144;
    float d = qw36_gguf_f16(p);
    float dmin = qw36_gguf_f16(p + 2);
    uchar sc[8], mn[8];
    qw36_gguf_q4_scales(p + 4, sc, mn);
    int iter = local >> 6;
    int h = (local >> 5) & 1;
    int lane = local & 31;
    int sub = iter * 2 + h;
    uchar byte = p[16 + iter * 32 + lane];
    float q = h == 0 ? float(byte & 0x0f) : float(byte >> 4);
    return q * (d * float(sc[sub])) - dmin * float(mn[sub]);
}

static inline float qw36_gguf_q6_k_value(device const uchar *row, int k)
{
    int sb = k >> 8;
    int local = k & 255;
    device const uchar *p = row + sb * 210;
    device const uchar *ql_all = p;
    device const uchar *qh_all = p + 128;
    device const uchar *scales_all = p + 192;
    float d = qw36_gguf_f16(p + 208);
    int h = local >> 7;
    int rem = local & 127;
    int lane = rem & 31;
    device const uchar *ql = ql_all + h * 64;
    device const uchar *qh = qh_all + h * 32;
    device const uchar *sc = scales_all + h * 8;
    int q;
    int scale_idx;
    if (rem < 32) {
        q = int((ql[lane] & 0x0f) | ((qh[lane] & 0x03) << 4)) - 32;
        scale_idx = lane / 16;
    } else if (rem < 64) {
        q = int((ql[lane + 32] & 0x0f) | (((qh[lane] >> 2) & 0x03) << 4)) - 32;
        scale_idx = lane / 16 + 2;
    } else if (rem < 96) {
        q = int((ql[lane] >> 4) | (((qh[lane] >> 4) & 0x03) << 4)) - 32;
        scale_idx = lane / 16 + 4;
    } else {
        q = int((ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 0x03) << 4)) - 32;
        scale_idx = lane / 16 + 6;
    }
    return d * float(qw36_gguf_i8(sc[scale_idx])) * float(q);
}

static inline float qw36_gguf_q5_k_value(device const uchar *row, int k)
{
    int sb = k >> 8;
    int local = k & 255;
    device const uchar *p = row + sb * 176;
    float d = qw36_gguf_f16(p);
    float dmin = qw36_gguf_f16(p + 2);
    uchar sc[8], mn[8];
    qw36_gguf_q4_scales(p + 4, sc, mn);
    device const uchar *qh = p + 16;
    device const uchar *qs = p + 48;
    int iter = local >> 6;
    int h = (local >> 5) & 1;
    int lane = local & 31;
    int sub = iter * 2 + h;
    uchar byte = qs[iter * 32 + lane];
    int nib = h == 0 ? int(byte & 0x0f) : int(byte >> 4);
    int hi = (int(qh[lane]) >> sub) & 1;
    float q = float(nib | (hi << 4));
    return q * (d * float(sc[sub])) - dmin * float(mn[sub]);
}

static inline float qw36_gguf_q8_0_value(device const uchar *row, int k)
{
    int block = k >> 5;
    int lane = k & 31;
    device const uchar *p = row + block * 34;
    return qw36_gguf_f16(p) * float(qw36_gguf_i8(p[2 + lane]));
}

/* Q4_K matmul (decode, M=1, output row n)
 * Each threadgroup handles one output row. We first cache d, dmin and the
 * 8 per-sub-block scale/min entries for every Q4_K block on this row into
 * threadgroup memory (max 16 blocks → 192 bytes), then each thread does a
 * strided per-element walk reading only the cached constants + a single
 * quant byte. Saves the redundant fp16 + 6-bit unpack each thread used to
 * do per element. */
kernel void qw36_matmul_q4_k_f32(
    device float       *y    [[buffer(0)]],
    device const float *x    [[buffer(1)]],
    device const uchar *w    [[buffer(2)]],
    constant uint      &K    [[buffer(3)]],
    constant uint      &N    [[buffer(4)]],
    threadgroup float  *scratch [[threadgroup(0)]],
    uint tid                 [[thread_position_in_threadgroup]],
    uint n                   [[threadgroup_position_in_grid]])
{
    if (n >= N) return;
    const uint TG = 256u;
    uint K_blocks = K >> 8;            /* num Q4_K blocks per row */
    uint row_bytes = K_blocks * 144u;
    device const uchar *row = w + n * row_bytes;

    /* Layout in scratch:
     *   scratch[0..K_blocks-1]            : d
     *   scratch[K_blocks..2*K_blocks-1]   : dmin
     *   scratch[2*K_blocks..18*K_blocks-1]: sc/mn (8 sc + 8 mn floats per block, packed)
     * For K=3584 → K_blocks=14, total <= 14*(2+16)=252 floats = 1 KiB. */
    threadgroup float *d_cache    = scratch;
    threadgroup float *dmin_cache = scratch + K_blocks;
    threadgroup float *sc_cache   = scratch + 2u * K_blocks;       /* [K_blocks][8] */
    threadgroup float *mn_cache   = scratch + 2u * K_blocks + 8u * K_blocks;
    if (tid < K_blocks) {
        device const uchar *p = row + tid * 144u;
        d_cache[tid] = qw36_gguf_f16(p);
        dmin_cache[tid] = qw36_gguf_f16(p + 2);
        uchar sc[8], mn[8];
        qw36_gguf_q4_scales(p + 4, sc, mn);
        for (uint s = 0; s < 8u; s++) {
            sc_cache[tid * 8u + s] = float(sc[s]);
            mn_cache[tid * 8u + s] = float(mn[s]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum = 0.0f;
    for (uint k = tid; k < K; k += TG) {
        uint sb = k >> 8;
        uint local = k & 255u;
        uint iter = local >> 6;
        uint h = (local >> 5) & 1u;
        uint lane = local & 31u;
        uint sub = iter * 2u + h;
        device const uchar *p = row + sb * 144u;
        uchar byte = p[16u + iter * 32u + lane];
        float q = (h == 0u) ? float(byte & 0x0fu) : float(byte >> 4);
        float d = d_cache[sb];
        float dmn = dmin_cache[sb];
        float wval = q * (d * sc_cache[sb * 8u + sub])
                   - dmn * mn_cache[sb * 8u + sub];
        sum += x[k] * wval;
    }

    float simd_v = simd_sum(sum);
    uint simd_lane = tid & 31u;
    uint simd_id = tid >> 5;
    /* Reuse the tail of scratch (past the cache) for per-SIMD partials. */
    threadgroup float *part = scratch + 18u * K_blocks;
    if (simd_lane == 0u) part[simd_id] = simd_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < (TG >> 5)) {
        float v = part[tid];
        v = simd_sum(v);
        if (tid == 0u) y[n] = v;
    }
}

kernel void qw36_matmul_q6_k_f32(
    device float       *y    [[buffer(0)]],
    device const float *x    [[buffer(1)]],
    device const uchar *w    [[buffer(2)]],
    constant uint      &K    [[buffer(3)]],
    constant uint      &N    [[buffer(4)]],
    threadgroup float  *partial [[threadgroup(0)]],
    uint tid                 [[thread_position_in_threadgroup]],
    uint n                   [[threadgroup_position_in_grid]])
{
    if (n >= N) return;
    const uint TG = 256u;
    uint row_bytes = (K / 256u) * 210u;
    device const uchar *row = w + n * row_bytes;
    float sum = 0.0f;
    for (uint k = tid; k < K; k += TG) {
        sum += x[k] * qw36_gguf_q6_k_value(row, int(k));
    }
    /* simd_sum-based reduction: collapse each 32-wide SIMD group with one
     * intrinsic, then reduce the 8 partials in a single SIMD step. */
    float simd_v = simd_sum(sum);
    uint simd_lane = tid & 31u;
    uint simd_id = tid >> 5;
    if (simd_lane == 0u) partial[simd_id] = simd_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < (TG >> 5)) {
        float v = partial[tid];
        v = simd_sum(v);
        if (tid == 0u) y[n] = v;
    }
}

kernel void qw36_matmul_q5_k_f32(
    device float       *y    [[buffer(0)]],
    device const float *x    [[buffer(1)]],
    device const uchar *w    [[buffer(2)]],
    constant uint      &K    [[buffer(3)]],
    constant uint      &N    [[buffer(4)]],
    threadgroup float  *partial [[threadgroup(0)]],
    uint tid                 [[thread_position_in_threadgroup]],
    uint n                   [[threadgroup_position_in_grid]])
{
    if (n >= N) return;
    const uint TG = 256u;
    uint row_bytes = (K / 256u) * 176u;
    device const uchar *row = w + n * row_bytes;
    float sum = 0.0f;
    for (uint k = tid; k < K; k += TG) {
        sum += x[k] * qw36_gguf_q5_k_value(row, int(k));
    }
    /* simd_sum-based reduction: collapse each 32-wide SIMD group with one
     * intrinsic, then reduce the 8 partials in a single SIMD step. */
    float simd_v = simd_sum(sum);
    uint simd_lane = tid & 31u;
    uint simd_id = tid >> 5;
    if (simd_lane == 0u) partial[simd_id] = simd_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < (TG >> 5)) {
        float v = partial[tid];
        v = simd_sum(v);
        if (tid == 0u) y[n] = v;
    }
}

kernel void qw36_matmul_q8_0_f32(
    device float       *y    [[buffer(0)]],
    device const float *x    [[buffer(1)]],
    device const uchar *w    [[buffer(2)]],
    constant uint      &K    [[buffer(3)]],
    constant uint      &N    [[buffer(4)]],
    threadgroup float  *partial [[threadgroup(0)]],
    uint tid                 [[thread_position_in_threadgroup]],
    uint n                   [[threadgroup_position_in_grid]])
{
    if (n >= N) return;
    const uint TG = 256u;
    uint row_bytes = (K / 32u) * 34u;
    device const uchar *row = w + n * row_bytes;
    float sum = 0.0f;
    for (uint k = tid; k < K; k += TG) {
        sum += x[k] * qw36_gguf_q8_0_value(row, int(k));
    }
    /* simd_sum-based reduction: collapse each 32-wide SIMD group with one
     * intrinsic, then reduce the 8 partials in a single SIMD step. */
    float simd_v = simd_sum(sum);
    uint simd_lane = tid & 31u;
    uint simd_id = tid >> 5;
    if (simd_lane == 0u) partial[simd_id] = simd_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < (TG >> 5)) {
        float v = partial[tid];
        v = simd_sum(v);
        if (tid == 0u) y[n] = v;
    }
}
