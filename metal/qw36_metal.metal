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
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint QW36_DTYPE_F32  = 0;
constant uint QW36_DTYPE_F16  = 1;
constant uint QW36_DTYPE_Q8_0 = 8;
constant uint QW36_DTYPE_Q4_K = 12;
constant uint QW36_DTYPE_Q5_K = 13;
constant uint QW36_DTYPE_Q6_K = 14;
constant uint QW36_DTYPE_BF16 = 30;
constant uint QW36_DTYPE_Q4K_AFFINE32 = 100;
constant uint QW36_DTYPE_Q5K_AFFINE32 = 101;
constant uint QW36_DTYPE_Q6K_SCALE16  = 102;

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
    } else if (dtype == QW36_DTYPE_Q8_0) {
        /* 34-byte blocks of 32 elements: fp16 d + int8 qs[32]. */
        uint block = i >> 5u;          /* i / 32 */
        uint local = i & 31u;
        device const uchar *b = ptr + block * 34u;
        half d = ((device const half *)b)[0];
        device const char *qs = (device const char *)(b + 2);
        int q = int(qs[local]);
        return float(d) * float(q);
    } else if (dtype == QW36_DTYPE_Q4K_AFFINE32) {
        /* 160-byte blocks of 256 elements:
         *   half scale[8] | half bias[8] | uint8 packed_q[8][16]
         * Each 32-element sub-block has one (scale,bias). 4-bit values
         * pack 2 per byte; even/odd low/high nibble. */
        uint block = i >> 8u;
        uint sub   = (i >> 5u) & 7u;        /* 0..7 */
        uint local = i & 31u;               /* 0..31 within sub-block */
        device const uchar *b = ptr + block * 160u;
        half scale = ((device const half *)b)[sub];
        half bias  = ((device const half *)b)[8u + sub];
        uchar packed = b[32u + sub * 16u + (local >> 1u)];
        uint nib = (local & 1u) ? (uint(packed) >> 4) : (uint(packed) & 0x0fu);
        return float(nib) * float(scale) + float(bias);
    } else if (dtype == QW36_DTYPE_Q5K_AFFINE32) {
        /* 192-byte blocks of 256 elements:
         *   half scale[8] | half bias[8] | uint8 packed_q[8][20]
         * Each sub-block stores 32 5-bit values in two 5-byte packs of 8. */
        uint block = i >> 8u;
        uint sub   = (i >> 5u) & 7u;
        uint local = i & 31u;
        device const uchar *b = ptr + block * 192u;
        half scale = ((device const half *)b)[sub];
        half bias  = ((device const half *)b)[8u + sub];
        device const uchar *p = b + 32u + sub * 20u + (local >> 3u) * 5u;
        uint idx = local & 7u;
        ulong bits = ulong(p[0]) |
                     (ulong(p[1]) << 8) |
                     (ulong(p[2]) << 16) |
                     (ulong(p[3]) << 24) |
                     (ulong(p[4]) << 32);
        uint q = uint((bits >> (5u * idx)) & 31ul);
        return float(q) * float(scale) + float(bias);
    } else if (dtype == QW36_DTYPE_Q6K_SCALE16) {
        /* 224-byte blocks of 256 elements:
         *   half scale[16] | uint8 packed_q[16][12]
         * Per-16 scale, value = (q - 32) * scale. */
        uint block = i >> 8u;
        uint sub   = (i >> 4u) & 15u;
        uint local = i & 15u;
        device const uchar *b = ptr + block * 224u;
        half scale = ((device const half *)b)[sub];
        device const uchar *p = b + 32u + sub * 12u;
        uint bit = 6u * local;
        uint byte = bit >> 3;
        uint shift = bit & 7u;
        uint bits = uint(p[byte]);
        if (byte + 1u < 12u) bits |= uint(p[byte + 1u]) << 8;
        uint q = (bits >> shift) & 63u;
        return (float(q) - 32.0f) * float(scale);
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

static inline uint qw36_kv_cache_offset(uint t, uint kvh, uint lane,
                                        uint head_dim, uint seq_capacity,
                                        uint kv_len, uint kv_transposed)
{
    uint head_lane = kvh * head_dim + lane;
    return kv_transposed ? head_lane * seq_capacity + t
                         : t * kv_len + head_lane;
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
    device uchar       *gate_io  [[buffer(0)]], /* in: gate, out: silu(gate)*up */
    device const uchar *up       [[buffer(1)]],
    constant uint      &n        [[buffer(2)]],
    constant uint      &gate_dtype [[buffer(3)]],
    constant uint      &up_dtype [[buffer(4)]],
    constant uint      &gate_offset [[buffer(5)]],
    constant uint      &up_offset [[buffer(6)]],
    uint                tid      [[thread_position_in_grid]])
{
    if (tid >= n) return;
    uint gi = gate_offset + tid;
    uint ui = up_offset + tid;
    float g = qw36_load_scalar(gate_io, gate_dtype, gi);
    float u = qw36_load_scalar(up, up_dtype, ui);
    qw36_store_scalar(gate_io, gate_dtype, gi,
                      (g / (1.0f + exp(-g))) * u);
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

kernel void qw36_moe_scale_add_f32(
    device uchar       *y          [[buffer(0)]],
    device const float *shared     [[buffer(1)]],
    device const float *scale_raw  [[buffer(2)]],
    constant uint      &hidden     [[buffer(3)]],
    constant uint      &has_scale  [[buffer(4)]],
    constant uint      &y_dtype    [[buffer(5)]],
    uint                tid        [[thread_position_in_grid]])
{
    if (tid >= hidden) return;
    float scale = has_scale ? (1.0f / (1.0f + exp(-scale_raw[0]))) : 1.0f;
    float v = qw36_load_scalar(y, y_dtype, tid) + scale * shared[tid];
    qw36_store_scalar(y, y_dtype, tid, v);
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
    uint group = (n_key && n_value % n_key == 0) ? n_value / n_key : 1;
    if (group == 0) group = 1;
    uint kh = (n_value >= n_key && n_value % n_key == 0) ? v / group : v % n_key;
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
    float q_scale = rsqrt(qss + float(key_dim) * 1.0e-6f) / sqrt(float(key_dim));
    float k_scale = rsqrt(kss + float(key_dim) * 1.0e-6f);

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

kernel void qw36_dn_gated_rmsnorm_f16_f32(
    device half        *y        [[buffer(0)]],
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
    y[gid] = half(gate * x[gid] * scale * w[d]);
}

kernel void qw36_dn_gated_rmsnorm_matmul_f32(
    device uchar       *y        [[buffer(0)]],
    device const float *x        [[buffer(1)]],
    device const float *z        [[buffer(2)]],
    device const float *norm_w   [[buffer(3)]],
    device const uchar *out_w    [[buffer(4)]],
    constant uint      &n_value  [[buffer(5)]],
    constant uint      &val_dim  [[buffer(6)]],
    constant uint      &out_rows [[buffer(7)]],
    constant uint      &in_cols  [[buffer(8)]],
    constant float     &eps      [[buffer(9)]],
    constant uint      &w_dtype  [[buffer(10)]],
    constant uint      &y_dtype  [[buffer(11)]],
    constant uint      &tg_size  [[buffer(12)]],
    threadgroup float  *scratch  [[threadgroup(0)]],
    uint                tid      [[thread_index_in_threadgroup]],
    uint3               tg       [[threadgroup_position_in_grid]])
{
    uint row = tg.x;
    if (row >= out_rows || n_value == 0 || val_dim == 0 ||
        in_cols != n_value * val_dim)
        return;

    threadgroup float *scale = scratch;
    threadgroup float *partial = scratch + n_value;

    for (uint h = tid; h < n_value; h += tg_size) {
        device const float *xh = x + h * val_dim;
        float ss = 0.0f;
        for (uint d = 0; d < val_dim; ++d) ss += xh[d] * xh[d];
        scale[h] = rsqrt(ss / float(val_dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    for (uint k = tid; k < in_cols; k += tg_size) {
        uint h = k / val_dim;
        uint d = k - h * val_dim;
        float zg = z[k];
        float gate = zg / (1.0f + exp(-zg));
        float xv = gate * x[k] * scale[h] * norm_w[d];
        float wv = qw36_load_scalar(out_w, w_dtype, row * in_cols + k);
        acc += xv * wv;
    }

    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) qw36_store_scalar(y, y_dtype, row, partial[0]);
}

static inline uint qw36_dn_raw_v_head(uint grouped_v, uint Hk, uint Hv)
{
    uint group = (Hk == 0) ? 1 : (Hv / Hk);
    if (group == 0) group = 1;
    if (Hk != 0 && Hv % Hk == 0)
        return (grouped_v % group) * Hk + grouped_v / group;
    return grouped_v;
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
        q_out[gid] = qh[d] * rsqrt(qss + float(Dk) * 1.0e-6f) / sqrt(float(Dk));
        k_out[gid] = kh[d] * rsqrt(kss + float(Dk) * 1.0e-6f);
    }

    if (gid < v_total) {
        uint v = gid / Dv;
        uint d = gid - v * Dv;
        uint raw_v = qw36_dn_raw_v_head(v, Hk, Hv);
        v_out[v * Dv + d] = qkv[q_total + k_total + raw_v * Dv + d];
    }

    if (gid < Hv) {
        uint v = gid;
        uint raw_v = qw36_dn_raw_v_head(v, Hk, Hv);
        float av = alpha_raw[raw_v] + dt_bias[v];
        float softplus = av > 20.0f ? av : log(1.0f + exp(av));
        g_out[v] = exp(-exp(a_log[v]) * softplus);
        beta_out[v] = 1.0f / (1.0f + exp(-beta_raw[raw_v]));
    }
}

static inline float qw36_dn_conv1d_silu_value_at(device const float *x,
                                                  device const float *conv_w,
                                                  device const float *conv_state,
                                                  uint channels,
                                                  uint kernel_size,
                                                  uint c_state,
                                                  uint c_x)
{
    float acc = 0.0f;
    if (kernel_size == 0) {
        acc = x[c_x];
    } else {
        device const float *wt = conv_w + c_state * kernel_size;
        for (uint t = 0; t + 1 < kernel_size; ++t) {
            acc += wt[t] * conv_state[t * channels + c_state];
        }
        acc += wt[kernel_size - 1] * x[c_x];
    }
    return acc / (1.0f + exp(-acc));
}

static inline float qw36_dn_conv1d_silu_value(device const float *x,
                                               device const float *conv_w,
                                               device const float *conv_state,
                                               uint channels,
                                               uint kernel_size,
                                               uint c)
{
    return qw36_dn_conv1d_silu_value_at(x, conv_w, conv_state,
                                        channels, kernel_size, c, c);
}

static inline void qw36_dn_conv1d_update_state_at(device float *conv_state,
                                                   device const float *x,
                                                   uint channels,
                                                   uint kernel_size,
                                                   uint c_state,
                                                   uint c_x)
{
    if (kernel_size <= 1) return;
    for (uint t = 0; t + 2 < kernel_size; ++t) {
        conv_state[t * channels + c_state] =
            conv_state[(t + 1) * channels + c_state];
    }
    conv_state[(kernel_size - 2) * channels + c_state] = x[c_x];
}

static inline void qw36_dn_conv1d_update_state(device float *conv_state,
                                                device const float *x,
                                                uint channels,
                                                uint kernel_size,
                                                uint c)
{
    qw36_dn_conv1d_update_state_at(conv_state, x, channels,
                                   kernel_size, c, c);
}

/* Fused DeltaNet prep: inline depthwise conv1d+silu before q/k L2 norm,
 * v copy, alpha/beta transform, and in-place conv_state shift+append.
 * One threadgroup owns one key/value head, so state updates occur only
 * after all reads for that head have finished. */
kernel void qw36_compute_g_beta_norm_qk_conv1d(
    device float       *q_out      [[buffer(0)]],
    device float       *k_out      [[buffer(1)]],
    device float       *v_out      [[buffer(2)]],
    device float       *g_out      [[buffer(3)]],
    device float       *beta_out   [[buffer(4)]],
    device const float *qkv_raw    [[buffer(5)]],
    device const float *alpha_raw  [[buffer(6)]],
    device const float *beta_raw   [[buffer(7)]],
    device const float *dt_bias    [[buffer(8)]],
    device const float *a_log      [[buffer(9)]],
    device const float *conv_w     [[buffer(10)]],
    device float       *conv_state [[buffer(11)]],
    constant uint      &Hk         [[buffer(12)]],
    constant uint      &Hv         [[buffer(13)]],
    constant uint      &Dk         [[buffer(14)]],
    constant uint      &Dv         [[buffer(15)]],
    constant uint      &kernel_size [[buffer(16)]],
    constant uint      &channels   [[buffer(17)]],
    constant uint      &tg_size    [[buffer(18)]],
    threadgroup float  *scratch    [[threadgroup(0)]],
    uint                tid        [[thread_index_in_threadgroup]],
    uint3               tg         [[threadgroup_position_in_grid]])
{
    uint head = tg.y;
    uint q_total = Hk * Dk;
    uint k_total = q_total;

    threadgroup float *q_part = scratch;
    threadgroup float *k_part = scratch + tg_size;

    if (head < Hk) {
        float qss = 0.0f;
        float kss = 0.0f;
        for (uint d = tid; d < Dk; d += tg_size) {
            uint qc = head * Dk + d;
            uint kc = q_total + head * Dk + d;
            float qv = qw36_dn_conv1d_silu_value(qkv_raw, conv_w,
                                                  conv_state, channels,
                                                  kernel_size, qc);
            float kv = qw36_dn_conv1d_silu_value(qkv_raw, conv_w,
                                                  conv_state, channels,
                                                  kernel_size, kc);
            q_out[qc] = qv;
            k_out[qc] = kv;
            qss += qv * qv;
            kss += kv * kv;
        }
        q_part[tid] = qss;
        k_part[tid] = kss;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = tg_size >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                q_part[tid] += q_part[tid + stride];
                k_part[tid] += k_part[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float q_scale = rsqrt(q_part[0] + float(Dk) * 1.0e-6f) / sqrt(float(Dk));
        float k_scale = rsqrt(k_part[0] + float(Dk) * 1.0e-6f);
        for (uint d = tid; d < Dk; d += tg_size) {
            uint qc = head * Dk + d;
            uint kc = q_total + head * Dk + d;
            q_out[qc] *= q_scale;
            k_out[qc] *= k_scale;
            qw36_dn_conv1d_update_state(conv_state, qkv_raw, channels,
                                         kernel_size, qc);
            qw36_dn_conv1d_update_state(conv_state, qkv_raw, channels,
                                         kernel_size, kc);
        }
    }

    if (head < Hv) {
        uint raw_head = qw36_dn_raw_v_head(head, Hk, Hv);
        for (uint d = tid; d < Dv; d += tg_size) {
            uint vc = q_total + k_total + head * Dv + d;
            uint raw_vc = q_total + k_total + raw_head * Dv + d;
            v_out[head * Dv + d] = qw36_dn_conv1d_silu_value_at(
                qkv_raw, conv_w, conv_state, channels, kernel_size,
                vc, raw_vc);
            qw36_dn_conv1d_update_state_at(conv_state, qkv_raw, channels,
                                            kernel_size, vc, raw_vc);
        }
        if (tid == 0) {
            float av = alpha_raw[raw_head] + dt_bias[head];
            float softplus = av > 20.0f ? av : log(1.0f + exp(av));
            g_out[head] = exp(-exp(a_log[head]) * softplus);
            beta_out[head] = 1.0f / (1.0f + exp(-beta_raw[raw_head]));
        }
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
    device uchar       *x        [[buffer(0)]],
    device const uchar *y        [[buffer(1)]],
    constant uint      &n        [[buffer(2)]],
    constant uint      &x_dtype  [[buffer(3)]],
    constant uint      &y_dtype  [[buffer(4)]],
    uint                tid      [[thread_position_in_grid]])
{
    if (tid >= n) return;
    float a = qw36_load_scalar(x, x_dtype, tid);
    float b = qw36_load_scalar(y, y_dtype, tid);
    qw36_store_scalar(x, x_dtype, tid, a + b);
}

/* ---------------------------------------------------------------- */
/* Embedding lookup                                                  */
/* ---------------------------------------------------------------- */
kernel void qw36_embedding_lookup_f32(
    device uchar       *y        [[buffer(0)]],
    device const uchar *embed    [[buffer(1)]],
    constant uint      &token    [[buffer(2)]],
    constant uint      &hidden   [[buffer(3)]],
    constant uint      &dtype    [[buffer(4)]],
    constant uint      &y_dtype  [[buffer(5)]],
    uint                tid      [[thread_position_in_grid]])
{
    if (tid < hidden)
        qw36_store_scalar(y, y_dtype, tid,
            qw36_load_scalar(embed, dtype, token * hidden + tid));
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
    constant uint      &head_dim     [[buffer(9)]],
    constant uint      &kv_transposed [[buffer(10)]],
    uint                tid          [[thread_position_in_grid]])
{
    if (tid >= kv_len || seq_pos >= seq_capacity) return;
    uint kvh = tid / head_dim;
    uint lane = tid - kvh * head_dim;
    uint off = qw36_kv_cache_offset(seq_pos, kvh, lane, head_dim,
                                    seq_capacity, kv_len, kv_transposed);
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
    constant uint      &kv_transposed [[buffer(9)]],
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
    float acc = 0.0f;
    for (uint d = 0; d < head_dim; ++d) {
        uint off = qw36_kv_cache_offset(pos, kv_h, d, head_dim,
                                        seq_capacity, kv_len, kv_transposed);
        acc += qh[d] * qw36_load_scalar(k_cache, k_cache_dtype, off);
    }
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
    constant uint      &kv_transposed [[buffer(9)]],
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
        uint off = qw36_kv_cache_offset(pos, kv_h, d, head_dim,
                                        seq_capacity, kv_len, kv_transposed);
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
    constant uint      &kv_transposed [[buffer(19)]],
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
    device const float *vh = v + kvh * head_dim;
    for (uint d = 0; d < head_dim; ++d) {
        uint off = qw36_kv_cache_offset(seq_pos, kvh, d, head_dim,
                                        seq_capacity, kv_len, kv_transposed);
        qw36_store_scalar(k_cache, k_cache_dtype, off, kh[d]);
        qw36_store_scalar(v_cache, v_cache_dtype, off, vh[d]);
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
    constant uint      &kv_transposed [[buffer(12)]],
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
        partials[lane] = (lane < head_dim)
            ? qh[lane] * qw36_load_scalar(
                k_cache, k_cache_dtype,
                qw36_kv_cache_offset(t, kvh, lane, head_dim,
                                     seq_capacity, kv_len, kv_transposed))
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
            uint off = qw36_kv_cache_offset(t, kvh, lane, head_dim,
                                            seq_capacity, kv_len,
                                            kv_transposed);
            acc += scores[t] * qw36_load_scalar(v_cache, v_cache_dtype, off);
        }
        y[h * head_dim + lane] = acc;
    }
}

kernel void qw36_attn_decode_fused_f32(
    device uchar       *y            [[buffer(0)]],
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
    constant uint      &y_dtype       [[buffer(22)]],
    constant uint      &q_offset      [[buffer(23)]],
    constant uint      &k_offset      [[buffer(24)]],
    constant uint      &v_offset      [[buffer(25)]],
    constant uint      &kv_transposed [[buffer(26)]],
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

    device const float *q_base = q_raw + q_offset;
    device const float *k_base = k_raw + k_offset;
    device const float *v_base = v_raw + v_offset;
    uint q_head_stride = q_has_gate ? (2u * head_dim) : head_dim;
    device const float *qh_base = q_base + h * q_head_stride;
    device const float *gate_base = q_base + h * 2u * head_dim + head_dim;
    device const float *kh_base = k_base + kvh * head_dim;

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
        uint off = qw36_kv_cache_offset(seq_pos, kvh, lane, head_dim,
                                        seq_capacity, kv_len, kv_transposed);
        qw36_store_scalar(k_cache, k_cache_dtype, off, kv);
        qw36_store_scalar(v_cache, v_cache_dtype, off, v_base[kvh * head_dim + lane]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_sqrt_d = rsqrt(float(head_dim));
    for (uint t = 0; t < count; ++t) {
        float kval = 0.0f;
        if (lane < head_dim) {
            kval = (t == seq_pos)
                ? kv
                : qw36_load_scalar(k_cache, k_cache_dtype,
                                   qw36_kv_cache_offset(t, kvh, lane, head_dim,
                                                        seq_capacity, kv_len,
                                                        kv_transposed));
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
                ? v_base[kvh * head_dim + lane]
                : qw36_load_scalar(v_cache, v_cache_dtype,
                                   qw36_kv_cache_offset(t, kvh, lane, head_dim,
                                                        seq_capacity, kv_len,
                                                        kv_transposed));
            acc += scores[t] * vv;
        }
        if (q_has_gate) {
            float g = gate_base[lane];
            acc *= 1.0f / (1.0f + exp(-g));
        }
        qw36_store_scalar(y, y_dtype, h * head_dim + lane, acc);
    }
}

kernel void qw36_attn_decode_fused_f16kv_f32(
    device uchar       *y            [[buffer(0)]],
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
    constant uint      &y_dtype      [[buffer(20)]],
    constant uint      &q_offset     [[buffer(21)]],
    constant uint      &k_offset     [[buffer(22)]],
    constant uint      &v_offset     [[buffer(23)]],
    constant uint      &kv_transposed [[buffer(24)]],
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
    threadgroup float *q_vec = scratch;
    threadgroup float *k_vec = scratch + tg_size;
    threadgroup float *scores = scratch + (kv_transposed ? 2u * tg_size : tg_size);
    uint simd_lane = lane & 31u;
    uint simd_id = lane >> 5;
    uint simd_count = (tg_size + 31u) >> 5;

    device const float *q_base = q_raw + q_offset;
    device const float *k_base = k_raw + k_offset;
    device const float *v_base = v_raw + v_offset;
    uint q_head_stride = q_has_gate ? (2u * head_dim) : head_dim;
    device const float *qh_base = q_base + h * q_head_stride;
    device const float *gate_base = q_base + h * 2u * head_dim + head_dim;
    device const float *kh_base = k_base + kvh * head_dim;

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
        uint off = qw36_kv_cache_offset(seq_pos, kvh, lane, head_dim,
                                        seq_capacity, kv_len, kv_transposed);
        k_cache[off] = half(kv);
        v_cache[off] = half(v_base[kvh * head_dim + lane]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_sqrt_d = rsqrt(float(head_dim));
    if (kv_transposed) {
        if (lane < head_dim) {
            q_vec[lane] = qv;
            k_vec[lane] = kv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint t0 = 0; t0 < count; t0 += tg_size) {
            uint t = t0 + lane;
            if (t < count) {
                float dot = 0.0f;
                for (uint d = 0; d < head_dim; ++d) {
                    float kval = (t == seq_pos)
                        ? k_vec[d]
                        : float(k_cache[qw36_kv_cache_offset(
                            t, kvh, d, head_dim, seq_capacity, kv_len, 1u)]);
                    dot += q_vec[d] * kval;
                }
                scores[t] = dot * inv_sqrt_d;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

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
                    ? v_base[kvh * head_dim + lane]
                    : float(v_cache[qw36_kv_cache_offset(
                        t, kvh, lane, head_dim, seq_capacity, kv_len, 1u)]);
                acc += scores[t] * vv;
            }
            if (q_has_gate) {
                float g = gate_base[lane];
                acc *= 1.0f / (1.0f + exp(-g));
            }
            qw36_store_scalar(y, y_dtype, h * head_dim + lane, acc);
        }
        return;
    }

    for (uint t = 0; t < count; ++t) {
        float kval = 0.0f;
        if (lane < head_dim) {
            kval = (t == seq_pos)
                ? kv
                : float(k_cache[qw36_kv_cache_offset(
                    t, kvh, lane, head_dim, seq_capacity, kv_len, 0u)]);
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
                ? v_base[kvh * head_dim + lane]
                : float(v_cache[qw36_kv_cache_offset(
                    t, kvh, lane, head_dim, seq_capacity, kv_len, 0u)]);
            acc += scores[t] * vv;
        }
        if (q_has_gate) {
            float g = gate_base[lane];
            acc *= 1.0f / (1.0f + exp(-g));
        }
        qw36_store_scalar(y, y_dtype, h * head_dim + lane, acc);
    }
}

/* x4 variant of qw36_attn_decode_fused_f16kv_f32. Score loop processes
 * 4 t positions per outer iteration: each lane computes 4 partial dot
 * products into 4 simd_sum results (one barrier per 4 t's instead of one
 * per t). Cross-simd reduction also packs 4 scores per pass.
 *
 * For n=1024 long contexts this cuts attention barrier count 4× — the
 * single biggest contributor to per-token wall time at long context (see
 * commit 27f62c6's MLX comparison: qw36 drops to 31% of MLX at n=1024
 * because our O(seq) per-t barrier overhead grows linearly).
 *
 * Same TG geometry as the fast variant: one TG per head, tg_size lanes
 * covering head_dim. Threadgroup memory layout grows partials[] from
 * simd_count to simd_count*4 floats. Caller must allocate
 * (simd_count * 4 + positions) floats of scratch. */
kernel void qw36_attn_decode_fused_f16kv_x4_f32(
    device uchar       *y            [[buffer(0)]],
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
    constant uint      &y_dtype      [[buffer(20)]],
    constant uint      &q_offset     [[buffer(21)]],
    constant uint      &k_offset     [[buffer(22)]],
    constant uint      &v_offset     [[buffer(23)]],
    constant uint      &kv_transposed [[buffer(24)]],
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

    uint simd_lane = lane & 31u;
    uint simd_id = lane >> 5;
    uint simd_count = (tg_size + 31u) >> 5;
    /* partials layout: 4 slots per simdgroup for the x4 batched K dots,
     * plus the standard 1-per-simdgroup space at the front for the
     * single-result reductions used in q_norm / k_norm. */
    threadgroup float *partials = scratch;
    threadgroup float *scores = scratch + simd_count * 4u;

    device const float *q_base = q_raw + q_offset;
    device const float *k_base = k_raw + k_offset;
    device const float *v_base = v_raw + v_offset;
    uint q_head_stride = q_has_gate ? (2u * head_dim) : head_dim;
    device const float *qh_base = q_base + h * q_head_stride;
    device const float *gate_base = q_base + h * 2u * head_dim + head_dim;
    device const float *kh_base = k_base + kvh * head_dim;

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
        uint off = qw36_kv_cache_offset(seq_pos, kvh, lane, head_dim,
                                        seq_capacity, kv_len, kv_transposed);
        k_cache[off] = half(kv);
        v_cache[off] = half(v_base[kvh * head_dim + lane]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_sqrt_d = rsqrt(float(head_dim));

    /* Scoring loop: 4 t per outer iter. Each lane computes 4 partial dots
     * in parallel (same qv, 4 kvals), 4 simd_sums, then one shared
     * threadgroup barrier + cross-simd reduce produces 4 scores. */
    uint t_full = count & ~3u;
    for (uint t0 = 0; t0 < t_full; t0 += 4u) {
        float k0val = 0.0f, k1val = 0.0f, k2val = 0.0f, k3val = 0.0f;
        if (lane < head_dim) {
            k0val = ((t0 + 0u) == seq_pos) ? kv : float(k_cache[
                qw36_kv_cache_offset(t0 + 0u, kvh, lane, head_dim,
                                      seq_capacity, kv_len, kv_transposed)]);
            k1val = ((t0 + 1u) == seq_pos) ? kv : float(k_cache[
                qw36_kv_cache_offset(t0 + 1u, kvh, lane, head_dim,
                                      seq_capacity, kv_len, kv_transposed)]);
            k2val = ((t0 + 2u) == seq_pos) ? kv : float(k_cache[
                qw36_kv_cache_offset(t0 + 2u, kvh, lane, head_dim,
                                      seq_capacity, kv_len, kv_transposed)]);
            k3val = ((t0 + 3u) == seq_pos) ? kv : float(k_cache[
                qw36_kv_cache_offset(t0 + 3u, kvh, lane, head_dim,
                                      seq_capacity, kv_len, kv_transposed)]);
        }
        float d0 = simd_sum((lane < head_dim) ? qv * k0val : 0.0f);
        float d1 = simd_sum((lane < head_dim) ? qv * k1val : 0.0f);
        float d2 = simd_sum((lane < head_dim) ? qv * k2val : 0.0f);
        float d3 = simd_sum((lane < head_dim) ? qv * k3val : 0.0f);
        if (simd_lane == 0u) {
            partials[simd_id * 4u + 0u] = d0;
            partials[simd_id * 4u + 1u] = d1;
            partials[simd_id * 4u + 2u] = d2;
            partials[simd_id * 4u + 3u] = d3;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 4u) {
            float total = 0.0f;
            for (uint s = 0u; s < simd_count; ++s) {
                total += partials[s * 4u + lane];
            }
            scores[t0 + lane] = total * inv_sqrt_d;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    /* Tail: <4 remaining t. */
    for (uint t = t_full; t < count; ++t) {
        float kval = 0.0f;
        if (lane < head_dim) {
            kval = (t == seq_pos)
                ? kv
                : float(k_cache[qw36_kv_cache_offset(
                    t, kvh, lane, head_dim, seq_capacity, kv_len,
                    kv_transposed)]);
        }
        float dot = simd_sum((lane < head_dim) ? qv * kval : 0.0f);
        if (simd_lane == 0u) partials[simd_id] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dot = simd_sum((lane < simd_count) ? partials[lane] : 0.0f);
        if (lane == 0u) scores[t] = dot * inv_sqrt_d;
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
                ? v_base[kvh * head_dim + lane]
                : float(v_cache[qw36_kv_cache_offset(
                    t, kvh, lane, head_dim, seq_capacity, kv_len,
                    kv_transposed)]);
            acc += scores[t] * vv;
        }
        if (q_has_gate) {
            float g = gate_base[lane];
            acc *= 1.0f / (1.0f + exp(-g));
        }
        qw36_store_scalar(y, y_dtype, h * head_dim + lane, acc);
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

/* qmv_quad-style Q4_K matmul (decode, M=1, output rows tiled by 64)
 *
 * Experimental GGUF Q4_K variant of MLX's qmv_quad schedule. One
 * 32-thread SIMD threadgroup contains 8 quadgroups; each quadgroup emits
 * 8 output rows and its 4 lanes split K into contiguous quarters. The
 * lane-local partial sums are reduced with quad_sum only, so this path
 * avoids the 256-thread kernel's threadgroup cache + barrier reduction.
 *
 * The Q4_K decode is intentionally identical to qw36_gguf_q4_k_value:
 *   val = q * (d * sc[sub]) - dmin * mn[sub]
 * Output stays fp32 to match the existing quant-matmul contract. */
kernel void qw36_matmul_q4_k_qmv_quad_f32(
    device float       *y    [[buffer(0)]],
    device const float *x    [[buffer(1)]],
    device const uchar *w    [[buffer(2)]],
    constant uint      &K    [[buffer(3)]],
    constant uint      &N    [[buffer(4)]],
    uint3 tg                 [[threadgroup_position_in_grid]],
    uint quad_gid            [[quadgroup_index_in_threadgroup]],
    uint quad_lid            [[thread_index_in_quadgroup]])
{
    constexpr uint quads_per_simd = 8u;
    constexpr uint results_per_quadgroup = 8u;
    uint out0 = tg.y * quads_per_simd * results_per_quadgroup + quad_gid;
    uint K_blocks = K >> 8;
    uint row_bytes = K_blocks * 144u;

    float acc[results_per_quadgroup];
    bool active[results_per_quadgroup];
    float d_cache[results_per_quadgroup];
    float dmin_cache[results_per_quadgroup];
    uchar sc_cache[results_per_quadgroup][8];
    uchar mn_cache[results_per_quadgroup][8];
    for (uint r = 0; r < results_per_quadgroup; ++r) {
        uint row = out0 + r * quads_per_simd;
        acc[r] = 0.0f;
        active[r] = row < N;
        d_cache[r] = 0.0f;
        dmin_cache[r] = 0.0f;
        for (uint s = 0; s < 8u; ++s) {
            sc_cache[r][s] = 0;
            mn_cache[r][s] = 0;
        }
    }

    uint values_per_lane = K >> 2;
    uint k_begin = quad_lid * values_per_lane;
    uint k_end = (quad_lid == 3u) ? K : min(k_begin + values_per_lane, K);
    uint cached_sb = 0xffffffffu;

    for (uint k = k_begin; k < k_end; ++k) {
        uint sb = k >> 8;
        uint local = k & 255u;
        uint iter = local >> 6;
        uint h = (local >> 5) & 1u;
        uint lane = local & 31u;
        uint sub = iter * 2u + h;

        if (sb != cached_sb) {
            cached_sb = sb;
            for (uint r = 0; r < results_per_quadgroup; ++r) {
                if (!active[r]) continue;
                uint row = out0 + r * quads_per_simd;
                device const uchar *p = w + row * row_bytes + sb * 144u;
                d_cache[r] = qw36_gguf_f16(p);
                dmin_cache[r] = qw36_gguf_f16(p + 2);
                qw36_gguf_q4_scales(p + 4, sc_cache[r], mn_cache[r]);
            }
        }

        float xv = x[k];
        for (uint r = 0; r < results_per_quadgroup; ++r) {
            if (!active[r]) continue;
            uint row = out0 + r * quads_per_simd;
            device const uchar *p = w + row * row_bytes + sb * 144u;
            uchar byte = p[16u + iter * 32u + lane];
            float q = (h == 0u) ? float(byte & 0x0fu) : float(byte >> 4);
            float wval = q * (d_cache[r] * float(sc_cache[r][sub]))
                       - dmin_cache[r] * float(mn_cache[r][sub]);
            acc[r] += xv * wval;
        }
    }

    for (uint r = 0; r < results_per_quadgroup; ++r) {
        float v = quad_sum(acc[r]);
        uint row = out0 + r * quads_per_simd;
        if (quad_lid == 0u && row < N) y[row] = v;
    }
}

static inline float qw36_affine32_half(device const uchar *p, uint half_idx)
{
    device const half *h = (device const half *)p;
    return float(h[half_idx]);
}

static inline float qw36_qdot4_affine32_16(device const uchar *qbytes,
                                           device const float *x,
                                           uint k,
                                           float scale,
                                           float bias)
{
    float qacc = 0.0f;
    float xsum = 0.0f;
    for (uint i = 0; i < 8u; ++i) {
        uchar byte = qbytes[i];
        float x0 = x[k + i * 2u + 0u];
        float x1 = x[k + i * 2u + 1u];
        xsum += x0 + x1;
        qacc += x0 * float(byte & 0x0fu) + x1 * float(byte >> 4);
    }
    return scale * qacc + bias * xsum;
}

/* MLX-style qdot: read packed weights as uint16 (4 nibbles each), use raw
 * masked bits without shifts — caller pre-scales x_thread by 1/16/256/4096
 * so masks 0x000f / 0x00f0 / 0x0f00 / 0xf000 multiply correctly. */
static inline float qw36_qdot4_affine32_16_mlx(device const uchar *qbytes,
                                               const thread float *x_thread,
                                               float scale,
                                               float bias,
                                               float sum)
{
    device const uint16_t *ws = (device const uint16_t *)qbytes;
    float accum = 0.0f;
    for (uint i = 0; i < 4u; ++i) {
        uint w = uint(ws[i]);
        accum += x_thread[4u*i + 0u] * float(w & 0x000fu)
              +  x_thread[4u*i + 1u] * float(w & 0x00f0u)
              +  x_thread[4u*i + 2u] * float(w & 0x0f00u)
              +  x_thread[4u*i + 3u] * float(w & 0xf000u);
    }
    return scale * accum + bias * sum;
}

static inline uint qw36_unpack5(device const uchar *p, uint idx)
{
    ulong bits = ulong(p[0]) |
                 (ulong(p[1]) << 8) |
                 (ulong(p[2]) << 16) |
                 (ulong(p[3]) << 24) |
                 (ulong(p[4]) << 32);
    return uint((bits >> (5u * idx)) & 31ul);
}

static inline float qw36_qdot5_affine32_16(device const uchar *qbytes,
                                           device const float *x,
                                           uint k,
                                           float scale,
                                           float bias)
{
    float qacc = 0.0f;
    float xsum = 0.0f;
    for (uint pack = 0; pack < 2u; ++pack) {
        device const uchar *p = qbytes + pack * 5u;
        for (uint i = 0; i < 8u; ++i) {
            uint xi = pack * 8u + i;
            float xv = x[k + xi];
            xsum += xv;
            qacc += xv * float(qw36_unpack5(p, i));
        }
    }
    return scale * qacc + bias * xsum;
}

/* MLX-style 5-bit qdot — masked byte × pre-scaled x_thread.  Caller passes
 * 16 floats laid out as two 8-element packs; each 8-element pack reads
 * 5 bytes of weight.  Pre-scaling table (see load_vector helper below)
 * mirrors MLX's bits=5 load_vector. */
static inline float qw36_qdot5_affine32_16_mlx(device const uchar *qbytes,
                                               const thread float *x_thread,
                                               float scale,
                                               float bias,
                                               float xsum)
{
    float accum = 0.0f;
    for (uint pack = 0; pack < 2u; ++pack) {
        device const uchar *w = qbytes + pack * 5u;
        const thread float *xt = x_thread + 8u * pack;
        accum += float(w[0] & 0x1fu) * xt[0]
              +  float(w[0] & 0xe0u) * xt[1]
              +  float(w[1] & 0x03u) * (xt[1] * 256.0f)
              +  float(w[1] & 0x7cu) * xt[2]
              +  float(w[1] & 0x80u) * xt[3]
              +  float(w[2] & 0x0fu) * (xt[3] * 256.0f)
              +  float(w[2] & 0xf0u) * xt[4]
              +  float(w[3] & 0x01u) * (xt[4] * 256.0f)
              +  float(w[3] & 0x3eu) * xt[5]
              +  float(w[3] & 0xc0u) * xt[6]
              +  float(w[4] & 0x07u) * (xt[6] * 256.0f)
              +  float(w[4] & 0xf8u) * xt[7];
    }
    return scale * accum + bias * xsum;
}

static inline float qw36_load_x_for_q5_affine32_16(device const float *x,
                                                   uint k,
                                                   thread float *x_thread)
{
    float xsum = 0.0f;
    for (uint pack = 0; pack < 2u; ++pack) {
        uint base = pack * 8u;
        float x0 = x[k + base + 0u];
        float x1 = x[k + base + 1u];
        float x2 = x[k + base + 2u];
        float x3 = x[k + base + 3u];
        float x4 = x[k + base + 4u];
        float x5 = x[k + base + 5u];
        float x6 = x[k + base + 6u];
        float x7 = x[k + base + 7u];
        xsum += x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
        x_thread[base + 0u] = x0;
        x_thread[base + 1u] = x1 * (1.0f / 32.0f);
        x_thread[base + 2u] = x2 * (1.0f / 4.0f);
        x_thread[base + 3u] = x3 * (1.0f / 128.0f);
        x_thread[base + 4u] = x4 * (1.0f / 16.0f);
        x_thread[base + 5u] = x5 * (1.0f / 2.0f);
        x_thread[base + 6u] = x6 * (1.0f / 64.0f);
        x_thread[base + 7u] = x7 * (1.0f / 8.0f);
    }
    return xsum;
}

static inline uint qw36_unpack6(device const uchar *p, uint idx)
{
    uint bit = 6u * idx;
    uint byte = bit >> 3;
    uint shift = bit & 7u;
    uint bits = uint(p[byte]);
    if (byte + 1u < 12u)
        bits |= uint(p[byte + 1u]) << 8;
    return (bits >> shift) & 63u;
}

static inline float qw36_qdot6_scale16_16(device const uchar *qbytes,
                                          device const float *x,
                                          uint k,
                                          float scale)
{
    float qacc = 0.0f;
    for (uint i = 0; i < 16u; ++i)
        qacc += x[k + i] * (float(qw36_unpack6(qbytes, i)) - 32.0f);
    return scale * qacc;
}

/* MLX-style Q6 qdot — uses masked-byte * pre-scaled x_thread to skip shifts.
 * x_thread layout (set by caller via qw36_load_x_for_q6_scale16_16):
 *   x_thread[4i + 0] = x[4i + 0]
 *   x_thread[4i + 1] = x[4i + 1] / 64
 *   x_thread[4i + 2] = x[4i + 2] / 16
 *   x_thread[4i + 3] = x[4i + 3] / 4
 * For Q6K_SCALE16 our value is (q - 32) * scale so we return
 *     scale * accum - 32 * scale * sum_x .  Caller passes sum_x.  */
static inline float qw36_qdot6_scale16_16_mlx(device const uchar *qbytes,
                                              const thread float *x_thread,
                                              float scale,
                                              float sum_x)
{
    float accum = 0.0f;
    for (uint i = 0; i < 4u; ++i) {
        device const uchar *w = qbytes + 3u * i;
        const thread float *xt = x_thread + 4u * i;
        accum += float(w[0] & 0x3fu) * xt[0]
              +  float(w[0] & 0xc0u) * xt[1]
              +  float(w[1] & 0x0fu) * (xt[1] * 256.0f)
              +  float(w[1] & 0xf0u) * xt[2]
              +  float(w[2] & 0x03u) * (xt[2] * 256.0f)
              +  float(w[2] & 0xfcu) * xt[3];
    }
    return scale * accum - 32.0f * scale * sum_x;
}

static inline float qw36_load_x_for_q6_scale16_16(device const float *x,
                                                  uint k,
                                                  thread float *x_thread)
{
    float sum_x = 0.0f;
    for (uint i = 0; i < 16u; i += 4u) {
        float x0 = x[k + i + 0u];
        float x1 = x[k + i + 1u];
        float x2 = x[k + i + 2u];
        float x3 = x[k + i + 3u];
        sum_x += x0 + x1 + x2 + x3;
        x_thread[i + 0u] = x0;
        x_thread[i + 1u] = x1 * (1.0f / 64.0f);
        x_thread[i + 2u] = x2 * (1.0f / 16.0f);
        x_thread[i + 3u] = x3 * (1.0f / 4.0f);
    }
    return sum_x;
}

/* Q4K_AFFINE32 qmv_fast-style matmul.
 *
 * Layout per 256 elements:
 *   half scale[8], half bias[8], uint8 packed_q[8][16]
 * where each 32-element group is affine: value = q * scale + bias.
 *
 * Threadgroup: 4 SIMD groups (128 threads). Each SIMD group owns 4 output
 * rows. Each lane processes 16 contiguous K values at a time, so two lanes
 * cover one affine32 group and 32 lanes cover a 512-wide K tile. This keeps
 * intra-row parallelism while reusing the same x segment across 4 rows. */
kernel void qw36_matmul_q4k_affine32_qmv_fast_f32(
    device float       *y        [[buffer(0)]],
    device const float *x        [[buffer(1)]],
    device const uchar *w        [[buffer(2)]],
    constant uint      &K        [[buffer(3)]],
    constant uint      &N        [[buffer(4)]],
    uint3               tg       [[threadgroup_position_in_grid]],
    uint                simd_gid [[simdgroup_index_in_threadgroup]],
    uint                simd_lid [[thread_index_in_simdgroup]])
{
    constexpr uint simdgroups_per_tg = 4u;
    constexpr uint rows_per_simd = 4u;
    constexpr uint values_per_thread = 16u;
    constexpr uint block_k = values_per_thread * 32u; /* 512 */

    uint row0 = tg.x * simdgroups_per_tg * rows_per_simd
              + simd_gid * rows_per_simd;
    uint row_bytes = (K >> 8) * 160u;

    float acc[rows_per_simd];
    bool active[rows_per_simd];
    for (uint r = 0; r < rows_per_simd; ++r) {
        uint row = row0 + r;
        acc[r] = 0.0f;
        active[r] = row < N;
    }

    for (uint k0 = 0; k0 < K; k0 += block_k) {
        uint k = k0 + simd_lid * values_per_thread;
        if (k + values_per_thread > K) continue;
        uint group = k >> 5;          /* 32-element affine group */
        uint block = group >> 3;      /* 8 groups per 256-element block */
        uint sub = group & 7u;
        uint group_local = k & 31u;   /* 0 or 16 for this schedule */
        uint qbyte_offset = group_local >> 1;

        for (uint r = 0; r < rows_per_simd; ++r) {
            if (!active[r]) continue;
            uint row = row0 + r;
            device const uchar *blk = w + row * row_bytes + block * 160u;
            float scale = qw36_affine32_half(blk, sub);
            float bias = qw36_affine32_half(blk + 16u, sub);
            device const uchar *qbytes = blk + 32u + sub * 16u + qbyte_offset;
            acc[r] += qw36_qdot4_affine32_16(qbytes, x, k, scale, bias);
        }
    }

    for (uint r = 0; r < rows_per_simd; ++r) {
        float v = simd_sum(acc[r]);
        uint row = row0 + r;
        if (simd_lid == 0u && row < N) y[row] = v;
    }
}

/* MLX-style variant: pre-scale x_thread once per outer K iteration so the
 * inner qdot uses raw masked uint16 bits and saves 16 nibble shifts per
 * lane per row. Same Q4K_AFFINE32 layout / TG geometry. Gate: env
 * QW36_METAL_Q4K_AFFINE32_MLX=1 picks this pipeline. */
kernel void qw36_matmul_q4k_affine32_qmv_mlx_f32(
    device float       *y        [[buffer(0)]],
    device const float *x        [[buffer(1)]],
    device const uchar *w        [[buffer(2)]],
    constant uint      &K        [[buffer(3)]],
    constant uint      &N        [[buffer(4)]],
    uint3               tg       [[threadgroup_position_in_grid]],
    uint                simd_gid [[simdgroup_index_in_threadgroup]],
    uint                simd_lid [[thread_index_in_simdgroup]])
{
    constexpr uint simdgroups_per_tg = 4u;
    constexpr uint rows_per_simd = 4u;
    constexpr uint values_per_thread = 16u;
    constexpr uint block_k = values_per_thread * 32u; /* 512 */

    uint row0 = tg.x * simdgroups_per_tg * rows_per_simd
              + simd_gid * rows_per_simd;
    uint row_bytes = (K >> 8) * 160u;

    float acc[rows_per_simd];
    bool active[rows_per_simd];
    for (uint r = 0; r < rows_per_simd; ++r) {
        uint row = row0 + r;
        acc[r] = 0.0f;
        active[r] = row < N;
    }

    thread float x_thread[values_per_thread];

    for (uint k0 = 0; k0 < K; k0 += block_k) {
        uint k = k0 + simd_lid * values_per_thread;
        if (k + values_per_thread > K) continue;

        /* Pre-scale x once per lane per K block; sum stays raw for bias. */
        float xsum = 0.0f;
        for (uint i = 0; i < values_per_thread; i += 4u) {
            float x0 = x[k + i + 0u];
            float x1 = x[k + i + 1u];
            float x2 = x[k + i + 2u];
            float x3 = x[k + i + 3u];
            xsum += x0 + x1 + x2 + x3;
            x_thread[i + 0u] = x0;
            x_thread[i + 1u] = x1 * (1.0f / 16.0f);
            x_thread[i + 2u] = x2 * (1.0f / 256.0f);
            x_thread[i + 3u] = x3 * (1.0f / 4096.0f);
        }

        uint group = k >> 5;
        uint block = group >> 3;
        uint sub = group & 7u;
        uint qbyte_offset = (k & 31u) >> 1;

        for (uint r = 0; r < rows_per_simd; ++r) {
            if (!active[r]) continue;
            uint row = row0 + r;
            device const uchar *blk = w + row * row_bytes + block * 160u;
            float scale = qw36_affine32_half(blk, sub);
            float bias = qw36_affine32_half(blk + 16u, sub);
            device const uchar *qbytes = blk + 32u + sub * 16u + qbyte_offset;
            acc[r] += qw36_qdot4_affine32_16_mlx(qbytes, x_thread, scale,
                                                  bias, xsum);
        }
    }

    for (uint r = 0; r < rows_per_simd; ++r) {
        float v = simd_sum(acc[r]);
        uint row = row0 + r;
        if (simd_lid == 0u && row < N) y[row] = v;
    }
}

/* Q5K_AFFINE32 sibling of the Q4 affine32 qmv_fast kernel.
 *
 * Layout per 256 elements:
 *   half scale[8], half bias[8], uint8 packed_q[8][20]
 * where each 32-element group stores four little-endian 5-byte packs of
 * eight 5-bit values. */
kernel void qw36_matmul_q5k_affine32_qmv_fast_f32(
    device float       *y        [[buffer(0)]],
    device const float *x        [[buffer(1)]],
    device const uchar *w        [[buffer(2)]],
    constant uint      &K        [[buffer(3)]],
    constant uint      &N        [[buffer(4)]],
    uint3               tg       [[threadgroup_position_in_grid]],
    uint                simd_gid [[simdgroup_index_in_threadgroup]],
    uint                simd_lid [[thread_index_in_simdgroup]])
{
    constexpr uint simdgroups_per_tg = 4u;
    constexpr uint rows_per_simd = 4u;
    constexpr uint values_per_thread = 16u;
    constexpr uint block_k = values_per_thread * 32u; /* 512 */

    uint row0 = tg.x * simdgroups_per_tg * rows_per_simd
              + simd_gid * rows_per_simd;
    uint row_bytes = (K >> 8) * 192u;

    float acc[rows_per_simd];
    bool active[rows_per_simd];
    for (uint r = 0; r < rows_per_simd; ++r) {
        uint row = row0 + r;
        acc[r] = 0.0f;
        active[r] = row < N;
    }

    for (uint k0 = 0; k0 < K; k0 += block_k) {
        uint k = k0 + simd_lid * values_per_thread;
        if (k + values_per_thread > K) continue;
        uint group = k >> 5;
        uint block = group >> 3;
        uint sub = group & 7u;
        uint group_local = k & 31u;
        uint qbyte_offset = (group_local >> 3) * 5u;

        for (uint r = 0; r < rows_per_simd; ++r) {
            if (!active[r]) continue;
            uint row = row0 + r;
            device const uchar *blk = w + row * row_bytes + block * 192u;
            float scale = qw36_affine32_half(blk, sub);
            float bias = qw36_affine32_half(blk + 16u, sub);
            device const uchar *qbytes = blk + 32u + sub * 20u + qbyte_offset;
            acc[r] += qw36_qdot5_affine32_16(qbytes, x, k, scale, bias);
        }
    }

    for (uint r = 0; r < rows_per_simd; ++r) {
        float v = simd_sum(acc[r]);
        uint row = row0 + r;
        if (simd_lid == 0u && row < N) y[row] = v;
    }
}

/* MLX-style Q5K_AFFINE32 matmul.  Same layout + threadgroup geometry as
 * qw36_matmul_q5k_affine32_qmv_fast_f32 but uses masked-byte × pre-scaled
 * x_thread to skip 16 dependent 5-bit shifts per lane per row.  Q5_K is
 * the gate_up matmul (6144×1024) — the most frequently called Q5_K shape
 * in Qwen3.5-0.8B-Q4_K_M — so per-element work matters here.  Gate behind
 * QW36_METAL_Q5K_AFFINE32_MLX=1. */
kernel void qw36_matmul_q5k_affine32_qmv_mlx_f32(
    device float       *y        [[buffer(0)]],
    device const float *x        [[buffer(1)]],
    device const uchar *w        [[buffer(2)]],
    constant uint      &K        [[buffer(3)]],
    constant uint      &N        [[buffer(4)]],
    uint3               tg       [[threadgroup_position_in_grid]],
    uint                simd_gid [[simdgroup_index_in_threadgroup]],
    uint                simd_lid [[thread_index_in_simdgroup]])
{
    constexpr uint simdgroups_per_tg = 4u;
    constexpr uint rows_per_simd = 4u;
    constexpr uint values_per_thread = 16u;
    constexpr uint block_k = values_per_thread * 32u; /* 512 */

    uint row0 = tg.x * simdgroups_per_tg * rows_per_simd
              + simd_gid * rows_per_simd;
    uint row_bytes = (K >> 8) * 192u;

    float acc[rows_per_simd];
    bool active[rows_per_simd];
    for (uint r = 0; r < rows_per_simd; ++r) {
        uint row = row0 + r;
        acc[r] = 0.0f;
        active[r] = row < N;
    }

    thread float x_thread[values_per_thread];

    for (uint k0 = 0; k0 < K; k0 += block_k) {
        uint k = k0 + simd_lid * values_per_thread;
        if (k + values_per_thread > K) continue;
        uint group = k >> 5;
        uint block = group >> 3;
        uint sub = group & 7u;
        uint group_local = k & 31u;
        uint qbyte_offset = (group_local >> 3) * 5u;

        float xsum = qw36_load_x_for_q5_affine32_16(x, k, x_thread);

        for (uint r = 0; r < rows_per_simd; ++r) {
            if (!active[r]) continue;
            uint row = row0 + r;
            device const uchar *blk = w + row * row_bytes + block * 192u;
            float scale = qw36_affine32_half(blk, sub);
            float bias = qw36_affine32_half(blk + 16u, sub);
            device const uchar *qbytes = blk + 32u + sub * 20u + qbyte_offset;
            acc[r] += qw36_qdot5_affine32_16_mlx(qbytes, x_thread, scale, bias, xsum);
        }
    }

    for (uint r = 0; r < rows_per_simd; ++r) {
        float v = simd_sum(acc[r]);
        uint row = row0 + r;
        if (simd_lid == 0u && row < N) y[row] = v;
    }
}

/* Q6K_SCALE16 qmv_fast kernel.
 *
 * Layout per 256 elements:
 *   half scale[16], uint8 packed_q[16][12]
 * where each 16-element group stores one 6-bit signed group encoded as
 * unsigned q in [0, 63] with value = (q - 32) * scale. */
kernel void qw36_matmul_q6k_scale16_qmv_fast_f32(
    device float       *y        [[buffer(0)]],
    device const float *x        [[buffer(1)]],
    device const uchar *w        [[buffer(2)]],
    constant uint      &K        [[buffer(3)]],
    constant uint      &N        [[buffer(4)]],
    uint3               tg       [[threadgroup_position_in_grid]],
    uint                simd_gid [[simdgroup_index_in_threadgroup]],
    uint                simd_lid [[thread_index_in_simdgroup]])
{
    constexpr uint simdgroups_per_tg = 4u;
    constexpr uint rows_per_simd = 4u;
    constexpr uint values_per_thread = 16u;
    constexpr uint block_k = values_per_thread * 32u; /* 512 */

    uint row0 = tg.x * simdgroups_per_tg * rows_per_simd
              + simd_gid * rows_per_simd;
    uint row_bytes = (K >> 8) * 224u;

    float acc[rows_per_simd];
    bool active[rows_per_simd];
    for (uint r = 0; r < rows_per_simd; ++r) {
        uint row = row0 + r;
        acc[r] = 0.0f;
        active[r] = row < N;
    }

    for (uint k0 = 0; k0 < K; k0 += block_k) {
        uint k = k0 + simd_lid * values_per_thread;
        if (k + values_per_thread > K) continue;
        uint group = k >> 4;          /* 16-element scale group */
        uint block = group >> 4;      /* 16 groups per 256-element block */
        uint sub = group & 15u;

        for (uint r = 0; r < rows_per_simd; ++r) {
            if (!active[r]) continue;
            uint row = row0 + r;
            device const uchar *blk = w + row * row_bytes + block * 224u;
            float scale = qw36_affine32_half(blk, sub);
            device const uchar *qbytes = blk + 32u + sub * 12u;
            acc[r] += qw36_qdot6_scale16_16(qbytes, x, k, scale);
        }
    }

    for (uint r = 0; r < rows_per_simd; ++r) {
        float v = simd_sum(acc[r]);
        uint row = row0 + r;
        if (simd_lid == 0u && row < N) y[row] = v;
    }
}

/* MLX-style Q6K_SCALE16 matmul. Same layout + threadgroup geometry as
 * qw36_matmul_q6k_scale16_qmv_fast_f32 but uses masked-byte * pre-scaled
 * x_thread (the MLX qdot trick) to cut per-element ops ~2-3×. Opt-in via
 * QW36_METAL_Q6K_SCALE16_MLX=1 — see metal_matmul dispatch.
 *
 * For lm_head (rows ≈ 248K) this is the single biggest gpu_ms line in PERF,
 * so the throughput-vs-shifts ratio matters more here than elsewhere. */
kernel void qw36_matmul_q6k_scale16_qmv_mlx_f32(
    device float       *y        [[buffer(0)]],
    device const float *x        [[buffer(1)]],
    device const uchar *w        [[buffer(2)]],
    constant uint      &K        [[buffer(3)]],
    constant uint      &N        [[buffer(4)]],
    uint3               tg       [[threadgroup_position_in_grid]],
    uint                simd_gid [[simdgroup_index_in_threadgroup]],
    uint                simd_lid [[thread_index_in_simdgroup]])
{
    constexpr uint simdgroups_per_tg = 4u;
    constexpr uint rows_per_simd = 4u;
    constexpr uint values_per_thread = 16u;
    constexpr uint block_k = values_per_thread * 32u; /* 512 */

    uint row0 = tg.x * simdgroups_per_tg * rows_per_simd
              + simd_gid * rows_per_simd;
    uint row_bytes = (K >> 8) * 224u;

    float acc[rows_per_simd];
    bool active[rows_per_simd];
    for (uint r = 0; r < rows_per_simd; ++r) {
        uint row = row0 + r;
        acc[r] = 0.0f;
        active[r] = row < N;
    }

    thread float x_thread[values_per_thread];

    for (uint k0 = 0; k0 < K; k0 += block_k) {
        uint k = k0 + simd_lid * values_per_thread;
        if (k + values_per_thread > K) continue;
        uint group = k >> 4;
        uint block = group >> 4;
        uint sub = group & 15u;

        float sum_x = qw36_load_x_for_q6_scale16_16(x, k, x_thread);

        for (uint r = 0; r < rows_per_simd; ++r) {
            if (!active[r]) continue;
            uint row = row0 + r;
            device const uchar *blk = w + row * row_bytes + block * 224u;
            float scale = qw36_affine32_half(blk, sub);
            device const uchar *qbytes = blk + 32u + sub * 12u;
            acc[r] += qw36_qdot6_scale16_16_mlx(qbytes, x_thread, scale, sum_x);
        }
    }

    for (uint r = 0; r < rows_per_simd; ++r) {
        float v = simd_sum(acc[r]);
        uint row = row0 + r;
        if (simd_lid == 0u && row < N) y[row] = v;
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

/* ----------------------------------------------------------------- *
 * qmv_quad-style fp16 GEMV (M=1 decode).
 *
 * Dense fp16 variant of MLX's qmv_quad_impl: one 32-thread SIMD
 * threadgroup is split into 8 quadgroups. Each quadgroup computes 8
 * output rows, so a TG emits up to 64 rows. The 4 lanes in a quad split
 * K by lane id (K/4 values each), reuse each x[k] across 8 output rows,
 * and reduce with quad_sum only. No threadgroup memory or barriers.
 * ----------------------------------------------------------------- */
kernel void qw36_matmul_qmv_quad_f16(
    device uchar       *y        [[buffer(0)]],
    device const uchar *x        [[buffer(1)]],
    device const half  *w        [[buffer(2)]],
    constant uint      &K        [[buffer(3)]],
    constant uint      &N        [[buffer(4)]],
    constant uint      &x_dtype  [[buffer(5)]],
    constant uint      &y_dtype  [[buffer(6)]],
    uint3 tg                     [[threadgroup_position_in_grid]],
    uint quad_gid                [[quadgroup_index_in_threadgroup]],
    uint quad_lid                [[thread_index_in_quadgroup]])
{
    constexpr uint quads_per_simd = 8u;
    constexpr uint results_per_quadgroup = 8u;
    uint out0 = tg.y * quads_per_simd * results_per_quadgroup + quad_gid;

    float acc[results_per_quadgroup];
    for (uint r = 0; r < results_per_quadgroup; ++r) acc[r] = 0.0f;

    uint vec_end = K & ~15u;
    for (uint k = quad_lid * 4u; k < vec_end; k += 16u) {
        float4 xv;
        if (x_dtype == QW36_DTYPE_F16) {
            xv = float4(*((device const half4 *)(((device const half *)x) + k)));
        } else {
            xv = *((device const float4 *)(((device const float *)x) + k));
        }
        for (uint r = 0; r < results_per_quadgroup; ++r) {
            uint row = out0 + r * quads_per_simd;
            if (row < N) {
                float4 wv = float4(*((device const half4 *)(w + row * K + k)));
                acc[r] += dot(xv, wv);
            }
        }
    }
    for (uint k = vec_end + quad_lid; k < K; k += 4u) {
        float xv = qw36_load_scalar((device const uchar *)x, x_dtype, k);
        for (uint r = 0; r < results_per_quadgroup; ++r) {
            uint row = out0 + r * quads_per_simd;
            if (row < N) acc[r] += xv * float(w[row * K + k]);
        }
    }

    for (uint r = 0; r < results_per_quadgroup; ++r) {
        float v = quad_sum(acc[r]);
        uint row = out0 + r * quads_per_simd;
        if (quad_lid == 0u && row < N) qw36_store_scalar(y, y_dtype, row, v);
    }
}

/* ----------------------------------------------------------------- *
 * Experimental simdgroup_matrix fp16 GEMV for M=1 decode.
 *
 * One threadgroup has 8 SIMD groups and emits up to 64 output rows.
 * Each SIMD group owns an 8-row output tile and uses 8x8 half MMA along K.
 * Since the input has one row, only row 0 of A is populated; the remaining
 * rows are zero. This intentionally trades extra tensor-core arithmetic for
 * lower per-row overhead and is opt-in from the host.
 * ----------------------------------------------------------------- */
kernel void qw36_matmul_mma_f16_f32(
    device uchar       *y        [[buffer(0)]],
    device const uchar *x        [[buffer(1)]],
    device const half  *w        [[buffer(2)]],
    constant uint      &K        [[buffer(3)]],
    constant uint      &N        [[buffer(4)]],
    constant uint      &x_dtype  [[buffer(5)]],
    constant uint      &y_dtype  [[buffer(6)]],
    threadgroup half   *ab_smem  [[threadgroup(0)]],
    threadgroup float  *c_smem   [[threadgroup(1)]],
    uint3               tg       [[threadgroup_position_in_grid]],
    uint                simd_gid [[simdgroup_index_in_threadgroup]],
    uint                simd_lid [[thread_index_in_simdgroup]])
{
    constexpr uint simdgroups_per_tg = 8u;
    constexpr uint tile_n = 8u;
    constexpr uint tile_k = 8u;
    uint row0 = tg.x * simdgroups_per_tg * tile_n + simd_gid * tile_n;

    threadgroup half *a_tile = ab_smem + simd_gid * (tile_k * tile_k * 2u);
    threadgroup half *b_tile = a_tile + tile_k * tile_k;
    threadgroup float *c_tile = c_smem + simd_gid * tile_k * tile_n;

    simdgroup_matrix<half, 8, 8> a_mat;
    simdgroup_matrix<half, 8, 8> b_mat;
    simdgroup_matrix<float, 8, 8> c_mat(0.0f);

    for (uint k0 = 0; k0 < K; k0 += tile_k) {
        for (uint e = simd_lid; e < tile_k * tile_k; e += 32u) {
            uint r = e / tile_k;
            uint c = e - r * tile_k;
            uint k = k0 + c;
            a_tile[e] = (r == 0u && k < K)
                ? half(qw36_load_scalar(x, x_dtype, k))
                : half(0.0);

            uint out_row = row0 + c;
            uint wk = k0 + r;
            b_tile[e] = (out_row < N && wk < K)
                ? w[out_row * K + wk]
                : half(0.0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(a_mat, a_tile, 8);
        simdgroup_load(b_mat, b_tile, 8);
        simdgroup_multiply_accumulate(c_mat, a_mat, b_mat, c_mat);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(c_mat, c_tile, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lid < tile_n) {
        uint row = row0 + simd_lid;
        if (row < N) qw36_store_scalar(y, y_dtype, row, c_tile[simd_lid]);
    }
}

/* ----------------------------------------------------------------- *
 * Fused residual_add + rmsnorm.
 *
 *   x += y;                                  // residual_add
 *   out = x * rsqrt(mean(x^2) + eps) * w;    // rmsnorm
 *
 * One pass over the same hidden-sized buffers, saves a full kernel
 * dispatch per (residual_add → rmsnorm) pair — there are two such
 * pairs per layer (post-attn → MLP rmsnorm, post-MLP → next layer's
 * input rmsnorm), so 24 layers × 2 = 48 dispatches / token.
 *
 * Threadgroup grid: 1 (single TG operates on the full hidden vec).
 * tg_size: rounded up to next 32 of hidden (= 1024 for Qwen3.5-0.8B).
 * ----------------------------------------------------------------- */
kernel void qw36_residual_rmsnorm_f32(
    device uchar       *x        [[buffer(0)]],   /* updated in-place: x += y */
    device const uchar *y        [[buffer(1)]],
    device uchar       *out      [[buffer(2)]],   /* rmsnorm output */
    device const uchar *w        [[buffer(3)]],
    constant uint      &hidden   [[buffer(4)]],
    constant float     &eps      [[buffer(5)]],
    constant uint      &x_dtype  [[buffer(6)]],
    constant uint      &y_dtype  [[buffer(7)]],
    constant uint      &out_dtype [[buffer(8)]],
    constant uint      &w_dtype  [[buffer(9)]],
    threadgroup float  *scratch  [[threadgroup(0)]],
    uint tid                     [[thread_position_in_threadgroup]],
    uint tg_size                 [[threads_per_threadgroup]])
{
    // Pass 1: x[i] = x[i] + y[i]; accumulate ss = sum(x[i]^2) in fp32.
    float local_ss = 0.0f;
    for (uint i = tid; i < hidden; i += tg_size) {
        float a = qw36_load_scalar(x, x_dtype, i);
        float b = qw36_load_scalar(y, y_dtype, i);
        float v = a + b;
        qw36_store_scalar(x, x_dtype, i, v);
        local_ss += v * v;
    }
    // simd_sum + cross-simd reduction
    float simd_v = simd_sum(local_ss);
    uint simd_lane = tid & 31u;
    uint simd_id = tid >> 5;
    if (simd_lane == 0u) scratch[simd_id] = simd_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint simd_count = (tg_size + 31u) >> 5;
    if (tid == 0u) {
        float sum = 0.0f;
        for (uint i = 0; i < simd_count; ++i) sum += scratch[i];
        scratch[0] = rsqrt(sum / float(hidden) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = scratch[0];

    // Pass 2: out[i] = x[i] * scale * w[i].
    for (uint i = tid; i < hidden; i += tg_size) {
        float v = qw36_load_scalar(x, x_dtype, i);
        float ww = qw36_load_scalar(w, w_dtype, i);
        qw36_store_scalar(out, out_dtype, i, v * scale * ww);
    }
}
