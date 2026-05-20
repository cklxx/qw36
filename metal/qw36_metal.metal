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

    for (uint pair = 0; pair < rot_dim / 2; ++pair) {
        uint d0 = 2 * pair;
        uint d1 = d0 + 1;
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
