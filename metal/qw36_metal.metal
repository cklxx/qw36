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

/* ---------------------------------------------------------------- */
/* RMSNorm: out[i] = x[i] * rsqrt(mean(x^2) + eps) * w[i]            */
/* ---------------------------------------------------------------- */
kernel void qw36_rmsnorm_f32(
    device float       *out      [[buffer(0)]],
    device const float *x        [[buffer(1)]],
    device const float *w        [[buffer(2)]],
    constant uint      &hidden   [[buffer(3)]],
    constant float     &eps      [[buffer(4)]],
    uint                tid      [[thread_position_in_grid]])
{
    (void)out; (void)x; (void)w; (void)hidden; (void)eps; (void)tid;
    /* TODO(codex): threadgroup reduce, one tg per token. */
}

/* ---------------------------------------------------------------- */
/* Matmul: y[r] = sum_c x[c] * W[r,c]                                */
/* ---------------------------------------------------------------- */
kernel void qw36_matmul_f32(
    device float       *y        [[buffer(0)]],
    device const float *x        [[buffer(1)]],
    device const float *w        [[buffer(2)]],
    constant uint      &rows     [[buffer(3)]],
    constant uint      &cols     [[buffer(4)]],
    uint                gid      [[thread_position_in_grid]])
{
    (void)y; (void)x; (void)w; (void)rows; (void)cols; (void)gid;
    /* TODO(codex): tiled with threadgroup memory. */
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
    (void)gate_io; (void)up; (void)n; (void)tid;
    /* TODO(codex): elementwise silu(gate) * up. */
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
/* Attention — biggest kernel. Suggested split:                      */
/*   qw36_qkv_proj      : QKV matmul + q_norm/k_norm + RoPE         */
/*   qw36_kv_append     : write k,v into the cache at seq_pos       */
/*   qw36_attn_scores   : q · k_cache^T  / sqrt(d) with causal mask */
/*   qw36_attn_softmax  : numerically stable softmax                */
/*   qw36_attn_combine  : weighted sum over v_cache                 */
/* TODO(codex): implement.                                           */
/* ---------------------------------------------------------------- */
