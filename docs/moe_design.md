# MoE GPU design — SwitchGLU + gather_qmm

The qw36 MoE GPU path is being rewritten to match MLX's SwitchGLU
shape. This doc captures the design intent before the kernel lands
so the implementation has a single source of truth and reviewers
have a checklist.

## Why the rewrite

Two failed kernels live in qw36's git history that this rewrite
replaces:

- **1-thread-per-output naive** (`qw36_moe_gate_up_f32` +
  `qw36_moe_down_combine_f32` — the original Metal MoE landed early
  in the project). Each output element gets a single thread that
  loops serially over `hidden` (2048 for 35B). On 35B-A3B this
  accounts for **79% of GPU time** (1375ms / 1736ms in
  `QW36_METAL_PERF=1` traces) and yields ~5 tok/s decode.
- **1-row-per-threadgroup TG experiment** (commit dropped before
  push). 32 lanes per output, simd_sum reduction. Conceptually
  right; in practice ~24× SLOWER on 35B because 4096 single-output
  threadgroups pay per-launch overhead the 1D dispatch (which packs
  many outputs per TG) avoided.
- **Q8-only fast path** (codex's `qw36_moe_gate_up_q8_f32` —
  dropped during this rewrite). Casts activations to Q8 before the
  expert matmul. Wrong direction: expert weights ARE quantized
  (Q4_K / Q5_K / Q6_K), so the kernel should operate on the
  **weight's** dtype directly, not detour through a Q8 activation
  format.

## The shape we want (lifted from MLX)

MLX's `SwitchGLU`:

```python
# routing
weights = router(x)             # (n_tokens, n_experts) — logits
indices, scores = top_k(weights, k=experts_per_tok)

# gathered expert matmul over the *selected* expert rows only
gate_out = gather_qmm(x, gate_proj, indices)   # (n_tokens, k, inter)
up_out   = gather_qmm(x, up_proj,   indices)
h        = silu(gate_out) * up_out
down_out = gather_qmm(h, down_proj, indices)   # (n_tokens, k, hidden)

# combine — weighted sum of expert outputs
y = sum_k(scores * down_out)
```

Key properties:

1. **Only the selected experts' rows are touched.** For
   `expert_count=256, experts_per_tok=8` (35B-A3B): we read 8/256 =
   **3%** of the expert weight bytes per token. Wallclock should
   approach the 3B-active-params bound, not the 35B-total bound.
2. **Quant-weight-aware dequant.** The gather_qmm kernel reads
   `Q4_K_AFFINE32` (or Q5K / Q6K_SCALE16) blocks directly, dequants
   inline per element, no Q8 detour.
3. **One SIMD group computes multiple output rows** of the
   gate/up/down matmul. With 32 lanes coalescing reads across the
   shared `hidden` axis, each block of K-quant weights is dequanted
   once and applied to multiple output rows — amortizes the
   per-block scale/bias overhead.

## Kernel sketch

Per attention head's MoE block, dispatch one threadgroup per
(token, expert_slot) where expert_slot ranges 0..k-1. Within the
threadgroup, multiple SIMD groups handle a chunk of the `inter`
dimension:

```metal
// One threadgroup = one (token, expert_slot) pair.
// tgsize = N_SIMD * 32  where each SIMD computes ROWS_PER_SIMD output rows.
kernel void qw36_moe_switchglu_gate_up(
    device float       *act,         // [token, k, inter]
    device const float *x,           // [token, hidden]
    device const uchar *expert_gate, // [n_experts, inter, hidden] in K-quant
    device const uchar *expert_up,
    device const uint  *top_idx,     // [token, k]
    constant uint &hidden, &inter, &k,
    constant uint &gate_dtype, &up_dtype,
    threadgroup float *xs_scratch,
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]])
{
    const uint token = tg.x;
    const uint slot  = tg.y;
    const uint expert = top_idx[token * k + slot];
    const uint simd_id = lid.x / 32;
    const uint simd_lane = lid.x & 31;

    // Cooperatively load x[token, *] into TG memory once.
    for (uint c = lid.x; c < hidden; c += tg_size)
        xs_scratch[c] = x[token * hidden + c];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each SIMD group handles ROWS_PER_SIMD rows of `inter`.
    const uint rows_per_simd = ROWS_PER_SIMD;
    const uint base_row = simd_id * rows_per_simd;

    for (uint r = 0; r < rows_per_simd; ++r) {
        const uint row = base_row + r;
        if (row >= inter) break;
        float g = 0, u = 0;
        const ulong w_base = (ulong)expert * inter * hidden + (ulong)row * hidden;
        // 32 lanes stride over hidden; each block of 32 consecutive
        // elements shares the same K-quant scale/bias — natural coalesce.
        for (uint c = simd_lane; c < hidden; c += 32) {
            float xv = xs_scratch[c];
            g += xv * qw36_load_scalar(expert_gate, gate_dtype, w_base + c);
            u += xv * qw36_load_scalar(expert_up,   up_dtype,   w_base + c);
        }
        g = simd_sum(g);
        u = simd_sum(u);
        if (simd_lane == 0)
            act[(token * k + slot) * inter + row] = silu(g) * u;
    }
}
```

Tunable constants:

| name | purpose | first value to try |
|------|---------|---------------------|
| `ROWS_PER_SIMD` | amortize per-block scale/bias reads across multiple rows | 4 |
| TG size | `N_SIMD × 32` — N_SIMD ≥ 2 to hide load latency | 128 (4 simds) |

## Why this should win on 35B-A3B

`gate_up` work per token = `k * inter * hidden = 8 * 512 * 2048 = 8.4M
multiplies`. With Q4_K weights at 4.5 bits/element and ~10 ops to
dequant a scalar:

- Naive kernel: 1 thread does the 8.4M iters serially.
  4096 outputs × ~25 µs/output = ~100 ms per token.
  Profile shows ~488 ms / 5 tokens = 97 ms/token. **Matches.**
- SwitchGLU + ROWS_PER_SIMD=4: 8 tokens × 8 experts × 128 SIMD
  blocks per inter ≈ 8192 threadgroups (one per `(token, slot,
  row-chunk)`). Each TG of 4 SIMDs computes 4×4=16 output rows
  cooperatively. ~32× cooperating lanes per output, ~4× row reuse
  → expected ~10ms/token → **30-40 tok/s on 35B-A3B**.

## What this rewrite explicitly is NOT

- Not a quant-format change. Expert weights stay as their on-disk
  Q4_K / Q5_K / Q6_K formats (or their `_AFFINE32` / `_SCALE16`
  repacked variants). No Q8 detour for activations.
- Not architecture-level. SwitchGLU is just a kernel; the engine's
  router + top-k + softmax stays where it is.
- Not Q4_0 KV. That's a separate task (`#84`) about KV cache
  precision, unrelated to expert matmul.
- Not multi-token. qw36 stays single-stream; SwitchGLU runs one
  token at a time. (Batching is `n_tokens > 1` in the prefill loop,
  already handled.)

## What CI will verify

- `tests/quant_fastest_smoke.sh` still passes on 0.8B (no MoE on
  0.8B, so this gates "didn't break the dense MLP path").
- `tests/correctness_greedy.sh` still passes 8/8 (KV configs + chat
  paths).
- 35B-A3B `Hello -n 8 --no-special` produces `,\n\nI am trying to
  use the` (codex's coherent CPU/Metal baseline from `59ebce1`).
- Per-kernel PERF trace shows `qw36_moe_switchglu_*` instead of the
  old `qw36_moe_gate_up_f32` / `qw36_moe_down_combine_f32`.
- 35B decode wallclock at least 4× the current ~5 tok/s.

## See also

- [`docs/architecture.md`](architecture.md) § "Per-token forward" — where
  the MoE block sits in the layer flow.
- [`docs/kv_quant_plan.md`](kv_quant_plan.md) — orthogonal: KV cache
  quantization (`Q8_KV` is shipped, Q4_KV deferred).
- [`docs/q4k_kernel_design_v2.md`](q4k_kernel_design_v2.md) — the
  Q4K-affine32 + qmv_fast kernel that unlocked the 0.8B path; same
  flavor of "operate on quant weight directly" approach.
- MLX source: `mlx-lm/models/switch_layers.py` and
  `mlx/c/quantized.cpp` for the gather_qmm primitive shape.
