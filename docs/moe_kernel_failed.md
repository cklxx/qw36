# MoE GPU kernel — two failed rewrites before SwitchGLU

Two dead-end attempts to optimize the MoE kernels on 35B-A3B,
documented so the next person who looks at this surface doesn't
re-try them. The actual winning design is in
[`docs/moe_design.md`](moe_design.md) (MLX-style SwitchGLU +
gather_qmm operating on Q4/Q5/Q6 expert weights directly).

The baseline this both attempted to beat:

```
qw36_moe_gate_up_f32          488 ms / 5 tokens   (2.4 ms / call avg)
qw36_moe_down_combine_f32     531 ms / 5 tokens   (2.7 ms / call avg)
qw36_moe_route_f32            356 ms / 5 tokens   (1.8 ms / call avg)
```

= 79% of total GPU time on 35B-A3B `--fast` decode. Decoding ~5 tok/s.

## Attempt 1 — 1-row-per-threadgroup simdgroup variant

**Idea (`qw36_moe_gate_up_tg_f32` / `qw36_moe_down_combine_tg_f32`):**
replace each 1-thread-per-output kernel with one threadgroup per
output, 32 lanes inside cooperating via `simd_sum` on the `hidden`
dot product. Standard GEMV tiling.

```metal
kernel void qw36_moe_gate_up_tg_f32(
    ...,
    uint gid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    /* one TG per (token, inter_row) output */
    float g = 0, u = 0;
    for (uint c = lane; c < hidden; c += 32) {
        float xv = x[c];
        g += xv * qw36_load_scalar(expert_gate, gate_dtype, base + c);
        u += xv * qw36_load_scalar(expert_up,   up_dtype,   base + c);
    }
    g = simd_sum(g);
    u = simd_sum(u);
    if (lane == 0) act[gid] = silu(g) * u;
}
```

Dispatch: `MTLSizeMake(top_k * inter, 1, 1)` threadgroups × `MTLSizeMake(32, 1, 1)`.

**Result:**

```
35B-A3B Hello -n 8 --no-special:
  baseline (default kernel)        ≈ 5 tok/s
  MoE TG (this attempt)            ≈ 0.2 tok/s         ← 24× slower
```

Output correctness: matched the baseline. Pure perf regression.

**Root cause.** Per-token MoE work = `top_k * inter = 4096` outputs.
The original 1-thread-per-output kernel dispatched 4096 threads
across roughly 16 threadgroups (`metal_dispatch_1d` picks a TG size
of ~256), 1 dispatch wave.

The TG variant dispatched **4096 separate threadgroups** of 32
threads each. Each TG pays:

- A separate launch overhead from the command encoder
- Initial cache miss on `x[]` (now read by 32 lanes × 4096 TGs
  rather than 4096 single threads sharing residency)
- A `simd_sum` reduction every output row

For a small inner loop (`hidden = 2048`), the parallelism win from
32 lanes was completely eaten by the TG launch overhead and
re-traversal of the input vector.

**Lesson.** Cooperating lanes per output is the right shape — but
ONE output per TG is wrong for this size class. Pack multiple
outputs (and ideally multiple rows of weight) per TG so that the
shared `x[]` and the shared K-quant scale/bias get amortized across
both lanes AND rows. This is the SwitchGLU `ROWS_PER_SIMD` knob in
the eventual rewrite.

**Decision.** Reverted before push. Code never landed. Lesson + this
doc are the artefacts.

## Attempt 2 — Q8-only fast path

**Idea (`qw36_moe_gate_up_q8_f32`):** specialize the kernel for the
case where expert weights happen to be Q8_0. Re-quantize activations
to Q8 too. Use a fused int8 × int8 → fp16 dot product.

**Why this looked attractive.** Q8 quant matmul has a clean `int8 ·
int8` SIMD intrinsic on most GPUs. The expert weights on 35B-A3B
*could* be repacked to Q8 at load time, and the activations are
naturally fp32 / fp16.

**Result.** Never benched at full speed because of the design flaw
below; abandoned during the rewrite.

**Root cause / why it was the wrong direction.** 35B-A3B's actual
on-disk expert dtypes are:

- `ffn_gate_exps` / `ffn_up_exps`: **Q4_K** (4-bit, per-32-element
  block scale + bias)
- `ffn_down_exps`: **Q5_K** (5-bit, similar)

Repacking these to Q8_0 means **decompressing 4-bit → 8-bit**: same
information, doubled storage. That doesn't unlock any compute win
(8-bit · 8-bit is no cheaper per byte than 4-bit · fp16 with
inline dequant) and it doubles the bytes read from DRAM — the
opposite of what an MoE optimization should do.

The real lesson is that **the kernel should operate on the weight's
NATIVE quant dtype**, not transform it. That's what
`Q4K_AFFINE32` does on the 0.8B dense path (`docs/q4k_kernel_design_v2.md`)
and what SwitchGLU should do on the 35B MoE path.

A Q8 activation × Q8 weight kernel would be valuable for models
that genuinely ship Q8 expert weights (none in qw36's target list)
or as part of a draft-model speculative-decode path. For mainline
35B-A3B it's strictly worse than dequanting Q4/Q5 inline.

**Decision.** Deleted during the SwitchGLU design pass. The kernel
and dispatch wiring did not land on `main`.

## What replaced both

`docs/moe_design.md` — MLX-style SwitchGLU:

1. Router → top-k → only access the selected expert rows (no
   serial sweep over 256 experts).
2. `gather_qmm` operating on the **on-disk** weight dtype
   (Q4_K_AFFINE32 / Q5K_AFFINE32 / Q6K_SCALE16), inline dequant
   per element.
3. One SIMD group computes `ROWS_PER_SIMD` (typical 4) consecutive
   output rows cooperatively — amortizes per-block scale/bias reads
   across both lanes AND rows.

Expected: **~10ms/token MoE on 35B → 30-40 tok/s** decode (vs
current ~5 tok/s).

## Related failed-experiment docs

- [`docs/q4k_qmv_quad_failed.md`](q4k_qmv_quad_failed.md) — first
  Q4_K qmv_quad port was 29 tok/s vs 58 baseline; the layout
  assumption was wrong. Same flavor of "naive port from MLX, didn't
  account for our actual data layout."
- [`docs/fp16_state_root_cause.md`](fp16_state_root_cause.md) —
  fp16 state buffer accumulation drift; bisection methodology.
- See also `FINAL_STATUS.md` § "Optimisations we tried that did not
  beat MPS on this host" — the running ledger.
