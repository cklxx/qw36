# Environment-variable reference

Every `QW36_*` env knob the engine reads, with file:line and what it
does. The CLI consolidates the common ones under `--profile` /
`--fast` — see `qw36_metal --help` and `common/qw36_policy.c`.

Conventions:

- **stable**: documented, supported, part of the user API. Don't break.
- **internal**: documented here, used by the engine via `--profile`,
  but not really intended for direct user use.
- **research / debug**: opt-in, may go away. Useful for bisecting a
  perf or correctness issue.

## User-facing (stable)

| env | values | what |
|-----|--------|------|
| `QW36_PROFILE` | `reference` (default) / `fp16` / `lowmem` / `fast` | top-level backend policy. `reference` is precision-friendly; `fp16` MPS fp16 path; `lowmem` keeps quants on GPU; `fast` is the smoke-gated everything-on path. `common/qw36_policy.c`. |
| `QW36_METAL_FAST` | `1` (set by `--fast`) | umbrella implied by `--profile=fast`. Same shortcut. |
| `QW36_METAL_QUANT_GPU` | `0|1` | keep K-quant blocks on GPU, dispatch via per-row dequant matmul kernels. Saves ~1 GB on the 0.8B model. `common/qw36.c:880`. |
| `QW36_METAL_FP16_WEIGHTS` | `0|1` | materialize quant weights to fp16 before upload. Conflicts with `QW36_METAL_QUANT_GPU`. `common/qw36.c:1332`. |
| `QW36_METAL_FP16_KV` | `0|1` | fp16 K/V cache. Defaults on when `QW36_METAL_FP16_WEIGHTS` or `QW36_METAL_QUANT_GPU` is on. `common/qw36.c:1335`. |

## Profile-driven (internal)

Switched on by the policy module; you only set these directly when
bisecting which lever produced a regression. Default values shown
match the `fast` profile.

| env | default under `--fast` | what | source |
|-----|------------------------|------|--------|
| `QW36_METAL_Q4K_AFFINE32` | `1` | repack Q4_K weights into per-32 affine layout for faster qmv kernels. | `common/qw36.c:885` |
| `QW36_METAL_Q5K_AFFINE32` | `1` | same for Q5_K (gate_up is Q5_K in Q4_K_M). | `common/qw36.c:890` |
| `QW36_METAL_Q6K_SCALE16` | `1` | repack Q6_K into per-16 scale layout. lm_head and ffn_down are Q6_K. | `common/qw36.c:895` |
| `QW36_METAL_QUANT_GPU_LM_HEAD` | `1` | decouple tied lm_head, run as Q6K_SCALE16 on GPU. +23% sustained. | `common/qw36.c:900` |
| `QW36_METAL_Q4K_AFFINE32_MLX` | `1` (default-on, opt-out=`0`) | MLX-style qdot for Q4K affine32. ~+5% sustained essay. `metal/qw36_metal.m:1261`. |
| `QW36_METAL_Q5K_AFFINE32_MLX` | `0` | MLX-style qdot for Q5K. Wash on this host; kept for reference. |
| `QW36_METAL_Q6K_SCALE16_MLX` | `1` (default-on) | MLX-style qdot for Q6K scale16. +3-4% short, +6% at n=2048. |

## Long-context / KV layout

| env | default | what | source |
|-----|---------|------|--------|
| `QW36_METAL_KV_TRANSPOSED` | `0` (opt-in) | transpose K/V cache to `[head][dim][t]` so adjacent-t reads coalesce. +24% n=2048, -10% short. Task 2.4 will add `auto`. | `metal/qw36_metal.m:1748` |
| `QW36_METAL_ATTN_X4` | `0` (research) | x4-batched scoring inside `qw36_attn_decode_fused_f16kv_x4_f32`. ~0% on this host; kept as opt-in research artifact. | `metal/qw36_metal.m:1768` |

## DeltaNet research / debug

These exist for 35B-A3B bring-up and are opt-in. Default behavior
matches what produced coherent output on Qwen3.5-0.8B-Q4_K_M and the
coherent CPU 35B path (`,\n\nI am trying to use the`).

| env | what | source |
|-----|------|--------|
| `QW36_SKIP_DN` | skip all DeltaNet layers (vanilla-only forward). Used to attribute degeneracies to DN vs vanilla. | `common/qw36_attn_deltanet.c` |
| `QW36_SKIP_CONV1D` | skip the conv1d step in DN forward; useful for isolating the SSM state update from the depthwise conv. | `common/qw36_attn_deltanet.c:202` |
| `QW36_METAL_FUSE_DN_CONV` | toggle the fused conv1d+gated_delta Metal kernel. Defaults on. | `common/qw36_attn_deltanet.c:258` |
| `QW36_METAL_FUSE_DN_TAIL` | toggle the fused gated_rmsnorm+dn_out tail kernel. Defaults on. | `common/qw36_attn_deltanet.c:294` |
| `QW36_METAL_FUSE_QKV` | toggle the vanilla QKV concat-into-one-matmul fusion. Defaults on when fp16-weights + non-quant. | `common/qw36.c:935` |
| `QW36_METAL_FUSE_DN_QKVZAB` | toggle the DN 4-projection concat fusion. Defaults on under fp16 + non-quant. | `common/qw36.c:940` |
| `QW36_METAL_DN_TAIL_DIRECT` | dispatch DN tail (gated_rmsnorm + dn_out) directly without the fused kernel. Debug. | `metal/qw36_metal.m:2235` |

## Tracing / debug

| env | what | source |
|-----|------|--------|
| `QW36_DEBUG_LAYER` | per-layer `\|\|x\|\|` + first residual components before/after each block. Used to bisect the first divergent layer. | `common/qw36.c` |
| `QW36_MAX_LAYERS=<n>` | stop forward after the first `n` transformer layers. | `common/qw36.c:1596` |
| `QW36_BYPASS_LAYERS=<spec>` | bypass selected layer ids / ranges. | `common/qw36.c` |
| `QW36_TRACE_LAYER`, `QW36_TRACE_POS`, `QW36_TRACE_TAKE`, `QW36_TRACE_OUT` | dump intermediate tensors to a binary file. Used with `tools/diff_layers.py`. | `common/qw36_attn_vanilla.c:43-50` |
| `QW36_USE_MROPE_SECTIONS` | use the section-based mRoPE encoding (off by default; section info often missing/wrong in GGUF metadata). | `common/qw36.c:681` |
| `QW36_METAL_PERF` | enable the per-kernel `[metal-perf]` profile table at exit. Disables persistent compute encoder; wallclock under PERF is NOT representative. | `metal/qw36_metal.m:745` |
| `QW36_METAL_TIMING` | per-forward wallclock log (dev only). | |
| `QW36_DN_A_RAW` | skip the `logf(\|a\|)` transform on `ssm_a` weight load. Used during 35B bring-up; kept as a probe. | `common/qw36.c:858` |
| `QW36_DN_KH_MOD` | flip the K-head / V-head pairing from `v / group` to `v % n_key`. Bisection-only. | `common/qw36_attn_deltanet.c` |

## Removed knobs

The following used to exist but have been deleted; if you find one
in an old script, that's why it stopped working.

| env | removed in | reason |
|-----|-----------|--------|
| `QW36_METAL_F16_GEMV_QUAD` | predates `--fast` | superseded by the affine32 kernels. The MMA-based GEMV experiment was slower than MPS (`docs/q4k_qmv_quad_failed.md`). |

## Discovery

Until `--print-config` ships (task 3.3):

```bash
grep -rn "getenv\(\"QW36_" common/ metal/ | sort -u
```

is the canonical way to enumerate. The list above was generated from
that output on commit `4e7e07a`.
