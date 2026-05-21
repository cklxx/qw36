# KV cache quantization plan

The honest scoreboard for KV cache precision support in qw36, what's
shipping today, what's planned, and what each costs to implement.

## Today (2026-05-21)

| dtype | status | env knob                            | bytes/elem | who |
|-------|--------|-------------------------------------|------------|------|
| fp32  | ✓ default reference path            | (none — set by `dev_kv_dtype`)       | 4 | shipped |
| fp16  | ✓ default under `--fast` / `QUANT_GPU` | `QW36_METAL_FP16_KV=1` (auto)     | 2 | shipped |
| bf16  | ⏳ planned (#73 AB)                  | `QW36_METAL_BF16_KV=1`               | 2 | this doc's PR |
| Q8_0  | 🚧 designed below                    | `QW36_METAL_Q8_KV=1`                 | 1.06 | future task |
| Q4_0  | 🚧 sketched below                    | `QW36_METAL_Q4_KV=1`                 | 0.56 | future task |

"Bytes/elem" for Q8/Q4 includes the per-32-element fp16 scale block.

## Why fp16 KV isn't enough

fp16 has 5 exponent bits, 10 mantissa bits. K cache values come out of
RoPE-rotated Q·K dot products — typical magnitudes 10⁻¹ to 10¹. The
fp16 dynamic range covers them, but the mantissa precision can cost
1-2 token decisions per ~24 layers when the head dot products are
close. We hit this once already; see
[`docs/fp16_state_root_cause.md`](fp16_state_root_cause.md) — same
bisection methodology will apply to KV-side regressions.

bf16 trades mantissa for exponent: 8 exponent bits, 7 mantissa bits.
Same 2 bytes/elem; wider safe range; rougher locally. For long-context
attention (n > 2048) where K magnitudes vary more across positions,
this is the right tradeoff.

Q8_0 cuts memory roughly in half again with a per-32-element fp16
scale; the cost is per-read arithmetic. For 35B-A3B with KV bandwidth-
bound long context, this is the biggest practical win available.

Q4_0 is aggressive: half-precision again over Q8. Likely too lossy
for attention; ships behind a feature flag with explicit precision
caveats. We're including it in the plan so the file format is stable
when we get there.

## What "writing bf16" actually means in qw36

The KV cache buffer is allocated with a chosen `qw36_dtype`. Reads
go through `qw36_load_scalar(ptr, dtype, index)` — that helper already
handles F16, BF16, Q4K_AFFINE32, Q5K_AFFINE32, Q6K_SCALE16, Q8_0
(landed in commit `8db80b6`). So the SCORE and COMBINE paths in the
attention kernels can already read bf16 cells today.

The missing piece is **writing**. The current attention kernels store
`half(value)` at the cache offset (e.g. `metal/qw36_metal.metal:1437`).
For bf16 we need `bf16_pack(value)` — i.e. truncate the upper 16
bits of a fp32:

```metal
static inline ushort qw36_f32_to_bf16(float v) {
    // round-to-nearest-even; ties-away is fine for our use
    uint bits = as_type<uint>(v);
    uint rounding_bias = ((bits >> 16) & 1) + 0x7FFFu;
    return ushort((bits + rounding_bias) >> 16);
}
```

The kernel needs to know the dtype to choose between `half` store and
`ushort` store. Cleanest: parametrise the f16kv kernel as a "kv16"
kernel that branches on a `kv_dtype` constant. The variant name stays
`_f16kv_` for the fp16 path and a sibling `_bf16kv_` for the bf16
path until we converge on a unified `_kv16_` form.

Same applies to the f32 fallback kernels for when QW36_METAL_QUANT_GPU
is off.

## Allocator side (the easy half)

In `common/qw36.c:1380`:

```c
const qw36_dtype dev_kv_dtype = use_bf16_dev_kv ? QW36_DTYPE_BF16
                              : use_fp16_dev_kv ? QW36_DTYPE_F16
                              : QW36_DTYPE_F32;
const size_t dev_kv_elem_bytes = (dev_kv_dtype == QW36_DTYPE_F32) ? 4 : 2;
```

The `use_bf16_dev_kv` is decided from `QW36_METAL_BF16_KV` env. bf16
and fp16 are mutually exclusive — last one wins, with a `--doctor`
WARN when both are set.

## Q8_0 plan (cost ≈ 2-3 days)

The bytes-per-block layout is the same as GGUF's Q8_0 weight format:

```
struct kv_block_q8 {
    half     scale;          // 2 B
    int8_t   q[32];          // 32 B
};  // 34 B per 32 elements → 1.0625 B/elem
```

Write path (per token, per (head, dim_block)):
1. Compute `s = max(|v_i|) / 127` over the 32 lanes of this block.
2. Store `scale = half(s)`.
3. For each lane: `q_i = (int8_t)round(v_i / s)`.

This adds a 32-wide reduction inside the attention kernel's K/V write,
which is the new cost. It's not free but it's bounded — one reduction
per per-token per (head, block).

Read path: `qw36_load_scalar` is already ready (Q8_0 case is in there
since `8db80b6`).

The wrinkle: the cache buffer's seq dimension stride is no longer
`seq_capacity * head_dim * bytes_per_elem` — it's `seq_capacity *
(head_dim / 32) * sizeof(kv_block_q8)`. Both the transposed and
legacy layout helpers (`qw36_kv_cache_offset`) need a Q8 variant.

## Q4_0 plan (cost ≈ 4-5 days; precision risk)

Per-32-element block with a half scale and 16 packed bytes:

```
struct kv_block_q4 {
    half     scale;
    uint8_t  q[16];          // two 4-bit nibbles per byte
};  // 18 B per 32 elements → 0.5625 B/elem
```

Write: 32-wide reduction for scale; nibble-pack each pair.
Read: per-lane shift + mask + scale × signed value.

Q4 KV is precision-risky enough that we'd need a 5-layer / 10-layer
bisection of layer outputs vs CPU reference before defaulting it on
anything. Likely shipped as opt-in research with an explicit caveat.

## Where each writes itself into the codebase

- `common/qw36.c` — env-knob decode + `dev_kv_dtype` selection.
- `metal/qw36_metal.m` — new dispatch branches for bf16/Q8/Q4 in
  the metal_attention site (lines around 1640–1860).
- `metal/qw36_metal.metal` — new kernel variants (`_bf16kv_f32`,
  `_q8kv_f32`, `_q4kv_f32`) plus their x4 batched siblings. Q8/Q4
  also need new offset helpers for the block-strided layout.
- `common/qw36.h` — `qw36_dtype` already includes BF16; add KV
  block formats if we choose them.
- `docs/env_knobs.md` — list each new knob, mark lifecycle.
- `docs/architecture.md` § KV cache — link to this doc.
- `tests/golden_kernels.sh` — extend to cover the new packed
  formats once they land.

## Why not just adopt llama.cpp's KV-Q8 format?

llama.cpp uses GGUF Q8_0 for KV with the same shape we're proposing.
Adopting their exact byte layout would let us share fixtures with
their CI. We're doing the same per-32-block + half-scale split here
so the file format actually matches.

(For Q4 the picture's hazier — llama.cpp tried multiple layouts.
We'll pick whichever produces the smallest perplexity hit at 32-block
granularity.)

## What this doc deliberately doesn't promise

- A specific tok/s number. KV is bandwidth-bound at long context;
  bf16 won't change wallclock vs fp16 (same bytes), Q8 should help
  at n=2048+, Q4 should help more — but the bench data lands with
  each PR, not with this doc.
- A timeline. bf16 is small (1-2 days codex brief). Q8 is several
  days of careful kernel work. Q4 is research.

## Next concrete steps

1. **Implement bf16 KV** — `QW36_METAL_BF16_KV=1` opt-in. New kernel
   variant `_bf16kv_f32` + dispatch. Smoke against fp16 baseline for
   regression; smoke against fp32 reference for correctness. (#73 AB)
2. **Implement Q8_0 KV** — kernel variant + block-stride offset
   helpers + write-side reduction. Likely a codex brief. (new task)
3. **Implement Q4_0 KV** — research first; bisection harness before
   default. (new task)
4. **Cross-backend support** — port to CUDA/AMD once Metal lands.
   (folded into Roadmap theme 1.2)
