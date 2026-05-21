# Codex brief: bf16 KV cache Metal kernel support (task #73 AB)

## Context

`QW36_METAL_BF16_KV=1` is acknowledged in `common/qw36.c` but falls
back to fp16 because the Metal attention kernels hardcode `half`
stores and `half *` reads against the KV buffer. The design is
locked in `docs/kv_quant_plan.md`.

`qw36_load_scalar` already handles `QW36_DTYPE_BF16` (commit
`8db80b6`) so the read path is half-built. The bf16-pack helper for
the write side does not exist yet.

## What "done" looks like

`QW36_METAL_BF16_KV=1` on Qwen3.5-0.8B-Q4_K_M (`--fast`) produces
coherent output that matches CPU within standard rtol — the smoke
`Hello! How can I help you today?` reproduces — and the engine
prints `qw36: KV cache dtype = bf16` (replacing the current
"acknowledged but not yet shipped" warning).

`tests/quant_fastest_smoke.sh` still passes (default fp16 KV path).

Bench wallclock vs fp16 KV (3-rep median, load < 3) at n=2048 essay;
expectation: parity with fp16 since bytes/elem is identical, but
better numerical stability over long context. Numbers go in the
commit body per `docs/performance_methodology.md`.

## Scope

`metal/qw36_metal.metal`:

1. Add `qw36_f32_to_bf16(float v) → ushort` helper:
   ```metal
   static inline ushort qw36_f32_to_bf16(float v) {
       uint bits = as_type<uint>(v);
       uint rounding_bias = ((bits >> 16) & 1) + 0x7FFFu;
       return ushort((bits + rounding_bias) >> 16);
   }
   ```
2. Branch the K/V store sites in the f16kv kernels by a new
   `kv_dtype` constant. Sites (current main):
   - `qw36_attn_decode_fused_f16kv_f32` lines ~1437-1438 (write)
     and ~1457-1485, ~1503-1533 (reads)
   - `qw36_attn_decode_fused_f16kv_x4_f32` sites ~1661-1662 (write)
     and similar reads
3. When `kv_dtype == BF16` (constant arg from host), the write does
   `((device ushort*)k_cache)[off] = qw36_f32_to_bf16(kv)` and the
   read does `qw36_load_scalar(k_cache, BF16, off)` (already in tree).

`metal/qw36_metal.m`:

1. Add a `kv_dtype` byte buffer to the attention dispatch (lines
   ~1819+) so the kernel knows whether to bf16-pack or half-cast.
2. Keep the pipeline name `_f16kv_` — the kernel is now technically
   "kv16", same byte count, branch on dtype. Or split into a sibling
   `_bf16kv_` pipeline; the dispatcher's `if (k_cache->dtype == BF16)`
   selects.

`common/qw36.c`:

1. Replace the "acknowledged but not yet shipped" warning with the
   coherent enable-message:
   ```
   qw36: KV cache dtype = bf16 (opt-in via QW36_METAL_BF16_KV;
   wider exponent range than fp16, same 2 bytes/elem)
   ```
2. Switch `dev_kv_dtype = use_bf16_dev_kv ? QW36_DTYPE_BF16 : …` —
   the env-knob plumbing is already there; just flip the branch
   when the kernel is ready.

## Acceptance gates

1. `make check` green (kvcache + 0.8B smoke + CLI + goldens).
2. `QW36_METAL_BF16_KV=1 ./qw36_metal --fast -m <0.8B> -p Hello -n 16`
   produces a coherent Hello! reply.
3. `QW36_METAL_BF16_KV=1 ./qw36_metal --fast … -n 2048 essay`
   wallclock 3-rep median within ±5% of fp16 KV baseline.
4. Commit body carries the bench data + load avg + N-rep.

Out of scope here: Q8 / Q4 KV (separate tasks #83, #84). The bf16
landing should NOT touch the `qw36_kv_cache_offset` helper since
bf16 is the same byte count as fp16; Q8/Q4 will add block-stride
offset variants in their own PRs.

## Why now

bf16 ships first because it is the smallest delta (same bytes/elem,
no offset helper changes). Q8/Q4 require new offset helpers + write-
side reductions; landing bf16 first establishes the `kv_dtype`
dispatch plumbing those will reuse.
