# Codex brief: Q8_0 KV cache (task #83)

## Context

bf16 KV just landed (codex, task #73 AB). The next quantization step
is Q8_0 — 8-bit with a per-32-element fp16 scale, matching GGUF's Q8_0
weight format. Halves KV bandwidth vs fp16 → biggest practical win for
long-context attention on 35B-A3B. Full design in
[`docs/kv_quant_plan.md`](../kv_quant_plan.md).

## What "done" looks like

- `QW36_METAL_Q8_KV=1` opt-in. With it set, KV cache is allocated as
  per-32-block Q8_0 layout and the engine prints
  `qw36: KV cache dtype = Q8_0` instead of the bf16/fp16 message.
- 0.8B `Hello! How can I help you today?` reproduces.
- 35B-A3B 8-token coherent output reproduces.
- n=2048 wallclock improves vs fp16/bf16 (expected ~20-30% on a
  bandwidth-bound shape). Numbers in the commit body.
- `tests/quant_fastest_smoke.sh` + `tests/cli_smoke.sh` +
  `tests/golden_kernels.sh` all green.
- An entry in `tests/perf_baseline.json` for `q8kv:2048` so the gate
  catches regressions.

## Block layout (GGUF-compatible)

```
struct kv_block_q8 {           // 34 bytes per 32 elements → 1.0625 B/elem
    half     scale;             // 2 B
    int8_t   q[32];             // 32 B
};
```

Cache buffer is `seq_capacity * (head_dim / 32) * 34` bytes per
(layer, kv-head) under legacy layout, or shape-rotated under
transposed.

## Scope

`common/qw36.h`:
- Add `QW36_DTYPE_KVQ8` (or reuse `QW36_DTYPE_Q8_0` — they have the
  same byte format). Reusing is cleaner; it just means a buffer that
  happens to be Q8_0-laid-out is allowed as a KV cache too.

`common/qw36.c`:
- Decode `QW36_METAL_Q8_KV` env knob, set `dev_kv_dtype` and adjust
  `dev_kv_elem_bytes` (no longer 2 or 4 — it's 34 per 32 elems, so
  the alloc math becomes `seq_capacity * (head_dim/32) * 34 *
  n_kv_heads`).
- Add the precedence rule: `Q8 > BF16 > FP16 > FP32`, with a
  `--doctor` WARN if multiple are set.

`metal/qw36_metal.metal`:
- Add `qw36_q8_block_pack(thread float *vals, device kv_block_q8 *)`
  helper. Per-32-element block:
  ```metal
  float s = 0.f;
  for (uint i = 0; i < 32u; i++) s = fmax(s, fabs(vals[i]));
  half scale = half(s / 127.0f);
  // store: scale + int8 quantized lanes
  ```
  Use a simdgroup reduction (`simd_max(fabs(v))`) when the kernel's
  threadgroup geometry permits — head_dim=128 is 4 blocks per kv-head,
  per dimension; lanes can cooperate per block.
- Add a `_q8kv_f32` variant of `qw36_attn_decode_fused_*`. Read path
  goes through `qw36_load_scalar(ptr, QW36_DTYPE_Q8_0, idx)` which is
  already in tree. Write path uses the new block-pack helper.
- Add `qw36_kv_cache_q8_offset(...)` helper for block-strided
  addressing. Both legacy and transposed variants:
  - Legacy:    `t * n_kv * (head_dim/32) + kvh * (head_dim/32) + (dim/32)`
              block index → multiply by 34 for byte offset
  - Transposed: `kvh * (head_dim/32) * seq + (dim/32) * seq + t`

`metal/qw36_metal.m`:
- Pipeline registration for `_q8kv_f32` + x4 variant.
- Dispatch route based on `k_cache->dtype == QW36_DTYPE_Q8_0`.
- `kv_dtype` constant fed to the unified KV kernel (or use a separate
  pipeline; whichever codex prefers).
- Update KV append site to choose between half-store / bf16-pack /
  Q8-block-pack.

`tools/gen_goldens.c` (optional, can be follow-up):
- Add a `kv_block_q8` golden so cross-backend ports have a fixture.

`docs/env_knobs.md` + `docs/kv_quant_plan.md` + `README.md` table:
- Mark Q8_0 as shipped, document the env knob, link this brief.

## Acceptance gates

1. `tests/quant_fastest_smoke.sh` green (default fp16 KV path unchanged).
2. `QW36_METAL_Q8_KV=1 ./qw36_metal --fast -m <0.8B> -p Hello -n 16`
   coherent.
3. `QW36_METAL_Q8_KV=1 ./qw36_metal --fast -m <35B> -p Hello -n 8 --no-special`
   matches CPU reference output (or has a documented numeric tolerance).
4. n=2048 wallclock: Q8 KV should win vs fp16 KV by 15-30%. Bench in
   commit body (5-rep median, load < 3).
5. `tests/perf_baseline.json` gets a new `q8kv:2048` row.

## Common traps

- **Block alignment.** head_dim must be a multiple of 32. Qwen3.5
  head_dim=256 ✓, Qwen3.6 head_dim=256 ✓, DN key_dim=128 ✓. If a
  future model has head_dim=80, this kernel rejects with a clear
  error rather than silently slicing.
- **Scale=0.** A KV block can come out all-zero (e.g. attention to
  padding before EOS). Guard the divide: `inv_scale = scale > 1e-30 ?
  1/scale : 0`. Store scale=0 and q=[0]*32; on read, the multiply
  produces 0 which is correct.
- **Saturating cast.** `(int8_t)round(v / s)` can overflow when v ≈
  s * 127.5. Saturate to ±127.
- **Transposed layout block offset math.** The block index is on
  `dim`, not on `t`. Transposed has `t` innermost, so the block
  spans 32 t-elements? Actually no — the block is along the dim
  axis whether layout is legacy or transposed; the difference is
  whether the block array steps along `t` (legacy) or stays put on
  `t` (transposed). Sanity-check this against a written-then-read
  fixture before declaring the kernel correct.

## Why now

Q8_0 KV is the #1 lever for long-context (n > 2048) wallclock left
on the table that doesn't depend on a flash-attn rewrite. Halving
bytes per KV read directly translates to a bandwidth win on
bandwidth-bound attention. Q4 KV (#84) is the next step beyond this
but has precision risk and needs a bisection harness; Q8 ships first.
