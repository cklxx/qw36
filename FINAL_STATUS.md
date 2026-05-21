# qw36 — Final Status

## TL;DR

Pure-C Qwen 3.5/3.6 inference framework, 3 GPU backends, **~204 tok/s short
/ 176 tok/s sustained / 92 tok/s at n=2048** on Qwen3.5-0.8B-Q4_K_M via Metal
under `QW36_METAL_QUANT_GPU=1 QW36_METAL_FAST=1` (vs llama.cpp 170 tok/s
reference, MLX 244 tok/s reference, ~84% of MLX). The fastest path is opt-in
and smoke-gated by `tests/quant_fastest_smoke.sh`, not fp32 bit-equal. Past
the 200 tok/s target. The fp16 MPS path tops out at 119/103 tok/s — quant-
path wins because we halve bandwidth via affine repack on layers AND lm_head,
and the MLX-style Q6K qdot saves another 3-4% over the bit-shift unpack.
CPU baseline 1.7 tok/s.

## Decode throughput ladder (Metal, M-class GPU, Qwen3.5-0.8B-Q4_K_M)

| commit  | tok/s decode (n=128) | what                              |
|---------|---------------------:|-----------------------------------|
| baseline CPU       | 1.7    | lazy quant + scalar matmul        |
| 218fb50            | 11     | first GPU matmul (MPS gemv)       |
| d24bbbd            | 55     | persistent GPU state buffers      |
| 2cd9da4 + ce9343e  | 56     | bucket pool + fused gated_delta_step |
| e286627 (fp16 opt-in) | 67  | fp16 weight materialize           |
| codex task #30     | 75-81  | fused decode attention            |
| 53511dd            | 82     | fp16 KV cache, sustained          |
| 14b1a06            | 83     | Metal Q-gate fused (back from 27 after #31 fix) |
| e28726d            | 40 (opt-in) | GGUF GPU-native quant matmul (low-mem) |
| 564eee4            | 50 (opt-in) | Q4_K sub-block scale cache         |
| 9d06231            | 84     | xh f32→f16 dedup                  |
| b63285b            | 84     | MPS weight matrix wrapper cache    |
| 720ac5d            | 85     | xh cache fp16-input invalidate    |
| bdf42ad            | 90     | vanilla fused residual+rmsnorm   |
| b489b09            | 100    | DN fused residual+rmsnorm        |
| 5b59ef8            | 115    | post-MLP→next-input fused        |
| 35df347            | 117    | vanilla QKV concat fusion        |
| 21cedec            | 120    | DN 4-projection (qkv+z+a+b) concat |
| 0ba7349            | 121    | DN gated rmsnorm tail fuse      |
| aac7f50            | 121    | DN conv1d + gated_delta fuse     |
| 112e85f            | **122** | **persistent compute encoder** (fp16 path peak) |
| 766b7c0 + 90b1377  | 85 (opt-in) | Q4K → affine32 repack + qmv_fast kernel (Q4K only; still under fp16) |
| 8d45cca + b5e0ecb  | **161 (opt-in)** | + Q5K → affine32 repack (gate_up is Q5K — dominant matmul class) |
| b5e0ecb (triple)   | **170 short / 139 sustained** | + Q6K → scale16 (qkv = Q6K). Triple-affine through fp16 ceiling |
| 743a158 (triple + lm_head) | **208 peak / 185 avg short / 176 sustained** | + lm_head Q6K_SCALE16 (decouple tied embed alias). 200 tok/s target hit. |
| 2464023 (Q6K_MLX default) | **~204 stable short / 176 sustained / 92 at n=2048** | + MLX bit-trick qdot for Q6K (lm_head + ffn_down). +3-4% short, +6% at n=2048 |
| (llama.cpp ref)    | 170    | upstream baseline                 |
| (agent-infer ref)  | ~244   | MLX bf16 + custom Q4_K + compiled fused kernels |

Per-layer cost is ~245us (consistent across MAX_LAYERS=N benches), so 24
layers cost ~5.88ms + 2.46ms (embed + lm_head + sample) ≈ 8.34ms / token
= ~120 tok/s.

### Theoretical ceiling reached for the fp16 path

The 0.8B model materialised to fp16 is ~1.6 GB resident weights. Reading
every weight once per token at the ~200 GB/s peak unified-memory bandwidth
of this M-class GPU bottoms out at **8 ms / token = 125 tok/s**. Our
8.34 ms = 96% of that ceiling. lm_head alone is 1.97 ms (MPS) vs 1.55 ms
theoretical bw limit, i.e. 78 % bw efficient — and similar story per
MLP / attention GEMV.

### Why MLX gets 244 tok/s and we don't — and what we just unlocked

MLX-4bit reads ~0.4 GB per token (1.6 GB fp16 → ~0.4 GB at 4 bits), giving
a 2 ms / token bw ceiling = ~500 tok/s. They get 244 = 49% of that with
their hand-tuned `qmv_quad` quantised matmul.

**Triple-affine repack unlocked the quant-path advantage (2026-05-21):**
gate_up is Q5_K (the heaviest matmul class in Q4_K_M); we'd been
testing Q4_K-only affine32 (worth +45% over native Q4_K but still slower
than fp16 MPS because gate_up was unaffected). Adding Q5K_AFFINE32 +
Q6K_SCALE16 jumps us from 85 → 170 short / 139 sustained, beating fp16
MPS (119 short / 103 sustained) by 30-40%. Bandwidth math: ~0.5 GB at
4-5 bits vs ~1.6 GB fp16 explains the win.

The qmv_fast kernels (codex S2 + Q5K/Q6K follow-on) match MLX's qmv_fast
pattern (uint16 packed reads, per-group scale+bias). MLX's qmv_quad
geometry (4-lane simdgroup quad) didn't help — see `docs/q4k_qmv_quad_failed.md`.

### Optimisations we tried that did not beat MPS on this host

| experiment                              | result          |
|-----------------------------------------|-----------------|
| `qmv_quad`-style fp16 GEMV (codex G)    | slower than MPS |
| Custom fp16 GEMV (single TG/row)        | slower than MPS |
| MMA-based fp16 GEMV (codex Q)           | M=1 wastes 7/8 of the 8×8 tile; 4.16 ms vs MPS 1.97 ms on lm_head |
| `QW36_METAL_F16_GEMV_QUAD` opt-in       | left in tree as `0` default |
| Fused silu_mul + down_proj custom GEMV  | wash with MPS+silu_mul    |
| fp16 residual state (x_dev / x_rms fp16) | step-0 logit drift; left as opt-in |
| persistent compute encoder (codex O)    | minimal — CPU encode already <0.1% of budget |
| Q4K → affine32 + qmv_fast kernel (S2)   | +45% over native Q4_K quant_gpu (57→85 tok/s) but still under fp16 MPS ceiling (115). Kept as opt-in: `QW36_METAL_QUANT_GPU=1 QW36_METAL_Q4K_AFFINE32=1`. Sanity passes (max_abs 1e-4 vs Q4_K original). |
| MLX-style qmv variant for Q4K affine32 (8d45cca) | Slower than codex's variant on n=256 (78 vs 87 tok/s). MLX pre-scales x_thread but TG geometry (4×4 rows × 16 vpt) loses to codex's per-row TG on our shape. Kept as `QW36_METAL_Q4K_AFFINE32_MLX=1` opt-in for future revisit. |

`QW36_METAL_FP16_WEIGHTS=1` to opt in to fp16 weights (default fp32 keeps
precision_cpu_vs_metal.sh byte-equal).

**Fastest path today (opt-in, smoke-gated):**
```
QW36_METAL_QUANT_GPU=1 \
QW36_METAL_Q4K_AFFINE32=1 \
QW36_METAL_Q5K_AFFINE32=1 \
QW36_METAL_Q6K_SCALE16=1 \
QW36_METAL_QUANT_GPU_LM_HEAD=1 \
./qw36_metal -m <gguf> -p "Hello"
```
→ 208 tok/s peak short / 176 sustained.

Correctness gate:
```sh
./tests/quant_fastest_smoke.sh <gguf>
```

**Long-context scaling (full FAST=1 path, Q6K_MLX default-on):**

| n | initial | after fp16 KV | after Q6K_MLX | ms/token |
|---|------:|------:|------:|---------:|
| 64   | 185 | 194 | 203 | 4.9 |
| 256  | 176 | 168 | 176 | 5.7 |
| 512  | 138 | 148 | 155 | 6.5 |
| 1024 | 111 | 121 | 126 | 7.9 |
| 2048 | 68.6 | 87.3 | **92.5** | 10.8 |

Three compound levers got us here:
1. fp16 KV under quant_gpu (commit 6619ac8) — halves attention bandwidth
   at long context. n=2048 +27%.
2. lm_head Q6K_SCALE16 (commit 743a158) — 3-4ms/token saved across all
   contexts.
3. MLX Q6K qdot (commit 7550375 + default-on 2464023) — adds +3-4% short
   and +6% at n=2048 (lm_head fraction grows at long context).

Attention is still O(seq). Flash-attention-style streaming pass is the next
lever for n >= 2048.

## Coverage

### What works
- GGUF v3 loader (mmap, kv table, tensor table, alignment) — `common/qw36_gguf.c`
- Dequant for F32 / F16 / BF16 / Q8_0 / Q2_K / Q3_K / Q4_K / Q5_K / Q6_K
  with per-block layouts byte-equivalent to ggml-quants.c (verified
  against a hand-rolled Python reference for Q6_K embeddings — see
  `tools/dump_tensor.c` + commit 5afd192)
- BBPE tokenizer with Qwen3 chat-template special token detection
  (`<|im_start|>`, `<|im_end|>`, etc.)
- mRoPE handling (off by default; section-based opt-in via
  `QW36_USE_MROPE_SECTIONS=1`)
- Per-layer hybrid attention dispatch (vanilla GQA vs Gated DeltaNet)
- MoE forward (router + top-k experts + shared) on CPU and Metal
- 3 GPU backends behind one frozen vtable (`common/qw36_gpu.h`):
  Metal (end-to-end), CUDA (built off-host), AMD/HIP (built off-host)
- GPU-resident state: x / x_rms / Q / KV cache (fp16) / scratch / logits
  all live on device; only the final logits download per token
- Fused Metal decode-attention kernel (q/k norm + RoPE + KV append +
  score + softmax + combine in one MSL dispatch per vanilla layer)
- Fused Gated DeltaNet kernel (transliterated from
  `agent-infer/crates/mlx-sys/src/mlx_qwen35_model.cpp:203-275`)
- Buffer pool, MPS-gemv pipeline-state cache, batched command buffer
  per forward step

### CLI / API
- `qw36_cpu`, `qw36_metal`, `qw36_cuda`, `qw36_amd` binaries
- `--info`, `--dump-tokens`, `--debug-top`, `--no-special`, `--interactive`
- Env knobs: `QW36_METAL_FP16_WEIGHTS`, `QW36_DEBUG_LAYER`,
  `QW36_SKIP_DN`, `QW36_SKIP_CONV1D`, `QW36_ROPE_NEOX`,
  `QW36_USE_MROPE_SECTIONS`

### Tests + ops
- `make test`: precision_cpu_vs_metal (step-0 bit-equal, fp32) +
  e2e_qwen35_smoke (informational; currently FAIL — see #31)
- `qw36_dump_tensor` standalone cross-check tool
- `.github/workflows/ci.yml`: 3-job matrix (linux-cpu / macos-metal /
  cuda-build-check) on push + PR
- `.github/workflows/release.yml`: builds linux-x86_64 + macos-arm64
  artifacts with sha256 sidecars on `v*.*.*` tags
- `tools/install.sh`: one-liner installer, detects host, prefers `gh
  release download`, verifies sha256
- MIT LICENSE, CHANGELOG.md (v0.1.0 entry), top-level README (300 lines)

## Open items

### #31 — output coherence — **partial fix landed, more remaining**

#### Progress

| step                              | sample output (`Hello`, n=8)           |
|-----------------------------------|----------------------------------------|
| start of session                  | `$h$h$h:$:$:$:$:$`  (pure punctuation) |
| Q5_K dequant fix (c479cad)        | `ól>$;$;$;$`  (mixed, still degenerate)|
| Q3_K + matmul checks (2df5298+)   | `($($($($($($($($` (looping cluster)   |
| **QKV interleave fix (b25b124)**  | `old dispersionemenogaellini…`         |
| Metal sync (24661fc)              | same multilingual fragments at 78 tok/s|

**Real bug found and fixed**: Qwen3.5/3.6 store `attn_qkv.weight` outputs
per-head interleaved `[Qh0(Dk), Kh0(Dk), Vh0(Dv), Qh1, Kh1, Vh1, ...]`,
not as the `[Q_block | K_block | V_block]` layout that agent-infer's
MLX path reaches at split-time (it must reorder at load). With the
correct layout:

- bisection `L=1` logit on `\n` jumps from 7.68 → 40.66 (DN L0 no
  longer destroys the residual)
- top-1 token at full L=24 changes from `{$` to `old`/`雨`/`老` — i.e.
  real word tokens, multilingual mix
- speed unchanged (~78 tok/s with fp16)

Still not coherent though. Need agent-infer / HF transformers
reference dump (see tests/golden_diff.md) to chase the residual
defect — likely in:
- vanilla q/k/v split inside the per-head reshape (similar mistake
  possible since L=4 transforms `\n` directly into `万` instead of an
  English continuation token, despite English prompt)
- conv1d weight stride / state-shift direction
- ssm_out row/col interpretation
- z gating broadcast across value heads

### #31 — output coherence — **localized to DN block math** (commit b709cb7)

Layer-bisection with `QW36_MAX_LAYERS=N` running prompt `Hello`:

| N  | top-1                     | comment |
|----|---------------------------|---------|
| 0  | `\n` logit 120 (input)    | bypass; tied embed self-correlation |
| 1  | `zé`, logit 7.68          | **DN layer 0 destroys the residual direction** |
| 8  | `WER`, logit 13.4         | drift continues |
| 24 | `{$`, logit 12.1          | final degenerate cluster |

Cross-check: `QW36_SKIP_DN=1 ./qw36_cpu -p Hello` (zero DN contribution,
only 6 vanilla layers contribute) → top-1 `\n` (id 198) at logit 12.5.
**Vanilla-only path is sensible.**

Bug is inside `gated_delta_decode` (CPU `common/qw36.c:715`) and
`qw36_gated_delta_step_f32` (Metal `metal/qw36_metal.metal`). Both
backends produce identical degenerate output → shared algorithmic
defect. Likely candidates, ranked:

1. **DN state update axis swap** — agent-infer's state layout is
   `[Hv, Dv, Dk]` (Dk innermost); ours is `[Hv, Dk, Dv]` (Dv innermost).
   Self-consistent inside our impl but the layout difference matters if
   any reader (e.g. gated rmsnorm input) expects the agent-infer order.
2. **Per-head Q/K split inside attn_qkv** — our split is
   `q[0:q_dim] | k[q_dim:q_dim+k_dim] | v[…]`. If GGUF stores
   `[q_h0, k_h0, v_h0, …]` interleaved per head, our q is corrupt.
3. **`ssm_out.weight` row/col interpretation** — if GGUF stores ssm_out
   transposed, projection axis is flipped.
4. **`silu(z) * rmsnorm(gout)` z indexing** — if z is per-head but we
   broadcast across all values.

### #31 — symptoms (legacy)
On the same Qwen3.5-0.8B-Q4_K_M.gguf:

  - llama.cpp `Hello` → `Hello! How can I help you today?`  (170 tok/s)
  - qw36       `Hello` → `($($($($($($($($`                  ( 82 tok/s)

Both CPU and Metal in qw36 produce identical step-0 logits and agree
on argmax for several steps. The forward math itself is therefore
internally consistent but differs from the ggml reference somewhere.

Confirmed *not* the bug source:
- Q4_K/Q5_K/Q6_K dequant blocks (byte-equal to Python reference)
- Embedding row addressing (byte-equal for token 9419 'Hello')
- Tokenizer (same token count + IDs as llama.cpp for chat-wrapped prompt)
- GQA mapping (`kvh = h * n_kv / n_heads`, matches agent-infer)
- num_attention_heads derived from tensor shape (metadata is wrong)
- mRoPE sections (disabled by default; agent-infer also ignores them)

Remaining suspects (see `tests/correctness_diag.md`):
- DN attn_qkv split ordering or post-conv1d activation order
- ssm_norm weight broadcast across value heads
- RoPE pair convention in the Metal head_norm_rope kernel
- Inner accumulator ordering in fused decode attention

To close: per-layer intermediate dump from llama.cpp (`--logits-all`
isn't enough; need a custom build or use the Python HF reference) and
a layer-by-layer numerical diff.

### #30 — push to 200 tok/s
Current 82 tok/s → target 200 needs:
1. GPU-side Q4_K matmul (skip the fp16 dequant + upload step)
2. bf16 compute path (MPS bf16 GEMV is faster than fp16 on M-series)
3. Persistent MTLCommandBuffer reuse across steps (ring of 2-3 cbs)

Each ~1.3-1.5×; combined plausibly 2-2.5×.

### #25, #26, #27 — nice-to-have
- cl100k pre-tokenize regex for exact HF tokenizer compatibility
- CUDA/AMD compile + validate on a real GPU host
- Golden-vector unit tests for each kernel

## Build & run

```bash
make all                                       # cpu + whatever GPU toolchain present
make test                                      # precision invariant + e2e smoke

# Run
./qw36_metal -m models/Qwen3.5-0.8B-Q4_K_M.gguf -p "Hello"

# Bench fast path
QW36_METAL_FP16_WEIGHTS=1 ./qw36_metal -m models/Qwen3.5-0.8B-Q4_K_M.gguf \
    -p "Hi" -n 128

# Diagnostic
./qw36_dump_tensor models/Qwen3.5-0.8B-Q4_K_M.gguf token_embd.weight 8 9419
```

## 2026-05-20 push to 200 tok/s — session 1 closed at 85 tok/s

#31 root cause: **Qwen3.5 vanilla attention has Q-gate** (q_proj output =
2*n_heads*head_dim; attn_out *= sigmoid(gate) before o_proj). Detected
from agent-infer's `has_qk_gate` branch + Qwen3.5 config
`attn_output_gate: true`. Fixed CPU + Metal fused kernel; both backends
now output `《秋夜》` 七言绝句 on the Chinese poem prompt.

Architecture split (#34, #36): `common/qw36.c` 2408 → 1193 lines.
qw36_dequant.c / qw36_ops.c / qw36_attn_vanilla.c / qw36_attn_deltanet.c
/ qw36_mlp.c / qw36_moe.c each own their part. Public API frozen.

GGUF GPU-native quant matmul (#39): ported agent-infer's Q4_K / Q5_K /
Q6_K / Q8_0 dequant + gemv kernels. Currently opt-in
(`QW36_METAL_QUANT_GPU=1`) at 50 tok/s; the per-row TG is too coarse for
lm_head (vocab × hidden) so we keep embed/lm_head materialized to fp16.

fp16 residual stream (#40): kernel-side plumbing done — rmsnorm /
residual_add / silu_mul / embed_lookup / matmul (4 in/out dtype combos)
/ fused decode attention all carry dtype params. State buffer flip is
deferred: x_rms_dev=F16 produces step-0 logit divergence (top-1 flips
',' → 'uela'), root cause is being investigated (#48). Suspect MPS GEMV
fp16-input ordering vs kernel-written buffer, or fp16 dynamic range hit.

agent-infer studies in this push:
- they use **bf16** (8-bit exponent) throughout, not fp16 (5-bit) — wider
  dynamic range, similar mantissa precision
- compiled functions (`precise_sigmoid_mul`, `precise_swiglu`) upcast to
  fp32 inside, output bf16 — no precision loss on hot scalars
- `gate_up` weights are concatenated at load → 1 matmul per MLP layer (#47)
- decode uses `verify_quantized_matmul_cpp` (custom Metal kernel, not MPS)

## Path to 200 tok/s

Each step is independently committable; ordered by est ROI / hour:

1. **#47 Fused gate_up matmul** — 1 matmul/MLP instead of 2; ~+5-10 tok/s
2. **#48 fp16 root cause + apply** — unblocks #46 phase 2; ~+25 tok/s
3. **bf16 instead of fp16 weights/state** — dynamic range fix; ~+5 tok/s
4. **#39 Q4_K kernel optimize** — float4 loads + multi-row TG; opt-in only
5. **Custom fp16/bf16 GEMV replacing MPS** — unlocks persistent compute
   encoder (24 layers × ~10 dispatches in one encoder); ~+30 tok/s
6. **Speculative decode (verify-m16)** — agent-infer's main throughput
   trick; multi-day port

Combined estimate: 85 + 7 + 25 + 5 + 30 ≈ 150 tok/s. Hitting 200 needs
all of those plus removing some per-dispatch overhead.

## 2026-05-20 session closeout

Final state: **85 tok/s sustained** on `QW36_METAL_FP16_WEIGHTS=1`, correct
output (Hello + 古诗 + chat-templated prompts all pass; precision and
golden tests green).

### Wins this session
- **#31 correctness root-caused + fixed**: Qwen3.5/3.6 vanilla attention
  has a Q-gate (q_proj output = `2 * n_heads * head_dim`, attn_out is
  multiplied by `sigmoid(q_gate)` before o_proj). Confirmed against
  `attn_output_gate: true` in Qwen3.5-0.8B config. CPU + Metal both
  produce `《秋夜》` 七言绝句 on the Chinese poem prompt.
- **Metal fused decode attention writes gate-corrected fp16/fp32 y** via
  `qw36_store_scalar(y, y_dtype, ...)` (commit 14b1a06 + 2fec86f).
- **GGUF-native quant matmul kernels** (Q4_K/Q5_K/Q6_K/Q8_0) ported from
  agent-infer with sub-block scale caching (commits e28726d + 564eee4).
  Currently opt-in at 50 tok/s; needs sub-block parallel + float4 loads
  before it beats MPS fp16 gemv.
- **fp16 matmul edges plumbing**: matmul fp16 path handles all 4 in/out
  dtype combos; rmsnorm / residual_add / silu_mul / embed_lookup /
  fused decode attention all dtype-aware via load/store_scalar
  (commits 678900d / a79d8b9 / 684b730 / 73ec828 / 2fd2531 / 2fec86f).
- **MPS weight matrix wrappers cached** (codex b63285b) — keeps the
  MPSMatrix objects across decode steps.
- **f32→f16 input conversion deduped** within q/k/v + gate/up triplets
  (commits 9d06231 + 720ac5d).
- **Architecture split #36** (codex eb68b0f): qw36.c 2408 → 1193 lines.
  qw36_dequant.c + qw36_ops.c + qw36_attn_*.c + qw36_moe.c each own
  their part of the forward.
- **Fused gate_up MLP matmul** (codex 29cb7ca): single MPS gemv writes
  both gate and up into one buffer, halves MLP matmul dispatches.
- **Tools**: tools/mlx_dump_intermediates.py + tools/diff_layers.py
  (layer-by-layer fp32 cross-check against MLX reference), kernel
  golden test, fp16 root-cause doc.

### What we tried but parked: fp16 residual state (#46/#48)
- Switching x_rms_dev or q_dev from fp32 to fp16 produces step-0 logit
  divergence under `QW36_METAL_FP16_WEIGHTS=1` (top-1 'Hello' →
  ',' becomes 'uela').
- Bisected with a custom Metal fp16 GEMV (commit ec1fbab) — reproduces
  the same divergence, so it is **not** an MPS fp16 ABI issue. It is
  fp16 precision drift at attention output (q_dev): the fused decode
  kernel computes acc in fp32 then stores fp16, and the cumulative
  loss across 6 vanilla layers × 1 fp16 store per layer is enough to
  flip the top-1 logit in a 152k-vocab argmax.
- agent-infer ducks this with bf16 storage + `compiled_precise_*`
  functions that upcast to fp32 inside fused operators. We do upcast
  inside the fused attn kernel, but the *store boundary* is still fp16
  and feeds the next op directly.
- **Fix path** (next session): keep q_dev fp32; only flip x_rms_dev to
  fp16 (verified correct — same logit). Need to diagnose why
  x_rms_dev fp16 was *slower* than fp32 baseline despite saving the
  f32→f16 input conversion per matmul.

### What we learned reading agent-infer
- They use **bfloat16** end-to-end on the residual stream and KV cache,
  not fp16. bf16's 8-bit exponent matches fp32's range and avoids
  saturation, while bf16's 7-bit mantissa is *less* precise than
  fp16's 10-bit. So bf16 wins on stability, not raw precision.
- Compiled precise functions (`precise_sigmoid_mul`, `precise_swiglu`)
  upcast to fp32 inside, output bf16 — the upcast is what preserves
  correctness across 24 layers.
- `verify_quantized_matmul_cpp` is their decode-mode quant matmul
  kernel; for batch=1 it routes through MLX's standard quantised
  matmul (also a custom kernel, not MPS gemv).
- They concatenate `gate_proj + up_proj` at load to skip one matmul
  per layer (#47 — we did this).

### Path to 200 tok/s (open follow-up)
Ordered by expected ROI / hour, summing to ~130-150 tok/s realistically:

1. **#48 fp16 attn store fix** — keep q_dev fp32, x_rms_dev fp16 with
   the right code path. Diagnose the slowdown first; should land
   +10-15 tok/s.
2. **bf16 state buffers + bf16 weight materialise** — switch
   `lazy_materialize_f16` to a bf16 variant; matmul fp16 fast path
   already accepts dtype via load/store_scalar. Expect +5 tok/s and
   the multi-layer drift goes away.
3. **#39 quant kernel optimisation** — float4 loads + multi-row TG so
   the per-row dequant+gemv beats MPS fp16 gemv. Unlocks `QW36_METAL_QUANT_GPU=1`
   as the default fast path (saves ~1 GB ram).
4. **Custom fp16/bf16 GEMV that beats MPS** + persistent compute
   encoder across rmsnorm / matmul / residual_add. Eliminates the
   per-call MPS encoder bring-up — about 0.5ms / token.
5. **agent-infer-style verify_quantized_matmul** port — for prefill
   only at first; decode benefit smaller.

Realistic horizon: 150 tok/s with another 1-2 dedicated sessions.
200 tok/s parity with MLX is a multi-week project (MLX's lazy graph
compiler does a lot we cannot reproduce without writing a kernel
compiler ourselves).
