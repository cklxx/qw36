# qw36 — Final Status

## TL;DR

Pure-C Qwen 3.5/3.6 inference framework, 3 GPU backends, **85 tok/s
sustained decode** on Qwen3.5-0.8B-Q4_K_M via Metal (vs llama.cpp's
170 tok/s reference, ~50%). CPU baseline 1.7 tok/s. Multi-session push
to 200 tok/s in progress (#30/#40).

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
| 720ac5d            | **85** | **xh cache fp16-input invalidate** |
| (llama.cpp ref)    | 170    | upstream baseline                 |
| (agent-infer ref)  | ~200   | MLX bf16 + custom Q4_K + compiled fused kernels |

`QW36_METAL_FP16_WEIGHTS=1` to opt in to fp16 weights (default fp32 keeps
precision_cpu_vs_metal.sh byte-equal).

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

## 2026-05-20 push to 200 tok/s — in progress

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
