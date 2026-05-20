# Changelog

All notable changes to qw36 are documented here. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-20

### Correctness
- Fixed the Qwen3.5/3.6 vanilla-attention Q-gate path. `attn_q.weight`
  outputs `{q, q_gate}` per head, and the attention output is now gated
  by `sigmoid(q_gate)` before `o_proj`.
- Added the layer-trace diff harness:
  `tools/mlx_dump_intermediates.py` and `tools/diff_layers.py`.

### Architecture
- Split the common engine into focused modules:
  `qw36_attn_vanilla.c`, `qw36_attn_deltanet.c`, `qw36_mlp.c`,
  `qw36_moe.c`, `qw36_ops.c`, and `qw36_dequant.c`.
- Kept `qw36.c` focused on engine lifecycle, state allocation, forward
  scheduling, prefill, and sampling.

### Metal
- Added the Metal Q-gate fused kernel path.
- Added GGUF GPU-native quant matmul for Q4_K, Q5_K, Q6_K, and Q8_0.
- Added Q4_K sub-block scale caching.
- Deduplicated `xh` f32-to-f16 conversion in the fp16 MPS GEMV path.
- Added the MPS weight matrix wrapper cache.
- Extended fp16 residual-stream infrastructure so rmsnorm,
  residual_add, silu_mul, embedding_lookup, matmul, and fused attention
  accept dtype-aware buffers.

## [0.1.0] - 2026-05-20

Initial release. Pure-C inference framework for Qwen 3.5 / 3.6 with three
GPU backends and a CPU reference. Metal decode hits ~81 tok/s peak (75 tok/s
in long context) on the Qwen3.5-0.8B-Q4_K_M baseline.

### Engine
- Pure-C single-engine-per-backend architecture, inspired by antirez/ds4.
- GGUF v3 loader with on-demand dequantization (Q2_K, Q3_K, Q4_K, Q5_K,
  Q6_K, Q8_0); 35B-A3B MoE checkpoints fit in memory.
- CPU reference forward pass: RMSNorm, RoPE (with mRoPE per-axis sections),
  GQA attention, sampling.
- Gated DeltaNet decode + per-layer dispatch for the Qwen 3.5/3.6 hybrid.
- MoE forward (router + top-k experts + shared expert).
- BBPE tokenizer with Qwen3 special-token recognition
  (`<|im_start|>`, `<|im_end|>`, etc.) at encode time.

### Backends
- **CPU** (`qw36_cpu`): reference, no GPU toolchain required.
- **Metal** (`qw36_metal`, macOS Apple Silicon):
  - MPSMatrixVectorMultiplication cache keyed by `(rows, cols)`.
  - Bucketed `MTLBuffer` pool that recycles transient scratch.
  - Persistent forward-state buffers.
  - Fused gated delta step kernel.
  - First-pass MoE forward.
  - Opt-in fp16 weight path (`QW36_METAL_FP16_WEIGHTS=1`).
- **CUDA** (`qw36_cuda`): nvcc + cuBLAS, gated delta step kernel.
- **AMD** (`qw36_amd`): hipcc + rocBLAS, gated delta step kernel.

### Tests
- `tests/precision_cpu_vs_metal.sh` — step-0 logits must be bit-identical
  between CPU and Metal (fp32 weights). Drift from step 1 onward is the
  expected GPU↔CPU contract.
- `tests/e2e_qwen35_smoke.sh` — informational coherence check
  (`<think>` block + English word) for Qwen3.5 output.

### Tools
- `tools/download_model.sh` — fetch Qwen3.5/3.6 GGUFs.
- `tools/dump_tensor.c` — fp32 dump of a single tensor for cross-checking
  against llama.cpp.

[0.2.0]: https://github.com/cklxx/qw36/releases/tag/v0.2.0
[0.1.0]: https://github.com/cklxx/qw36/releases/tag/v0.1.0
