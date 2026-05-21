# Changelog

All notable changes to qw36 are documented here. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-05-21

### Performance
- **Auto-flipped KV transposed layout under `--fast`** when `seq_capacity > 512`
  (`27f64ec`). Sessions sized for long context get the +24% n=2048 win
  by default; short sessions keep the cheap write path. Tri-state knob
  `QW36_METAL_KV_TRANSPOSED=auto|0|1`.
- **MLX-style Q4K_AFFINE32 qdot promoted to default-on** (`aa293f7`,
  earlier in cycle). Reverses an earlier "no win" call after re-bench
  at quiet load with 5-rep median.

### Correctness
- **35B-A3B MoE+DN Metal alignment** (`59ebce1`, codex). Three root
  causes: MoE input/output aliasing (CPU zeroed `y` over `x` before
  expert matmul), missing shared-expert scalar gate, and GGUF
  V-within-K head ordering. CPU output went from gibberish to
  `,\n\nI am trying to use the`; Metal now matches at n=8.
- **`qw36_load_scalar` extended to quant types** (`8db80b6`). Was
  returning 0 for Q8_0 / Q4K_AFFINE32 / Q5K_AFFINE32 / Q6K_SCALE16,
  which silently corrupted Metal MoE expert reads on 35B. Discovered
  via PERF kernel dispatch audit, not output.

### Infrastructure
- **KV prefix cache (Roadmap AD)** with tier-composing design
  (`c51030f`, `4e7e07a`). Generic cache loop over a vtable per
  storage medium; ships `ram_lru` + `disk` tiers. Mac (unified
  memory) gets 2 tiers; future CUDA / AMD / serving setups can
  prepend `vram_lru` / append `redis` without scheduler changes.
- **Engine attach point for the cache** (`0296183`). `qw36_prefill`
  consults the cache on entry and inserts on exit; full state
  snapshot/hydrate is a deferred follow-up (task #82).
- **Kernel goldens harness** (`c374b26`). `tools/gen_goldens.c` +
  `tools/check_goldens.c` + `tests/golden_kernels.sh` cover
  rmsnorm / silu / swiglu; rtol 1e-5. CI runs it on every PR.
  Precondition for the flash-attn rewrite (Roadmap #71 Z).
- **CI perf regression gate** (`6096edb`). `tests/perf_gate.sh`
  fires on macOS runner when `QW36_TEST_MODEL` is supplied;
  blocks PRs that regress > 10% (5% local, 10% CI for runner
  noise) with one-shot retest under quiet conditions.
- **Sanitizer CI jobs + lint** (`ffa8b08`). ASAN + UBSAN CPU
  builds + `clang-format --dry-run -Werror` lint job. `Make`
  targets: `asan`, `ubsan`, `lint`, `fmt`, `check`, `perf`,
  `help`.

### CLI / product
- **`--doctor`** (`c566a52`): preflight checks (SDK / backend
  binding / model file + GGUF magic / env conflict). OK/WARN/FAIL
  per line; non-zero exit on FAIL. Works without `-m`.
- **`--print-config`** (`c566a52`): dump effective profile + every
  `QW36_*` env knob's current value with lifecycle (stable /
  internal / research / debug). Discoverable env-knob inventory.
- **`tools/download_model.sh` now functional**. Pulls registered
  variants (0.8B Q4_K_M / Q5_K_M / Q8_0, 35B-A3B Q4_K_XL) from HF
  via huggingface-cli / curl / wget. sha256 verify when registered.
  `list` subcommand shows the registry.

### Documentation
- **`ROADMAP.md`** (`ffa8b08`): 4-8 week plan with 5 themes,
  prioritized task table per theme, next 5 PRs ranked, explicit
  non-goals (speculative decoding, kernel compiler, CUDA/AMD perf
  beyond functional parity).
- **`CONTRIBUTING.md`** (`ffa8b08`): build / test matrix, perf
  discipline, commit style, codex / claude delegation notes.
- **`docs/architecture.md`** (`4775334`): engine deep-dive with
  vtable contract, lazy weight materialization, per-token forward,
  fusion flags, persistent encoder, KV layouts, "where the
  surprises live".
- **`docs/performance_methodology.md`** (`ffa8b08`): why median-of-N,
  wallclock-not-gpu_ms, what counts as a real win.
- **`docs/env_knobs.md`** (`ffa8b08`): exhaustive `QW36_*` reference
  with file:line + lifecycle.
- **`docs/troubleshooting.md`** (`ffa8b08`): top failure modes with
  concrete checks.
- **`docs/kv_quant_plan.md`** (`0901e2e`): KV quantization scoreboard
  — fp16 shipped, bf16 (#73 AB) staged at allocator, Q8_0 / Q4_0
  designed (#83 / #84). `QW36_METAL_BF16_KV=1` recognized with
  fallback warning until kernel side lands.
- **`docs/model_support_matrix.md`** (new): green/yellow/red per
  (model, backend) cell with explicit testing scope.
- **`docs/kvcache_design.md`** (`c51030f`): tier model + per-platform
  tier stacks + file format spec.

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
