# qw36

Pure-C inference engine for the Qwen 3.5 / 3.6 family. One `.c` engine
per backend, no C++ on the host, no Python in the hot path. Three GPU
backends (Metal, CUDA, AMD HIP) sit behind one frozen vtable
(`common/qw36_gpu.h`); a CPU reference forward in `common/qw36.c` is
the source of truth that every backend must match numerically.

The model architecture is the Qwen 3.5/3.6 **hybrid**: per-layer flavor
between vanilla **Grouped-Query Attention** (Q-/K-norm + partial mRoPE
+ Q-gate) and **Gated DeltaNet** (depthwise conv1d + recurrent rank-1
state), SwiGLU MLP or Top-K **MoE** with shared expert, tied or untied
`lm_head`.

Current decode throughput on `Qwen3.5-0.8B-Q4_K_M.gguf` (Apple M-class
GPU; load avg < 3; 5-run median):

| build                                  | n=64 short | n=256 sustained | n=2048 |
|----------------------------------------|-----------:|----------------:|-------:|
| CPU reference (`qw36_cpu`)             |        1.7 |               — |      — |
| Metal, fp32 weights                    |         55 |              50 |      — |
| Metal, `QW36_METAL_FP16_WEIGHTS=1`     |        119 |             103 |      — |
| **Metal, `--fast` (opt-in path)**      |   **204**  |        **176**  | **92** |
| llama.cpp reference, same model        |        170 |             170 |     — |
| MLX-LM reference (4-bit, same machine) |        290 |             290 |     — |

The full perf ladder + every lever lives in
[`FINAL_STATUS.md`](FINAL_STATUS.md). The bench methodology lives in
[`docs/performance_methodology.md`](docs/performance_methodology.md).
The 4-8 week plan is in [`ROADMAP.md`](ROADMAP.md). New contributors:
[`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Quick start

```bash
git clone <this-repo> qw36
cd qw36

# Build CPU (always works) plus whatever GPU toolchains are present.
make cpu        # → ./qw36_cpu
make metal      # → ./qw36_metal     (macOS, Apple Silicon)
make cuda       # → ./qw36_cuda      (needs nvcc; compile-checks on hosts w/o GPU)
make amd        # → ./qw36_amd       (needs hipcc / ROCm)
make all        # builds CPU plus whichever GPU toolchains exist

# Preflight check (model path optional; verifies SDK + env conflicts):
./qw36_metal --doctor -m /path/to/model.gguf

# Run greedy:
./qw36_metal --fast -m /path/to/Qwen3.5-0.8B-Q4_K_M.gguf -p "Hello" -n 64
```

If your build is missing a backend, `--doctor` says why. If you need
to know what `--fast` actually turns on, `--print-config` dumps the
effective profile + every env knob's current value:

```bash
./qw36_metal --fast --print-config
```

`--info` runs the engine init without entering forward and dumps the
parsed config + per-layer attention flavor:

```bash
./qw36_cpu -m model.gguf --info
```

---

## Supported models

Tested end-to-end on:

| model | quant | CPU | Metal | notes |
|---|---|---|---|---|
| Qwen3.5-0.8B | Q4_K_M | ✅ | ✅ `--fast` 204 tok/s | reference; perf target |
| Qwen3.6-35B-A3B | Q4_K_XL | ✅ coherent | ✅ MoE+DN aligned (commit `59ebce1`) | functional smoke; perf cycle pending |

Other Qwen3-family GGUFs (1.7B, 4B, 8B, 14B) probably work since they
share the GQA + DN + MoE architecture, but we haven't tuned for them.
File an issue with `--info` output if something breaks.

---

## Backends

| backend | binary | toolchain | status |
|---------|--------|-----------|--------|
| CPU | `qw36_cpu` | clang/gcc | Reference forward. ~1.7 tok/s on 0.8B; algorithmic-truth source. |
| Metal | `qw36_metal` | `xcrun metal` + clang | End-to-end on Apple Silicon. Full quant + lm_head + KV layout + fused decode kernels under `--fast`. MoE + DN validated against CPU (35B). |
| CUDA | `qw36_cuda` | `nvcc` | Compile-check only on this host. Kernel ports lag Metal — full parity is Roadmap theme 1.2. |
| AMD HIP | `qw36_amd` | `hipcc` / ROCm | Same caveat as CUDA. |

The CPU build links `cpu/qw36_cpu_stub.c`, whose `qw36_backend_create()`
returns `NULL`, so the CLI falls through to the reference forward in
`common/qw36.c`.

---

## CLI flags

```
Usage: ./qw36_metal -m <model.gguf> [options]
```

| flag | description |
|------|-------------|
| `-m <path>` | Path to a GGUF model file. **Required for inference**, optional for `--doctor` / `--print-config`. |
| `-p <prompt>` | Prompt text. Chat-wrapped with `<\|im_start\|>user … <\|im_end\|>` unless `--no-special`. |
| `-n <int>` | Max new tokens. Default 128. |
| `-t <float>` | Sampling temperature. `<= 0` ⇒ argmax. Default 0. |
| `--top-p <f>` / `--top-k <k>` / `--seed <u64>` | Sampler knobs. |
| `--seq <int>` | KV cache capacity (tokens). Default 2048. Auto-flips KV layout to transposed when > 512. |
| `--no-special` | Skip the Qwen3 chat template; feed raw prompt. |
| `--profile reference\|fp16\|lowmem\|fast` | Backend policy. CLI aliases: `--fast`, `--strict`. |
| `--info` | Print parsed config + tokenizer + per-layer attention flavor and exit. |
| `--doctor` | Preflight: SDK / model file / GGUF magic / env conflicts. OK/WARN/FAIL per line. Non-zero exit on FAIL. |
| `--print-config` | Dump effective profile + every `QW36_*` env knob and exit. |
| `--debug-top <K>` | Per-step top-K logits + decoded surface forms. |
| `--dump-tokens` | Tokenize prompt, print ids + decoded forms, exit. |
| `--layer-trace <L> --layer-trace-out <path>` | Dump intermediate tensors for layer L to a binary file (use `tools/diff_layers.py`). |
| `-h`, `--help` | Usage. |

---

## Environment knobs

User-facing knobs. The full reference (with lifecycle: stable /
internal / research / debug) is in [`docs/env_knobs.md`](docs/env_knobs.md);
`--print-config` shows the effective values at runtime.

| var | values | effect |
|-----|--------|--------|
| `QW36_PROFILE` | `reference\|fp16\|lowmem\|fast` | Backend policy. Same as `--profile`. Default `reference`. |
| `QW36_METAL_FAST` | `0\|1` | Umbrella for `--fast`. Sets the full quant + lm_head + KV path. |
| `QW36_METAL_FP16_WEIGHTS` | `0\|1` | Materialize quants to fp16 + dispatch MPS fp16 GEMV. Conflicts with `QUANT_GPU`. |
| `QW36_METAL_QUANT_GPU` | `0\|1` | Keep K-quant blocks on GPU; per-row dequant kernels. |
| `QW36_METAL_FP16_KV` | `0\|1` | fp16 K/V cache. Defaults on under `--fast`. |
| `QW36_METAL_BF16_KV` | `0\|1` | bf16 K/V cache (allocator-side; Metal kernel support is the in-flight follow-up — currently falls back to fp16 with a stderr warning). See [`docs/kv_quant_plan.md`](docs/kv_quant_plan.md). |
| `QW36_METAL_KV_TRANSPOSED` | `auto\|0\|1` | KV layout. `auto` (default) flips on at seq_capacity > 512. +24% n=2048; -10% n=64. |
| `QW36_METAL_PERF` | `0\|1` | Per-kernel `[metal-perf]` table at exit. **Disables persistent encoder; never bench under PERF.** |

Profile-driven, debug, and research knobs in
[`docs/env_knobs.md`](docs/env_knobs.md). Discoverable at runtime via
`--print-config`.

---

## Testing

Fast smoke (no model needed for the CLI / kvcache / goldens):

```bash
make check         # cpu build + 0.8B smoke + CLI smoke + kvcache smoke + kernel goldens
```

Full perf bench (needs the 0.8B model):

```bash
make perf          # compare_mlx.sh short + precision_cpu_vs_metal.sh
```

Standalone scripts under `tests/`:

| script | what it proves |
|--------|----------------|
| `tests/quant_fastest_smoke.sh` | `--fast` on 0.8B produces "Hello! How can I help you today?" |
| `tests/precision_cpu_vs_metal.sh` | step-0 fp32 CPU == Metal bitwise |
| `tests/cli_smoke.sh` | `--doctor` / `--print-config` / `--help` paths |
| `tests/kvcache_smoke.sh` | KV prefix cache (ram_lru + disk + promotion across processes) |
| `tests/golden_kernels.sh` | per-kernel goldens (rmsnorm / silu / swiglu) regenerate + verify |
| `tests/perf_gate.sh` | CI perf regression gate; fails on > 5% drop (10% on CI) |
| `tests/compare_mlx.sh [scope]` | qw36 vs MLX side-by-side decode tok/s |

Sanitizer builds (CPU only):

```bash
make asan          # → ./qw36_cpu_asan
make ubsan         # → ./qw36_cpu_ubsan
```

CI on GitHub Actions runs five jobs per PR: linux-cpu, macos-metal (+
kernel goldens), cuda-compile-check, linux-sanitizers, lint
(clang-format dry-run). A perf-gate job is wired but only fires when
`QW36_TEST_MODEL` is supplied.

---

## Architecture

The deep-dive is in [`docs/architecture.md`](docs/architecture.md). The
30-second mental model: single-stream `embed → N layers → output_norm →
lm_head → sample`, where layers alternate vanilla GQA and Gated DeltaNet,
with an MLP or top-K MoE in each. Every GPU backend implements the same
frozen vtable; CPU is the reference truth. Weights stay mmap'd as GGUF
and dequantize lazily (per-row on CPU; pre-uploaded to GPU under
`--fast`).

KV prefix cache infrastructure (multi-tier: ram_lru + disk, future
vram_lru + redis) lives in `common/qw36_kvcache.{h,c}` and
[`docs/kvcache_design.md`](docs/kvcache_design.md). The engine
attach point is in commit `0296183`; full state snapshot/hydrate is
the in-flight follow-up.

---

## Reference implementations

- **[agent-infer](../agent-infer)** — Rust + MLX reference for
  Qwen 3.5/3.6 semantics. We diffed against `crates/qwen35-spec` for
  arithmetic and `crates/mlx-sys` for the Gated DeltaNet step. MLX
  side-by-side bench in `tests/compare_mlx.sh`.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — ground
  truth for GGUF block layouts, K-quant byte formats, and CPU output.
  The 170 tok/s row in the hero table is `llama-cli` on the same
  GGUF + host.

---

## Known limitations

1. **CUDA / AMD parity.** Compile-checked only on this host;
   kernel-by-kernel correctness against the Metal set is Roadmap
   theme 1.2. PRs against those backends should produce step-0 logit
   diff vs CPU before claiming functional.
2. **bf16 / Q8_0 / Q4_0 KV.** Designed and tracked
   ([`docs/kv_quant_plan.md`](docs/kv_quant_plan.md), tasks #73 / #83
   / #84) but not yet shipping. `QW36_METAL_BF16_KV=1` is recognized
   but currently falls back to fp16 with a clear warning.
3. **Flash-attention single-pass decode.** Roadmap #71 Z. Long-context
   throughput at n=2048 is 92 tok/s vs MLX's ~300 essentially because
   we read K and V cache in separate passes. The transposed layout
   (`b4bb6f6`) cleared the layout obstacle; the streaming kernel is
   the next perf lever.
4. **35B-A3B perf cycle.** 35B is functional under `--fast` on Metal
   (commit `59ebce1`) but decode is still slow because the MoE
   expert dispatch hasn't gone through a perf pass yet. Codex is
   actively working it. Not a correctness problem.
5. **Speculative decoding.** Explicit non-goal for the current 4-8
   week window; design notes deferred to a future doc.
6. **Tokenizer.** Qwen3 BBPE + special vocab works for the chat
   template; cl100k pre-tokenization regex (#25) lands when needed
   for non-Qwen3 GGUFs.

---

## Layout

```
qw36/
├── common/                 cross-backend engine + reference forward
│   ├── qw36.{h,c}          public API + engine lifecycle + per-token forward
│   ├── qw36_internal.h     private: engine struct, forward_ctx, lazy_w
│   ├── qw36_gpu.h          backend vtable (FROZEN ABI)
│   ├── qw36_dequant.c      GGUF dtype conversion + affine32 / scale16 repacks
│   ├── qw36_ops.c          CPU rmsnorm / silu / residual / matmul reference
│   ├── qw36_attn_vanilla.c vanilla GQA + Q-gate
│   ├── qw36_attn_deltanet.c Gated DeltaNet (conv1d + gated_delta + tail)
│   ├── qw36_mlp.c          SwiGLU MLP
│   ├── qw36_moe.c          Top-K MoE + shared expert
│   ├── qw36_kvcache.{h,c}  tier-composing prefix cache (ram_lru + disk)
│   ├── qw36_gguf.{h,c}     GGUF v3 mmap loader
│   ├── qw36_tokenizer.{h,c} BBPE + Qwen3 specials
│   ├── qw36_policy.c       QW36_PROFILE → env flag resolution
│   └── qw36_cli.c          main(), --doctor, --print-config, --info
├── cpu/                    CPU build → qw36_cpu (NULL backend)
├── metal/                  qw36_metal.m + qw36_metal.metal (MSL)
├── cuda/                   qw36_cuda.cu
├── amd/                    qw36_amd.cpp
├── tools/
│   ├── install.sh          binary release installer
│   ├── download_model.sh   (stub today — see docs/troubleshooting.md)
│   ├── dump_tensor.c       dequant + print first N elems
│   ├── gen_goldens.c       per-kernel deterministic fixture generator
│   ├── check_goldens.c     fixture verifier (CPU reference)
│   ├── diff_layers.py      compare qw36 + MLX intermediate tensor dumps
│   └── audit_tensor_f16.c  f16 materialization audit (used for 35B bring-up)
├── tests/                  smoke + correctness + bench (see Testing above)
├── docs/                   design notes, post-mortems, env reference
│   ├── INDEX.md            entry point — what each doc is
│   ├── architecture.md     engine deep-dive (read for an afternoon)
│   ├── env_knobs.md        every QW36_* env variable
│   ├── performance_methodology.md  bench discipline
│   ├── troubleshooting.md  top symptoms with concrete checks
│   ├── kvcache_design.md   tier-composing KV prefix cache
│   ├── kv_quant_plan.md    fp16/bf16/Q8_0/Q4_0 KV scoreboard
│   ├── q4k_kernel_design_v2.md  the affine32 lever
│   ├── q4k_qmv_quad_failed.md   first port post-mortem
│   ├── fp16_state_root_cause.md fp16 KV drift bisection
│   ├── qwen36_35b_a3b_status.md 35B-A3B bring-up notes
│   └── briefs/             codex task briefs (historical record)
├── .github/workflows/      ci.yml + release.yml
├── ROADMAP.md              4-8 week plan
├── CONTRIBUTING.md         how to land a useful PR
├── AGENTS.md / CLAUDE.md   project contract (SOLID rules, perf discipline)
├── FINAL_STATUS.md         perf ladder + MLX comparison + failed experiments
├── CHANGELOG.md            release notes
└── README.md               this file
```

---

## License

[MIT](LICENSE) (when LICENSE lands — current state is research-only).
