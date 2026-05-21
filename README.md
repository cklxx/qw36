# qw36

Pure-C inference framework for the Qwen 3.6 / 3.5 family. Single-`.c` engine
per backend, no GGUF-runner abstraction, no C++ on the host. Three GPU
backends (Metal, CUDA, AMD HIP) sit behind one frozen vtable
(`common/qw36_gpu.h`); a CPU reference forward in `common/qw36.c` is the
source of truth.

The model architecture is the Qwen 3.5/3.6 **hybrid**: per-layer flavor
between vanilla **Grouped-Query Attention** (Q-/K-norm + partial mRoPE) and
**Gated DeltaNet** (fused QKV, depthwise conv1d, recurrent rank-1 state),
SwiGLU MLP, optional Top-K **MoE** with shared expert, tied or untied
`lm_head`.

Current decode throughput on `Qwen3.5-0.8B-Q4_K_M.gguf` (Apple M-class GPU):

| build                                  | decode tok/s |
|----------------------------------------|--------------|
| CPU reference (`qw36_cpu`)             |  1.7         |
| Metal, fp32 weights                    | 55           |
| Metal, `QW36_METAL_FP16_WEIGHTS=1`     | **81**       |
| llama.cpp reference, same model        | 170          |

The full perf ladder lives in [`DIVISION_OF_WORK.md`](DIVISION_OF_WORK.md).
The Qwen3.5 Q-gate fix makes the 0.8B baseline produce coherent text on
both CPU and Metal; `tests/precision_cpu_vs_metal.sh` keeps the fp32 Metal
path bit-identical to CPU at step 0.

---

## Quick start

```bash
git clone <this-repo> qw36
cd qw36

# Build a CPU build (always works) and whichever GPU toolchains exist.
make cpu        # → ./qw36_cpu
make metal      # → ./qw36_metal      (macOS, Apple Silicon)
make cuda       # → ./qw36_cuda       (needs nvcc)
make amd        # → ./qw36_amd        (needs hipcc / ROCm)
make all        # builds CPU plus whatever toolchains are present

# Fetch a Qwen3 GGUF (script picks Qwen3.5-0.8B-Q4_K_M by default).
./tools/download_model.sh

# Run it.
./qw36_metal -m models/Qwen3.5-0.8B-Q4_K_M.gguf -p "Hello"
```

`--info` is the cheapest sanity check — it prints the parsed config and
per-layer attention flavor survey without allocating KV cache:

```bash
./qw36_cpu -m models/Qwen3.5-0.8B-Q4_K_M.gguf --info
```

---

## Supported tensor formats

The GGUF loader (`common/qw36_gguf.c`) and dequantizer (`common/qw36.c`)
cover the K-quants and legacy floats below. Quantized blocks are read into
fp32 on demand (`matmul_lazy`) or pre-materialized when a GPU backend is
attached.

| dtype  | enum (`qw36_dtype`) | block size (bytes / 256-elem super-block) |
|--------|---------------------|-------------------------------------------|
| F32    | `QW36_DTYPE_F32`    | 4 bytes / element                          |
| F16    | `QW36_DTYPE_F16`    | 2 bytes / element                          |
| BF16   | `QW36_DTYPE_BF16`   | 2 bytes / element                          |
| Q8_0   | `QW36_DTYPE_Q8_0`   | 34 bytes / 32-elem block                   |
| Q2_K   | `QW36_DTYPE_Q2_K`   | `QK_K/16 + QK_K/4 + 2 + 2` = 84 / 256      |
| Q3_K   | `QW36_DTYPE_Q3_K`   | `QK_K/8 + QK_K/4 + 12 + 2`  = 110 / 256    |
| Q4_K   | `QW36_DTYPE_Q4_K`   | `2 + 2 + 12 + QK_K/2`       = 144 / 256    |
| Q5_K   | `QW36_DTYPE_Q5_K`   | `2 + 2 + 12 + QK_K/8 + QK_K/2` = 176 / 256 |
| Q6_K   | `QW36_DTYPE_Q6_K`   | `QK_K/2 + QK_K/4 + QK_K/16 + 2` = 210 / 256 |

`QK_K = 256`. Block layouts are byte-equivalent to `ggml-quants.c`; the
CPU `dequant_*` helpers were diffed against ggml block layouts (commits
c479cad, 2df5298, ce9343e).

---

## Backends

| backend | binary       | toolchain        | status                                                  |
|---------|--------------|------------------|---------------------------------------------------------|
| CPU     | `qw36_cpu`   | clang/gcc        | Reference forward. Slow (1.7 tok/s) but algorithmic-truth. |
| Metal   | `qw36_metal` | `xcrun metal` + clang | End-to-end on Apple Silicon. fp32 and `QW36_METAL_FP16_WEIGHTS=1` fast paths. Fused decode kernels for q/k norm + RoPE + KV-append and attention score/softmax/combine. Metal MoE wired but not validated. |
| CUDA    | `qw36_cuda`  | `nvcc`           | Kernel set mirrors Metal (rmsnorm, matmul, head_norm_rope, kv_append, attn). **Not compiled on this Apple host** — code reviewed against the Metal source but unverified. |
| AMD HIP | `qw36_amd`   | `hipcc` / ROCm   | HIP port of the CUDA kernels. Same caveat — **not compiled on this Apple host**. |

The CPU build links `cpu/qw36_cpu_stub.c`, whose `qw36_backend_create()`
returns `NULL` so the CLI falls through to the reference forward path in
`common/qw36.c`.

---

## CLI flags

All four binaries share `common/qw36_cli.c`. Flags:

| flag                   | description                                                                   |
|------------------------|-------------------------------------------------------------------------------|
| `-m <path>`            | Path to a GGUF model file. **Required.**                                       |
| `-p <prompt>`          | Prompt text. Chat-wrapped with `<|im_start|>user … <|im_end|>` unless `--no-special`. |
| `-n <int>`             | Max new tokens to generate. Default 128.                                       |
| `-t <float>`           | Sampling temperature. `<= 0` means argmax. Default 0.                          |
| `--top-p <float>`      | Nucleus sampling threshold. `0` or `1` disables. Default 1.                    |
| `--top-k <int>`        | Top-K sampling. `0` disables. Default 0.                                       |
| `--seed <u64>`         | RNG seed for stochastic sampling. Default 42.                                  |
| `--seq <int>`          | KV cache capacity (tokens). Default 2048.                                      |
| `--interactive`        | REPL mode (parsed but minimal — single-turn for now).                          |
| `--no-special`         | Skip the Qwen3 chat template; feed the raw prompt unwrapped.                   |
| `--debug-top <K>`      | Before each sampling step, dump the top-K logits and decoded token strings.    |
| `--dump-tokens`        | Tokenize the (chat-wrapped) prompt, print ids + decoded surface forms, exit.   |
| `--info`               | Print parsed config, tokenizer specials, per-layer attention flavor survey; exit. |
| `-h`, `--help`         | Usage.                                                                         |

---

## Environment knobs

Runtime and debug knobs are read at startup unless noted otherwise:

| var                          | effect                                                                                          |
|------------------------------|-------------------------------------------------------------------------------------------------|
| `QW36_METAL_FP16_WEIGHTS=1`  | Metal only. Materialize large lazy weights as fp16 and dispatch MPS half GEMV. Typical Qwen3.5-0.8B-Q4_K_M decode improves from ~55 tok/s to ~83 tok/s. |
| `QW36_METAL_FP16_KV=1`       | Metal only. Store the persistent KV cache as fp16. Long-context decode benefits because attention scans half the cache bytes. Enabled by default when fp16 weights are enabled unless explicitly set to `0`. |
| `QW36_METAL_FP16_EDGES=1`    | Metal diagnostic. Store `x_rms_dev` and `q_dev` as fp16 when fp16 weights are enabled. Currently reproduces the #46 step-0 divergence, so it is off by default. |
| `QW36_METAL_FP16_XRMS=1` / `QW36_METAL_FP16_Q=1` | Metal diagnostics. Flip only the RMSNorm-output edge or only the attention-output edge to fp16 for #46 bisection. |
| `QW36_METAL_F16_GEMV_QUAD=1` | Metal diagnostic. Route fp16-weight GEMV through a qmv_quad-style MSL kernel for rows up to `QW36_METAL_F16_GEMV_QUAD_MAX_ROWS` (default 512). Correct but slower than MPS on this host, so off by default. |
| `QW36_METAL_MMA_GEMV=1`      | Metal diagnostic. Route fp16-weight GEMV through an 8x8 `simdgroup_matrix` MMA kernel, optionally bounded by `QW36_METAL_MMA_GEMV_MIN_ROWS` / `QW36_METAL_MMA_GEMV_MAX_ROWS`. Correct but slower than MPS for M=1 decode on this host. |
| `QW36_METAL_QUANT_GPU=1`     | Metal only. Use GPU-native quant matmul instead of host fp32/fp16 materialization. This saves roughly 1 GB RAM on the 0.8B model and is an opt-in low-memory path. |
| `QW36_METAL_QK_REPACK=1`     | Metal quant path. Repack hot Q4_K/Q5_K weights into qmv-friendly affine32 layouts. On Qwen3.5-0.8B-Q4_K_M this raises the quant path from roughly 35-60 tok/s to about 100 tok/s in the short Hello bench while preserving the standard smoke output. |
| `QW36_METAL_Q4K_AFFINE32=1` / `QW36_METAL_Q5K_AFFINE32=1` | Metal diagnostics. Enable only one affine32 repack family for bisecting Q4_K or Q5_K kernels. |
| `QW36_METAL_Q6K_SCALE16=1`   | Metal opt-in quant path. Repack Q6_K into a scale16 qmv layout. It is faster, but intentionally separate from `QW36_METAL_QK_REPACK=1`; run `tests/quant_fastest_smoke.sh` when using it as part of the fastest path. |
| `QW36_METAL_QUANT_GPU_LM_HEAD=1` | Metal opt-in quant path. When `lm_head` aliases `embed_tokens`, split a Q5_K/Q6_K lm_head descriptor so output projection can stay quantized while embedding lookup remains materialized. Used with Q4/Q5 affine32 plus Q6 scale16 for the fastest smoke-gated path. |
| `QW36_METAL_FAST=1`          | Convenience umbrella: under `QW36_METAL_QUANT_GPU=1`, defaults `Q4K_AFFINE32`, `Q5K_AFFINE32`, `Q6K_SCALE16`, and `QUANT_GPU_LM_HEAD` to on. Individual flags still override (`=0` to opt out of a single component). Smoke gate: `tests/quant_fastest_smoke.sh`. |
| `QW36_DEBUG_LAYER=1`         | Per-layer trace: prints `||x||` and the first residual components before/after each block. Useful for bisecting the first divergent layer. |
| `QW36_MAX_LAYERS=<n>`        | Stop forward after the first `n` layers. Used with layer traces for range bisection. |
| `QW36_BYPASS_LAYERS=<spec>`  | Bypass selected layer ids or ranges during forward. Used to isolate a bad block without changing model loading. |
| `QW36_SKIP_DN=1`             | Skip Gated DeltaNet layers, leaving vanilla attention layers active. Useful for separating DN regressions from GQA regressions. |

---

## Architecture

### Engine

`qw36_engine_open()` (in `common/qw36.c`) mmaps the GGUF file, parses the
v3 metadata, derives `qw36_config` (correcting model-metadata bugs — e.g.
`num_attention_heads` is recovered from `attn_q.weight` shape, not the
declared field, which is wrong in the 0.8B GGUF), detects per-layer
attention flavor by checking for `blk.X.attn_q.weight` (vanilla) vs
`blk.X.attn_qkv.weight` + `blk.X.ssm_conv1d.weight` (DeltaNet), and binds
every tensor to either a host pointer (small tensors: norms, biases) or a
**`qw36_lazy_w`** descriptor.

### `qw36_lazy_w`

A lazy weight descriptor records `{ data_ptr, ggml_type, rows, cols }` and,
optionally, a materialized `gpu_buf`. On a CPU-only run, `matmul_lazy`
dequantizes one row at a time into a small fp32 scratch and accumulates
the dot product. With a GPU backend attached, large lazy weights (embed,
lm_head, every projection inside each layer's hot path) are
**pre-materialized** at engine-open time — either fp32 or fp16 depending
on `QW36_METAL_FP16_WEIGHTS` — and the descriptor's `gpu_buf` is set so
the matmul vtable dispatches straight to MPS / cuBLAS / rocBLAS GEMV
without re-uploading per step.

### Backend vtable

`common/qw36_gpu.h` defines a single struct of function pointers. Every
backend exports exactly one symbol, `qw36_backend_create()`, returning a
pointer to its statically-initialized vtable. The hot path:

- `init` / `destroy` — pick the default device, allocate command queue / stream.
- `upload` / `download` / `copy_from_host` / `alloc` / `free` — buffer mgmt.
- `begin_batch` / `end_batch` — optional. Lets a backend hold one command
  buffer across all ops in a forward step and commit once at the end
  (Metal uses this; CPU fallback is no-op).
- `rmsnorm`, `matmul`, `residual_add`, `embedding_lookup`,
  `swiglu_mlp`, `attention` — the basic kernel set.
- `dn_conv1d_silu`, `dn_gated_delta`, `dn_gated_rmsnorm` — Gated DeltaNet
  decode (optional; CUDA/AMD may lag Metal).
- `moe_forward` — Top-K MoE with shared expert (optional; dense-only
  backends set to NULL).

### Persistent GPU state

`qw36_state` carries two parallel sets of buffers: host scratch
(`st->x`, `st->q`, `st->k`, `st->logits`, …) and device buffers
(`st->x_dev`, `st->q_dev`, `st->logits_dev`, plus per-layer
`st->k_cache_dev[L]`, `st->v_cache_dev[L]`, `st->conv_state_dev[L]`,
`st->delta_state_dev[L]`). After task #28 (commit `d24bbbd`), the entire
residual stream, KV cache, MLP scratch, and logits live on the device
across decode steps — the host scratch is touched only on debug-trace
paths, sampling, and the gated_delta_decode host fallback. This is what
unlocked the 11 → 55 tok/s jump on Metal.

### Gated DeltaNet path

`gated_delta_decode()` in `common/qw36.c` is the host reference for the
DN block: per-head rank-1 state `S ∈ [n_v_heads, key_dim, val_dim]`
updated as `S ← g * S + β * (k ⊗ v)` with `g = exp(neg_exp_a * softplus(a + dt_bias))`
and `β = sigmoid(b)`, then `y = S · q` followed by gated RMSNorm and
output projection. The Metal backend has a fused `dn_conv1d_silu` +
`dn_gated_delta` + `dn_gated_rmsnorm` path, modeled on agent-infer's
`gated_delta_step` kernel (state layout `[B, Hv, Dv, Dk]`, threadgroup
x=32 simd lanes along Dk, simd-sum reductions for the kv_mem and output
dot product). CUDA / AMD currently fall back to host for DN.

---

## Reference implementations

We cross-reference two:

- **[agent-infer](../agent-infer)** — Rust + MLX reference for Qwen 3.5
  semantics. Source of the exact kernel layout we ported to Metal for
  the Gated DeltaNet step. Notable files:
  - `crates/qwen35-spec/src/lib.rs` — model arithmetic, `rope_inv_freq`,
    QK normalization, compute_g_beta.
  - `crates/mlx-sys/src/mlx_qwen35_model.cpp:203-275` — the
    `gated_delta_step` Metal kernel.
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** — the
  ground-truth for GGUF tensor layout, K-quant block packing, and decode
  output. The 170 tok/s number in the hero table is `llama-cli` on the
  same `Qwen3.5-0.8B-Q4_K_M.gguf` on the same host.

---

## Testing

```bash
make test
```

Runs, in order:

- `tests/precision_cpu_vs_metal.sh` — single-step fp32 forward on CPU
  and Metal, asserts bit-equal logits at step 0. **Currently green.**
- `tests/e2e_qwen35_smoke.sh ./qw36_cpu` — end-to-end generation, looks
  for a non-degenerate top-1 token. Informational only — currently
  regressed (see below).
- `tests/e2e_qwen35_smoke.sh ./qw36_metal` — same, on Metal. Same caveat.

The layer-trace workflow lives in
**[`tests/correctness_diag.md`](tests/correctness_diag.md)** and
`tools/diff_layers.py`. It records the Q-gate root cause and the tensor
dumps used to keep Qwen3.5/3.6 vanilla attention aligned with the MLX
reference.

---

## Known limitations

1. **GPU MoE is built but unvalidated.** `metal/qw36_metal.m` implements
   `moe_forward` (route + top-K, expert gate/up, down+combine), but no
   MoE GGUF is on disk locally so we have no numeric reference. CUDA/AMD
   MoE not yet implemented.
2. **35B-A3B does not fit.** Eager weight materialization at engine-open
   would need ~140 GB fp32 / ~70 GB fp16. A streaming / lazy-quantized
   matmul path that keeps weights in Q4_K on the device and dequantizes
   per row inside the kernel is the prereq (task TBD).
3. **Hybrid layer coverage.** Vanilla GQA and Gated DeltaNet run
   end-to-end on CPU and Metal for the 0.8B baseline. CUDA/AMD still lag
   Metal parity for the latest quant, DN, and MoE paths.
4. **Decode rate degrades with context.** 81 tok/s at n=16 falls to 74
   at n=128 because attention score/softmax/combine are three separate
   Metal dispatches per layer; fusing into one MSL kernel is task #30.
5. **`QW36_ROPE_NEOX=1`** is documented in `correctness_diag.md` but
   not currently part of the default Qwen3.5 path; treat it as a
   diagnostic knob.

---

## Layout

```
qw36/
├── common/                 shared host C
│   ├── qw36.h              public API: config, weights, state, sampler
│   ├── qw36.c              engine lifecycle, state alloc, forward, prefill, sampling
│   ├── qw36_dequant.c      GGUF dtype conversion and block dequantization
│   ├── qw36_ops.c          lazy matmul, embedding lookup, scalar/GPU op dispatch
│   ├── qw36_attn_*.c       vanilla GQA and Gated DeltaNet layer bodies
│   ├── qw36_mlp.c          SwiGLU / dense MLP layer body
│   ├── qw36_moe.c          router, top-k expert combine, shared expert
│   ├── qw36_gpu.h          backend vtable (frozen)
│   ├── qw36_gguf.[ch]      GGUF v3 mmap loader
│   ├── qw36_tokenizer.[ch] BBPE tokenizer + special-token vocab
│   └── qw36_cli.c          main(), arg parsing, prompt wrapping, generation loop
├── cpu/                    NULL backend → CPU reference path
├── metal/                  Metal: qw36_metal.m + qw36_metal.metal (+ .metallib)
├── cuda/                   CUDA: qw36_cuda.cu
├── amd/                    HIP:  qw36_amd.cpp
├── tools/
│   ├── download_model.sh   pull a Qwen3 GGUF from HF
│   ├── dump_tensor.c       dequant + print first N elems for diffing
│   └── diff_layers.py      compare qw36 and MLX intermediate tensor dumps
├── tests/
│   ├── precision_cpu_vs_metal.sh
│   ├── e2e_qwen35_smoke.sh
│   ├── kernel_golden.sh
│   └── correctness_diag.md ← open issue tracker
├── DIVISION_OF_WORK.md     who-owns-what + perf ladder
├── Makefile                top-level dispatch
└── README.md               this file
```
