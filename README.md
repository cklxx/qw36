# qw36

Pure-C inference framework for Qwen 3.6. Three backends, three binaries:

| Backend | Folder    | Binary       | Toolchain     |
|---------|-----------|--------------|---------------|
| AMD     | `amd/`    | `qw36_amd`   | hipcc / ROCm  |
| Metal   | `metal/`  | `qw36_metal` | clang + Metal |
| CUDA    | `cuda/`   | `qw36_cuda`  | nvcc          |

Inspired by [antirez/ds4](https://github.com/antirez/ds4) (single-`.c` engine
per backend, no GGUF-runner abstraction) and the Rust reference
implementation in `../agent-infer/crates/qwen35-spec` (model arithmetic).

## Layout

```
qw36/
├── common/         shared C code (engine, loader, tokenizer, CLI)
│   ├── qw36.h           model config, weights, state, public API
│   ├── qw36.c           CPU reference forward pass, RMSNorm, RoPE, sampling
│   ├── qw36_gpu.h       backend vtable — what every backend must implement
│   ├── qw36_gguf.[ch]   GGUF v3 file loader
│   ├── qw36_tokenizer.[ch] BPE tokenizer
│   └── qw36_cli.c       main(), arg parsing, REPL
├── amd/            HIP backend (qw36_amd.cpp + Makefile)
├── metal/          Metal backend (qw36_metal.m + qw36_metal.metal + Makefile)
├── cuda/           CUDA backend (qw36_cuda.cu + Makefile)
├── tools/          model download / GGUF inspection
└── tests/          golden vectors, unit harness
```

## Build

```bash
make amd      # builds qw36_amd   (requires hipcc)
make metal    # builds qw36_metal (macOS, Apple Silicon)
make cuda     # builds qw36_cuda  (requires CUDA toolkit)
make all      # whichever toolchains are present
```

## Run

```bash
./qw36_metal -m models/qwen3.6-4b.gguf -p "Hello, world"
```

## Model

Qwen 3.6 — dense and MoE variants. Architecture:

- Pre-norm transformer, RMSNorm (eps = 1e-6).
- Grouped-Query Attention with Q-norm and K-norm.
- Rotary position embedding (partial rotary).
- SwiGLU MLP; MoE layers use Top-K routing with optional shared expert.
- Tied or untied `lm_head`.

See `common/qw36.h` for the exact config struct and tensor-name contract.

## Division of work

See `DIVISION_OF_WORK.md`. Short version: Claude owns the host-side C (loader,
tokenizer, CPU reference, CLI, build). Codex owns GPU kernels (HIP, Metal
shaders, CUDA).
