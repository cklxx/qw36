# Division of work — Claude × Codex

The scaffold compiles to **stubs** (every kernel returns zero or aborts). Two
agents fill it in. Their work is partitioned by *file* so they never edit the
same line.

## Rules

1. **One agent per file.** If you need to change a file the other agent owns,
   leave a `/* HANDOFF: ... */` comment instead of editing.
2. **The header contract in `common/qw36.h` and `common/qw36_gpu.h` is
   frozen.** If you need to change a signature, post in the PR thread first.
3. **CPU reference (`common/qw36.c`) is the source of truth.** Every GPU
   kernel must match it bit-for-bit at fp32 on `tests/golden_*.bin` before
   the PR lands.
4. **No `printf` debugging left in.** Use `qw36_log` (TODO: add).
5. **Build before declaring done.** `make amd|metal|cuda` must succeed on
   the relevant host.

## Claude owns (host-side C, deterministic)

| File                          | Status   | Notes                                  |
|-------------------------------|----------|----------------------------------------|
| `common/qw36.h`               | ✅ done  | Public API frozen.                     |
| `common/qw36_gpu.h`           | ✅ done  | Backend vtable frozen.                 |
| `common/qw36.c`               | 🟡 stub  | CPU reference forward pass + sampling. |
| `common/qw36_gguf.c/h`        | 🟡 stub  | GGUF v3 reader. Mmap-based.            |
| `common/qw36_tokenizer.c/h`   | 🟡 stub  | BPE tokenizer.                         |
| `common/qw36_cli.c`           | 🟡 stub  | `main`, arg parsing, REPL.             |
| `Makefile` (top)              | ✅ done  | Dispatches to `amd/`, `metal/`, `cuda/`. |
| `tests/`                      | 🟡 stub  | Golden-vector harness.                 |
| `tools/download_model.sh`     | 🟡 stub  | HF model download script.              |

## Codex owns (GPU kernels)

| File                          | Status   | Reference                                  |
|-------------------------------|----------|--------------------------------------------|
| `amd/qw36_amd.cpp`            | 🟡 stub  | HIP implementation of `qw36_gpu_backend`.  |
| `amd/Makefile`                | 🟡 stub  | `hipcc` build.                             |
| `metal/qw36_metal.m`          | 🟡 stub  | Objective-C host, MTLComputeCommandEncoder.|
| `metal/qw36_metal.metal`      | 🟡 stub  | Metal Shading Language kernels.            |
| `metal/Makefile`              | 🟡 stub  | `xcrun metal` + `clang`.                   |
| `cuda/qw36_cuda.cu`           | 🟡 stub  | CUDA kernels + host wrappers.              |
| `cuda/Makefile`               | 🟡 stub  | `nvcc` build.                              |

### Kernel checklist (per backend)

Each backend must implement, in this order of priority:

1. `init` / `destroy` — pick the default device, set up a stream/queue.
2. `upload` / `download` / `alloc` / `free` — straight `cudaMemcpy` / Metal
   `newBufferWithBytes` / HIP equivalents.
3. `rmsnorm` — one block per token, fp32 accumulator, vector-load `weight`.
4. `matmul` — naive tiled fp16 → fp32 accumulate is fine for v0. Targets:
   `[hidden, hidden]`, `[hidden, intermediate]`, `[hidden, vocab]`.
5. `embedding_lookup` — trivial copy.
6. `residual_add` — trivial vector add.
7. `swiglu_mlp` — fused gate/up matmul then silu*up then down matmul. v0
   can compose three matmuls + an elementwise op.
8. `attention` — fused: project QKV, apply q_norm/k_norm, rotate (RoPE
   partial), append to KV cache at `seq_pos`, score against history,
   softmax, weighted sum. **Hard part — match CPU reference at fp32.**
9. `moe_forward` — last. v0 can fall back to looping over experts.

### Numerics

- All accumulations in **fp32**. Weights stay in storage dtype (bf16/f16).
- RMSNorm: `x * rsqrt(mean(x^2) + eps) * weight`. eps from config.
- RoPE: apply to the first `head_dim * partial_rotary_factor` components.
  Use base = `rope_theta`. See `qwen35-spec/src/lib.rs::rope_inv_freq`.
- Softmax: subtract max, fp32 exp, fp32 sum, divide. Causal mask: positions
  `> seq_pos` are `-inf`.
- SiLU: `x * sigmoid(x)`.

## Hand-off protocol

When you finish a backend, append a row to the table below and commit. The
other agent picks up from there.

| Date       | Agent  | Backend | Commit  | Notes |
|------------|--------|---------|---------|-------|
| 2026-05-20 | Claude | scaffold  | (init)   | All stubs in place. |
| 2026-05-20 | Claude | cpu       | -        | Full GGUF v3 loader + Q4_K/Q6_K/Q8_0 dequant + BBPE tokenizer + vanilla GQA forward + CLI; `qw36_cpu --info` works on Qwen3.5-0.8B-Q4_K_M. |
| 2026-05-20 | Codex  | metal     | -        | init/destroy, host↔device, rmsnorm, matmul, embedding, residual, silu·mul, head_norm_rope, kv_append, attn_scores/softmax/combine kernels — all smoke-tested. |
| 2026-05-20 | Codex  | cuda      | -        | init/destroy, memory ops, mirror of metal kernels (qw36_rmsnorm_kernel, qw36_matmul_kernel, qw36_head_norm_rope_kernel, etc). Not compiled on this Apple machine. |
| 2026-05-20 | Codex  | amd       | -        | Same set as cuda, HIP variant. |
| 2026-05-20 | Claude | engine    | 218fb50  | matmul→backend (10x speedup, Metal hits 11 tok/s). |
| 2026-05-20 | Claude | engine    | a7e691c  | tokenizer special-token detection (`<|im_start|>`). |
| 2026-05-20 | Claude | engine    | c479cad  | Q5_K dequant (DN was reading zeros silently). |
| 2026-05-20 | Claude | engine    | 727bb10  | mRoPE per-axis sections from `qwen35.rope.dimension_sections`. |
| 2026-05-20 | Codex  | engine    | d24bbbd  | persistent GPU state (task #28): residual / rms / Q/K/V / KV cache / MLP scratch / logits live on device. DN/MoE still bridge through host. |

## Reference: agent-infer Metal kernel for Gated Delta Rule

For task #22, agent-infer's exact Metal kernel for the per-step gated
delta rule lives at:

  `../agent-infer/crates/mlx-sys/src/mlx_qwen35_model.cpp:203-275`

It's a `fast::metal_kernel("gated_delta_step", ...)` call. Key facts:

- **State layout**: `[B, Hv, Dv, Dk]` (Dk innermost). Different from our
  CPU code's `[Hv, Dk, Dv]` — adapt when porting to GPU.
- **Threadgroup**: x=32 (simd lanes split Dk), y=Dv, z=B*Hv.
- **Per-thread state**: `Dk/32` floats; uses `simd_sum` for cross-lane
  reductions on kv_mem and the output dot product.
- **bf16 in / fp32 state / bf16 out**.
- **GQA**: `hk_idx = hv_idx / (Hv/Hk)` (group, not modulo).

QK norm scaling (`mlx_qwen35_model.cpp:395`):
```
q = fast::rms_norm(q_raw, std::nullopt, 1e-6f) * (1/sqrt(d))²
k = fast::rms_norm(k_raw, std::nullopt, 1e-6f) * (1/sqrt(d))
```
Equivalent to: L2-normalize then `q *= 1/sqrt(d)` (attention temperature).

`compute_g_beta` (`mlx_qwen35_model.cpp:377`):
```
ab = a_raw + dt_bias
softplus = ab if ab > 20 else log1p(exp(ab))
g = exp(neg_exp_a * softplus)         # neg_exp_a = -exp(a_log), precomputed
beta = sigmoid(b_raw)
```

## Path to 85 tok/s

Current state on M-class GPU, Qwen3.5-0.8B-Q4_K_M:
- CPU build:   ~1.7 tok/s
- Metal build: ~12 tok/s (after #28 GPU-resident state)
- Target:      85+ tok/s (agent-infer)

Remaining bottleneck: DN (18/24 layers) and MoE still bridge through host
in qw36_forward, forcing many GPU→CPU→GPU sync points per token.

Open work:
1. **task #22**: port gated_delta_step to Metal (kernel source available
   above) so the DN block stays on GPU.
2. **task #23**: same for MoE expert dispatch.
3. Wire `backend->begin_batch` / `end_batch` around the whole forward
   once no host bridges remain — single MTLCommandBuffer per token.
4. **task #21**: rmsnorm / residual_add / swiglu / attention all via
   `_dev` variants exclusively (some already done in d24bbbd).

## Known blocker for Qwen 3.6 end-to-end

Qwen 3.5 / 3.6 models use a **hybrid architecture**: some layers are
vanilla GQA (`blk.X.attn_q.weight` etc.), others are **Gated DeltaNet**
(`blk.X.attn_qkv.weight` fused + `blk.X.ssm_*` tensors).

For Qwen3.5-0.8B specifically: 6/24 layers vanilla GQA, 18/24 DeltaNet.

The runtime currently supports vanilla GQA layers only. The DeltaNet path
needs:
- Per-layer attention-kind detection in `qw36_engine_open` (track in
  `qw36_config.layer_types`).
- New tensor name BINDs for the SSM tensors (`ssm_conv1d`, `ssm_alpha`,
  `ssm_beta`, `ssm_a`, `ssm_dt.bias`, `ssm_norm`, `ssm_out`,
  `attn_qkv`, `attn_gate`).
- A `delta_net_decode_f32` reference op (~200 lines): short-conv state
  buffer + per-head recurrent state update + gated RMSNorm output.
- Backend vtable extension: `delta_net_decode(ctx, ...)`.

Until that lands, the CPU forward returns -2 on a DeltaNet layer.
