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
| 2026-05-20 | Codex  | metal     | -        | Opt-in `QW36_METAL_FP16_WEIGHTS=1`: large lazy weights materialize as fp16 and run MPS half GEMV with f32↔f16 staging; q/k norm+RoPE+KV append and score/softmax/combine use fused decode kernels. Qwen3.5-0.8B `Hello -n 50`: 79.6 tok/s decode. |
| 2026-05-20 | Codex  | metal     | -        | task #23 first pass: Metal MoE route/top-k, expert gate/up, down+combine kernels wired through `backend->moe_forward`; CPU fallback remains. Built on Apple Silicon; no local MoE GGUF found for numeric validation. |
| 2026-05-20 | Codex  | metal     | -        | task #30: full decode-attention fp16 fast path folds q/k RMSNorm + RoPE + KV append + score/softmax/combine into one MSL dispatch per vanilla layer, using `simd_sum` two-stage head reductions. Qwen3.5-0.8B `Hello`: 82.1 tok/s (`n=50`), 81.8 tok/s (`n=128`). `tests/precision_cpu_vs_metal.sh` passes on the fp32 path. |
| 2026-05-20 | Codex  | metal     | -        | 200 tok/s follow-up: Metal fp16 KV cache now allocates device KV buffers as F16 when `QW36_METAL_FP16_WEIGHTS=1` (override with `QW36_METAL_FP16_KV=0`) and uses a dedicated `qw36_attn_decode_fused_f16kv_f32` kernel. `qw36_prefill` skips final norm/lm_head/logit download for all non-final prompt tokens. BF16 MPS GEMV was tested and rejected: MPS aborts with "Only 32b and 16b floating point data and 8-bit integer types are supported." Naive fused MLP down-proj was tested and rejected: 44.9 tok/s vs ~82 tok/s. |

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

Tracking on Qwen3.5-0.8B-Q4_K_M, M-class GPU:

| step | commit | decode tok/s | notes |
|------|--------|--------------|-------|
| CPU baseline                | —        | 1.7  | lazy quant matmul |
| `backend->matmul` (MPS gemv)| 218fb50  | 11   | first Metal hook |
| MPS gemv cache + batch infra| 51b2778  | 13   | per-shape pipeline cache |
| persistent GPU state        | d24bbbd  | 55   | task #28 — KV cache + scratch on device |
| fused GDR Metal kernel      | 1ecb8c4  | 55   | DN block stays on GPU |
| bucketed MTLBuffer pool     | 2cd9da4  | 56   | recycle transient scratch |
| fp16 weights (opt-in)       | e286627  | **81** | `QW36_METAL_FP16_WEIGHTS=1`, MPS half GEMV |
| Metal MoE first pass        | efc7f6c  | 81   | task #23 — no MoE in 0.8B so no-op here |
| full fused decode attention | -        | **82** | task #30 — prep+score+softmax+combine in one fp16 fast-path MSL dispatch |
| fp16 KV + prefill skip      | -        | **82** | dedicated f16-KV attention path; prefill avoids unused logits on non-final prompt tokens |

Sustained (longer ctx) at fp16:
- n=50:  83.1 tok/s best observed under load
- n=128: 81.3 tok/s with or without fp16 KV at short context

Decode rate now holds around 80-82 tok/s through `n=128` on this busy
workstation. Closing the upgraded 200 tok/s target likely needs:

1. **GPU-side quantized matmul**: current fp16 path still materializes
   all lazy weights to half and drives many MPS GEMVs. Agent-infer-level
   speed likely needs dequant+matvec fusion for Q4_K/Q5_K/Q6_K rows.
2. **Real fused MLP**: a naive one-thread-per-output MSL down projection
   is much slower than MPS. A useful fusion needs tiled/simdgroup matvec
   or an MPSGraph/MLX-style fused graph.
3. **Long-context validation**: fp16 KV is wired, but at the current
   9-token prompt + 128 decode benchmark the KV scan is too small to
   move the needle. Re-test at 1K-4K context before optimizing further.
4. **Command buffer streaming**: decode still needs one CPU-visible
   logits download per generated token. Prefill now skips unused logits;
   deeper streaming needs an API that separates "advance state" from
   "produce logits".

`task #30+` tracks the 200 tok/s performance ladder.

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
