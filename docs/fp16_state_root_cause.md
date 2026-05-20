# fp16 state root cause

agent-infer 全程用 bf16 残差/KV/GDR 激活，RMSNorm、sigmoid/silu 先转 fp32
中间值，最后才写回 bf16；matmul 也走自定义 quantized_matmul，不走 MPS
fp16->fp16 GEMV。我们的 step-0 翻转发生在 `x_rms_dev` fp16 喂给 MPS
projection 后，基本不是 RMSNorm 求和问题，而是 fp16 边界存储 + MPS half
GEMV 路径的数值/ABI问题；优先试 bf16 state 或自定义 GEMV。

Repro:

```bash
QW36_METAL_FP16_WEIGHTS=1 QW36_METAL_FP16_EDGES=1 \
  ./qw36_metal -m <model.gguf> -p Hello -n 1 --no-special --debug-top 5
```

For bisection, use `QW36_METAL_FP16_XRMS=1` or `QW36_METAL_FP16_Q=1`
instead of `QW36_METAL_FP16_EDGES=1`.

Current result: `QW36_METAL_FP16_XRMS=1` keeps the expected step-0 top-1
`,` (id 11). `QW36_METAL_FP16_Q=1` flips top-1 to `uela` (id 22995).
So the bad edge is the attention output buffer `q_dev` feeding `o_proj`,
not the RMSNorm output feeding q/k/v/gate/up projections.

`QW36_METAL_F16_GEMV_QUAD=1` adds a qmv_quad-style fp16 GEMV to replace
MPS on selected row counts. It does not fix the `q_dev=fp16` divergence,
so the issue is not just an MPS half-input quirk; fp16 attention-output
storage itself is too lossy or the writer/reader layout still needs a
separate diff.

## MLX vs qw36 perf analysis (session 2026-05-20)

| model + path                          | tok/s | notes |
|---------------------------------------|------:|-------|
| MLX Qwen3.5-0.8B-MLX-4bit             | 244   | reference |
| qw36 Qwen3.5-0.8B-Q4_K_M + fp16 W     |  85   | this branch |
| MLX Qwen3.5-4B-MLX-4bit               |  74   | reference |
| qw36 Qwen3.5-4B-Q4_K_M + fp16 W       |  20   | output degenerate at 4B |
| MLX Qwen3.5-9B-MLX-4bit               |  32   | reference |
| MLX Qwen3.6-35B-A3B-4bit (MoE)        |  56   | A3B = 3B active |

`QW36_METAL_TIMING=1` reveals qw36 at 85 tok/s is **100% GPU-bound**:
  commit (CPU encoding) = 0.01 ms/step
  gpu  (waitUntilCompleted) = 11.9 ms/step

So the 85 → 200 gap is entirely on-GPU work. Theoretical floor for
fp16 0.8B (~1 GB resident weights × 200 GB/s mem bandwidth) = 5 ms/step
= 200 tok/s. We are at 42% of bw limit.

### Why MLX gets there: `qmv_quad`
MLX `backend/metal/kernels/quantized.h` `qmv_quad_impl`:
- 1 SIMD per threadgroup (32 threads), not 256
- 8 output rows per quadgroup, 8 quads per SIMD → 64 output rows / TG
- per-thread D/4 values via templated `load_vector`, packed uint32 reads
- `quad_sum` (4-way) reduction, no threadgroup barriers

Our Q4_K kernel `qw36_matmul_q4_k_f32`: 256 threads × 1 output row /
TG, threadgroup reduction with simd_sum + cross-simd. Higher
threadgroup count, more reduction work — 50 tok/s vs MLX's 244.

### But MLX 244 tok/s is NOT from `qmv_quad` alone
agent-infer's `gguf_quantized_matmul_cpp` (GGUF Q4_K input) uses a
256-thread TG just like ours, and at decode batch=1 MLX falls through
to a similar path. The 3× gap is from MLX's **graph compiler fusing
operations** — `fast::rms_norm + matmul`, `precise_sigmoid_mul`, etc.,
get JIT-compiled into one or two big kernels per layer, not 11 small
ones.

### Realistic path to 200 tok/s in qw36
1. **Reduce dispatch count from 264 → ~100 / token** via kernel fusion:
   - rmsnorm + matmul fused (per-output-row kernel reads x, computes
     rmsnorm scale, then dot-products against W)
   - silu_mul fused with down_proj (custom GEMM that reads gate, up,
     and W down in one pass)
   - residual_add + next-layer rmsnorm fused
   Each fusion saves a full kernel dispatch (~45 us). Aim for 5 ms /
   token = 200 tok/s.
2. **qmv_quad-style GEMV** for the matmul side: 32 threads/TG, 8 rows
   per quadgroup, vector load. Replace MPS for the fp16 fast path so
   we control encoding (and can fuse).
3. **Persistent compute encoder** across all kernels in a layer once
   matmul leaves MPS. Saves the `[cb computeCommandEncoder]` /
   `[enc endEncoding]` brackets that currently bracket every dispatch.

Estimated effort: a focused kernel-rewrite week to land all three.
