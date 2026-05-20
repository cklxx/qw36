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
