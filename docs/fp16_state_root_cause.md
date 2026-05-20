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
