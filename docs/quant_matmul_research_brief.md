# 4-bit GEMV 重写：研究 brief

## 问题
当前 `QW36_METAL_QUANT_GPU=1` 走 GGUF-native Q4_K/Q5_K/Q6_K/Q8_0 matmul kernels (metal/qw36_metal.metal `qw36_matmul_q4_k_f32` 等)，**58 tok/s**，比 fp16 MPS baseline (122 tok/s) 慢 2.1×。

理论上 4-bit weight 是 fp16 的 1/3.55 字节量 → 应快 3.55× → ~430 tok/s 上限。MLX 实测 244 tok/s 是这条路径的成熟实现。

## 已知数据（profile commit bea7be3, QW36_METAL_PERF=1）
当前 quant kernel per-call avg：
```
qw36_matmul_q4_k_f32_3584x1024  91 us  (vs MPS fp16 1024x3584: 34us)
qw36_matmul_q5_k_f32_6144x1024  250 us (vs MPS fp16 7168x1024: 61us)
qw36_matmul_q4_k_f32_2048x1024  54 us  (vs MPS fp16 1024x2048: 22us)
qw36_matmul_q6_k_f32_1024x3584  65 us  (lm_head NOT quant_gpu currently)
```
quant kernel **每个 call 慢 2.5-4×**，吃光了 4-bit 带宽优势。

## 当前 kernel 设计 (qw36_matmul_q4_k_f32)
- 1 threadgroup per output row
- 256 threads per TG
- 每 thread loop K/256 = 14 元素 (K=3584)
- 每元素重新 decode block sb / local / iter / h / lane / sub  
- d/dmin/sc/mn cached in threadgroup memory after 第一次 decode (Q4_K only)
- simd_sum + cross-simd reduction

问题：256 threads ÷ 8 simd per TG，threadgroup reduction 需 multiple barriers。output 行数 N（3584-7168）很多时 threadgroup 数太多。

## 参考实现 — 必读

### MLX qmv_quad（针对 MLX-affine 4-bit quant，**这是金标准**）
- 文件: `/Users/bytedance/code/agent-infer/crates/mlx-sys/vendor/mlx/mlx/backend/metal/kernels/quantized.h:693-747`
- 调度: `/Users/bytedance/code/agent-infer/crates/mlx-sys/vendor/mlx/mlx/backend/metal/quantized.cpp:177-233`
- 设计要点：
  - **TG = 1 SIMD = 32 threads**（不是 256）
  - **8 results per quadgroup**，每个 quad = 4 threads，8 quads per SIMD = 32 threads → 8 rows × 8 quads = 64 outputs per TG
  - Grid: `MTL::Size(M, (N + bn - 1) / bn, B)`，bn = quads_per_simd × results_per_quadgroup = 8 × 8 = 64
  - `values_per_thread = D / QUAD_SIZE = D / 4` — 每 thread 处理 D/4 elements
  - `load_vector` template — bulk vector load x
  - `qdot` template — packed uint32 weight read + 4×nibble→fp32 dequant inline
  - `quad_sum(result[row])` — 4-way reduction (一条指令)
  - **NO threadgroup_barrier needed** — quad_sum 在同一 SIMD 内
- 编译时模板参数 D 是 in_vec_size，让 compiler 展开循环
- 输出: y[row * quads_per_simd] = T(partial[0])

### agent-infer gguf_quantized_matmul (针对 GGUF Q4_K/Q5_K/Q6_K **我们的格式**)
- 文件: `/Users/bytedance/code/agent-infer/crates/mlx-sys/src/mlx_bridge.cpp:620-743` (helpers) + 748-812 (kernel)
- 设计要点：
  - **TG = 256 threads**（跟我们一样）
  - 1 result per TG（跟我们一样）
  - per-element `gguf_q4_k_value(row, k)` 重新 decode (跟我们一样)
  - **比我们多的优化**：m4 variant (M>=2 batched)，每 TG 算 4 个 M batch 同时 → 节省 weight read，但 decode M=1 不用 m4。
- agent-infer 用这条**只在 prefill (M>=2)**，**decode (M=1) 走 MLX qmv_quad** 因为 MLX-affine quant 比 GGUF quant 更适合 qmv_quad。

### Apple Metal Performance Shaders Programming Guide
- `MTLDispatchType`、`simdgroup_matrix` (8×8 fp16 MMA)、quad-level intrinsic `quad_sum/quad_max/quad_shuffle`
- MMA 不适用 M=1 GEMV (codex Q 已证实)

## 任务

写一个 **GGUF Q4_K-format 的 qmv_quad-style kernel**，参考 MLX qmv_quad 的设计套到 GGUF 格式。

设计要点（针对 Q4_K format）：
1. **TG = 32 threads** (1 SIMD)
2. **每 quad (4 threads) 算 ROWS_PER_QUAD output rows**，建议 8（同 MLX）。SIMD 共 8 quad → 64 outputs per TG
3. **values_per_thread = K / 4** — 每 thread 处理 1/4 K
4. **Q4_K decoding 内 thread**：
   - 每 block (256 elem) 的 super-scale d (fp16) + dmin (fp16) + 8 sub-block scales/mins → 18 个 fp32 写到 thread-local 数组
   - K=1024 → 4 Q4_K blocks per row → cache 4 × 18 = 72 个 fp32 per thread (放 thread-local 数组，**不**放 threadgroup memory)
   - per-element dequant: 1 byte read + 1 mul + 1 sub
5. **quad_sum** for reduction → 不需要 threadgroup barrier！
6. **vector load x**: 用 `device const half *` cast to `half4` 之类，bulk read x_thread[values_per_thread]
7. Grid: `MTLSizeMake(M, (N + 63) / 64, B)` (N=output rows)，threadgroup `MTLSizeMake(32, 1, 1)`
8. 模板化 D (K) — JIT compile per shape; 调度时按 K 选 specialization

实现路径：
1. **先研究** MLX qmv_quad_impl 完整代码 + agent-infer gguf_q4_k_value/gguf_q4_scales。确认懂了再下手。
2. **写 kernel** `qw36_matmul_q4_k_qmv_quad_f32` (新增，**不删旧的**)。
3. **host dispatch**：QW36_METAL_QUANT_GPU=1 + 新 env QW36_METAL_Q4K_QUAD=1 toggle 切到新 kernel。bench 对比旧 vs 新。
4. **如果新 kernel 至少快 2×**: 默认开。然后再做 Q5_K / Q6_K 同样改写。
5. **测**：
   - make metal
   - precision_cpu_vs_metal.sh
   - Hello → "Hello! How can I help you today?"
   - 古诗 → 《秋夜》七言绝句
   - bench QW36_METAL_QUANT_GPU=1 QW36_METAL_Q4K_QUAD=1 -n 128 应 > 80 tok/s (目前 58)；好的话 > 150 (理论可)

## ROI 估计
- Q4_K 新 kernel 若达 MLX qmv_quad 效率：~250-300 tok/s 可能（理论 430，MLX 现实 244 因为 prefill 不算）。
- 若只达 MPS fp16 效率：~150-180 tok/s（4-bit 带宽优势 1.5-2×）。
- 即使保守 1.5× over MPS: ~180 tok/s。

## 注意事项
1. **不要**改 `QW36_METAL_QUANT_GPU=1` 默认行为，加新 toggle 先 bisect。
2. 同一个 kernel 输出 fp32 (y dtype)，跟现有 dispatch path 兼容。
3. quad_sum 是 Apple GPU intrinsic — `quad_sum(value)` 在每 quad (4 threads) 内求和并广播给所有 4 个 lane。
4. 各 SIMD 内的 8 quads 输出 8 个**不同** rows（不是同一个 row），所以无 cross-quad reduction needed。
5. lane → thread_index_in_quadgroup (0..3)；quad_gid → quadgroup_index_in_threadgroup (0..7)。
6. Output address: `y[tid.x * out_vec_size + row * quads_per_simd + quad_gid]` (per MLX, 但要按我们的 layout 调整)
7. Q4_K block bytes = 144。row 的 K=1024 → 4 blocks → 576 bytes / row weight.

慢慢做。这刀重了 (~2-4h)，先研究透 MLX 代码 + agent-infer 代码 + 我们当前实现，画好设计草图，再写 kernel。不要急。

