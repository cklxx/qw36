# Q4_K kernel 真正快的方案：深度调研任务

## 上下文
两轮失败：
- 任务 G (qmv_quad fp16): 慢于 MPS
- 任务 Q (MMA fp16): M=1 浪费 7/8 tile
- 任务 R (Q4_K qmv_quad): 慢于旧 256-thread kernel (docs/q4k_qmv_quad_failed.md)

**用户原话**："找到正确的方案细节很重要"。不再急着写 kernel，先调研清楚。

## 核心矛盾的数字
当前观察 (commit bea7be3 profile 数据, 64 token decode bench)：

| kernel | shape | gpu/call avg | 注 |
|---|---|---:|---|
| **MPS fp16 GEMV** | 1024x3584 (down) | **34 us** | bw ~110 GB/s 等效 |
| **MPS fp16 GEMV** | 7168x1024 (gate_up) | **61 us** | bw ~240 GB/s 等效 |
| MPS fp16 GEMV | 248320x1024 (lm_head) | 1967 us | bw ~260 GB/s |
| 老 Q4_K kernel | 3584x1024 | 91 us | 2.7× MPS,但 weight 是 fp16 的 1/3.55 |
| qmv_quad Q4_K | 3584x1024 | 235 us | 6.9× MPS, 失败 |

**bw 等效计算**：
- 1024x3584 fp16 weight = 7.3 MB. 34 us → 215 GB/s
- 7168x1024 fp16 weight = 14.7 MB. 61 us → 240 GB/s
- 248320x1024 fp16 weight = 508 MB. 1967 us → 258 GB/s

→ MPS GEMV ~85% of 300 GB/s peak (Apple M-class)，**已接近 hardware ceiling**.

老 Q4_K kernel 1024x3584：weight 是 Q4_K 144*14=2016 bytes/row × 3584 = 7.2 MB. 91us → 79 GB/s, 26% peak. **kernel 浪费 ~70% bw**.

## 关键问题（先研究，再写 kernel）

### Q1: 老 256-thread Q4_K kernel 为什么 only 26% bw eff?
hypothesis：3584 个 TG × 256 thread = ~1M thread, threadgroup reduction barrier 太多。每 TG 只读 7.2 MB / 3584 = 2 KB weight + 4 KB threadgroup mem (scales cache)。读模式可能不 coalesced。

**调研任务**：用 Metal `command buffer GPU times` per kernel dispatch (我们已有 perf 框架), 然后用 `MTLCounterSampleBuffer` 或 Xcode GPU Frame Capture 分析单个 TG 内 thread 在哪卡：barrier wait vs memory wait？

如果 thread 主要等 barrier — 减 barrier (qmv_quad 思路, 但 GGUF Q4_K decode 重)
如果 thread 主要等 memory — 改 read 模式 (vectorize / coalesce)

### Q2: MPS GEMV 怎么做到 ~85% bw?
MPS 私有 kernel — 看不到源码。但可以从 shape×time 反推 thread layout：
- 1024x3584 / 34us 假设 256 thread = 7.2MB / 34us = 215 GB/s, per thread 8.4 GB/s
- 假设 1 TG = 32 thread (1 SIMD), 256 cols / 32 = 8 cols/thread coalesced read
- 多 SIMD per TG = ? 

**调研任务**: 用 `xcrun simctl ... gpu-capture` 或 Instruments 看 MPS kernel 的 grid layout / TG count / threadgroup memory usage。或读 Apple docs `MPSMatrixVectorMultiplication.h` 提到的内部使用。

### Q3: GGUF Q4_K decode 的"真" overhead
每元素：1 byte read + 4-bit unpack + 1 sub-block scale lookup + 1 sub-block min lookup + 2 muladds.

- byte read: 1 cycle/elem (cached coalesced)
- nibble extract: 1 cycle (mask + shift)
- scale/min lookup (cached): 2 cycles (2 mem ops)
- fma: 1 cycle
= ~5 cycles/elem

对比 MLX affine: 1 byte + 1 nibble + 1 mul + 1 add = 4 cycles/elem. **几乎一样!**

→ 不是 decode cost 慢，是 thread layout / memory access pattern 慢。

### Q4: MLX qmv_quad 真正快在哪
读 `quantized.h:693-747` 的 qmv_quad_impl + `qdot` 实现，关注：
1. `load_vector` 怎么 vector load 32-wide x 元素 (用 half4 or stride packed)?
2. `qdot` template 怎么把 4-bit weight 解 packed (用 `bit_cast` 或 SIMD shuffle)?
3. quad_sum 替代 simd_sum 怎么省 barrier?
4. 8 outputs per quad 怎么 amortize weight row stride?

→ 写到 docs/q4k_kernel_design_v2.md，先有清晰 design pseudo-code 再写 MSL.

### Q5: 实测 Apple Silicon GPU 在 M=1 GEMV 的 bw upper bound
写 microbenchmark：纯 memcpy 1.6GB GPU → GPU (用 blit encoder + 自己 compute kernel)。看实测 bw 是多少。MPS 85% 是否对应这个 bound。

如果 bw bound ~250 GB/s, 那 Q4_K kernel 在 7.2MB weight 上的最快是 ~29us。比 MPS fp16 同 shape (34us) 快 ~15%. 收益小但有可能。

→ Q4_K kernel 设计目标：1024x3584 < 50us, 7168x1024 < 80us. **现实目标，不再追 100us range**.

## 设计要求 (写 design doc 时遵循)

`docs/q4k_kernel_design_v2.md` 应包含：
1. **target 数字** (per-shape us)
2. **TG / thread / memory layout 决策树**：
   - TG count: ?
   - threads per TG: ? (32/64/128/256?)
   - rows per TG: ?
   - threadgroup memory usage per TG: ?
3. **关键 inner loop pseudo-code**：
   - x 怎么 load (vector? scalar?)
   - weight 怎么 read + dequant (packed? per-byte?)
   - sub-block scale 怎么访问 (threadgroup memory? thread-local? register?)
4. **reduction strategy**: simd_sum / quad_sum / no reduction (multi-row per TG)?
5. **预期问题** 和 mitigation

## 实施流程
1. **调研 phase** (2-3h): 回答 Q1-Q5, 写 design doc
2. **设计 review** (我看了 design doc 再批准 implement)
3. **实现 phase** (2-3h): 一个 K specialization 先 (K=1024 most common)
4. **bench 验证** (1h): 必须 ≥ MPS fp16 on same shape
5. **如果 ≥ MPS**: 扩 Q5_K + 其他 K, 默认开
6. **如果 < MPS**: 写失败 doc, 总结学到的，关闭这条路

## 当前任务给 codex
**只做 phase 1 (调研 + design doc)**。不要写 kernel。
写完 `docs/q4k_kernel_design_v2.md` commit，我看了再决定是否进 phase 3.

调研工具：
- Xcode GPU Frame Capture (./qw36_metal 跑时 capture)
- `xcrun simctl spawn ... metal-profiler` 如果可用
- 读 MLX/agent-infer 源码再读
- microbench (Q5)

如果调研发现 fundamental 不可行 (例如 MPS 用专属 Apple internal kernel, 我们 user-mode kernel 撞不到), 把这写进 design doc 作 "可行性结论：不"。

