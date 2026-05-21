# llama.cpp Metal 推理深度调研

调研对象：[`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) 当前主线
（2026-05-21 拉取），`ggml/src/ggml-metal/` 目录，21k 行源码。

**核心问题**：llama.cpp 用同一套 GGUF K-quant（Q4_K / Q5_K / Q6_K）格式 —— 跟
qw36 是一样的磁盘 layout，没有 MLX 那种 flat affine —— 为什么 Metal 上跑得不
慢？我们之前 `docs/q4k_qmv_quad_failed.md` 里得出的结论是 "K-quant layout 对
SIMD 不友好"，那 llama.cpp 怎么解决的？

**结论先行**：llama.cpp 没有 repack。靠的是 **kernel 端 6 个具体技巧** 把
K-quant 的 layout 劣势全 amortize 掉了。这份文档逐一拆解。

---

## 0. Headline 数据

llama.cpp 的 Metal 在 M1/M2 上 Qwen-0.5B Q4_K_M 的 decode 大致 130–160 tok/s
区间（公开 benchmark 报道 + 同台机器复现）。qw36 当前 **native Q4_K 58 tok/s
/ AFFINE32 repack 85 tok/s**。差距主要在 GEMV kernel 的 6 个技巧上 —— **不是
靠 ANE / 不是靠 repack / 不是靠 MMA**，是 GEMV kernel 写得更细。

参考实现 path：

```
ggml/src/ggml-metal/
├── ggml-metal.metal          (10699 行)  ← 全部 kernel
├── ggml-metal-ops.cpp        (4622 行)  ← op dispatch / pipeline 挑选
├── ggml-metal-device.cpp     (2059 行)  ← pipeline 创建 / function constant
├── ggml-metal-device.m       (1829 行)  ← MTL pipeline 缓存
├── ggml-metal-context.m      (739 行)   ← 命令缓冲区 / 多线程编码
└── ggml-metal-impl.h         (per-dtype NR0/NSG 调优常数)
```

---

## 1. Q4_K GEMV kernel —— 全部秘密在这

文件：`ggml-metal.metal:7783-7889`

函数模板：`kernel_mul_mv_q4_K_f32_impl<nr0>`，由 host-name
`kernel_mul_mv_q4_K_f32` (line 7891-7902) 实例化。

### 1.1 SIMD lane 切分

```cpp
const short ix = tiisg/8;  // 0...3  ← 4 个 lane-group
const short it = tiisg%8;  // 0...7  ← group 内 8 lane
const short iq = it/4;     // 0 或 1
const short ir = it%4;     // 0...3
```

**32 lanes 切成 4×8**：4 个 lane-group 各自处理一个 QK_K=256 super-block，
group 内 8 个 lane 协作处理 sub-blocks。这样**单 SIMD group 一次 outer-loop
iter 处理 4 个 QK_K block = 1024 个 K-维元素**，远超我们的 32-lane × 1-block
模式。

### 1.2 仿射 factoring —— 最关键的技巧

Q4_K 的解码公式按"朴素写法"是：
```
val[i] = q[i] * sub_scale - sub_min
result += y[i] * val[i]
```

llama.cpp 把它**代数重排**成：
```
result = sub_scale * sum(q[i] * y[i]) - sub_min * sum(y[i])
                                              ^^^^^^^^^^^
                                  只要 sum(y) 算一次，min 就只乘一次
```

代码（`ggml-metal.metal:7867-7871`）：

```cpp
sumf[row] += dh[0] * ((acc1[0] + 1/256 * acc1[1]) * sc8[0] +
                      (acc1[2] + 1/256 * acc1[3]) * sc8[1] * 1/16 +
                      (acc2[0] + 1/256 * acc2[1]) * sc8[4] +
                      (acc2[2] + 1/256 * acc2[3]) * sc8[5] * 1/16) -
             dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                      sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                      sumy[k] = sum(y_chunk_k)，每 32 元素只算 1 次
```

`sumy` 在 `:7834-7839` 用 `for (i = 0; i < 8) sumy[0] += yl[i+0]; ...` 累一遍。
**每 32 元素一个 sub-block，min 项的 cost 从 32× mul 降到 1× mul** —— 直接抵消
掉 K-quant per-32-sub-scale 的劣势。

这条 trick **qw36 完全没用** —— 我们的 `dq_q4_K_block` 还是逐元素
`q*scale - min`，浪费 32× 的 min 乘法。

### 1.3 bit-pack 读取

权重 `q1` 按 `uint16_t*` 读：

```cpp
for (short i = 0; i < 4; ++i) {
    acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);   // 低 nibble
    acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);   // 隔 nibble
    acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);   // 中 nibble
    acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);   // 高 nibble
    ...
}
```

**一次 uint16 load = 4 个 4-bit quant 同时被处理**。注意 `acc1[1]` 用的是
`q*16`（因为 `& 0x0F00` 保留高位没右移），最后 `acc1[1]` 被乘以 `1/256`
（其实是 `1/16 / 16`）补回来。**用 immediate scale 抵消，省了 shift 指令**。

`acc1` 和 `acc2` 都是 `float4`，意味着 4 lane 在硬件 SIMD 内并发计算 4 个独立
累加器 —— 等价于 4-way ILP。

### 1.4 sub-scale 的 6-bit 解包，全在寄存器

Q4_K 的 12-byte `scales[12]` 编码 8 个 sub_scale + 8 个 sub_min，每个 6-bit。
llama.cpp 用三个 mask 在线展开：

```cpp
constexpr uint16_t kmask1 = 0x3f3f;
constexpr uint16_t kmask2 = 0x0f0f;
constexpr uint16_t kmask3 = 0xc0c0;

sc16[0] = sc[0] & kmask1;
sc16[1] = sc[2] & kmask1;
sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
```

`scales[12]` 总共 96 bit，按"低 6 + 高 2 拆开"的方式 packed。这 4 行 unpack
**整个 super-block 8 个 sub-scale 一次性算完，存在 thread-local register**。
后续整个 super-block 的所有 sub-block 都用这套 register-resident scales，
**不再 access device memory 拿 scale**。

这就把 K-quant "per-32 scale lookup" 的劣势降到 0 —— super-block 范围内只 load
一次 scale block。

### 1.5 多 row per SIMD group（**ROWS_PER_SIMD**）

`nr0` 是模板参数。`ggml-metal-impl.h:51` 定义：
```
#define N_R0_Q4_K 2     // 每个 SIMD group 算 2 个 output row
#define N_SG_Q4_K 2     // 每个 TG 2 个 SIMD group
```

即**一个 64-thread TG 一次算 4 行 output**，共享同一份 `y`/`sumy`/sub_scale
读取。这是我们 `docs/moe_design.md` SwitchGLU 设计里 `ROWS_PER_SIMD` 的同名
机制 —— llama.cpp 在所有 GEMV kernel 里都用。

per-dtype 调优（impl.h 全表）：

| dtype | NR0 | NSG | 一个 TG 算几 row | 备注 |
|-------|-----|-----|----------------|------|
| Q4_0  | 4   | 2   | 8              | 简单 affine，吃得下 8 row |
| Q4_K  | 2   | 2   | 4              | dequant 重，4 row 够 |
| Q5_K  | 1   | 2   | 2              | dequant 最重（高位单独 buffer），降到 2 row 避免寄存器外溢 |
| Q6_K  | 2   | 2   | 4              | 跟 Q4_K 一档 |
| Q8_0  | 2   | 4   | 8              | dequant 几乎 0，4 SG 拉满 |

**每种 quant 单独调过**。这是 qw36 完全缺的层 —— 我们 `qmv_fast` 一刀切都是
1 row。

### 1.6 收尾用 simd_sum

`:7884`：
```cpp
float sum_all = simd_sum(sumf[row]);
if (tiisg == 0) dst_f32[first_row + row] = sum_all;
```

跨 32 lane 累加用硬件 `simd_sum`，1 条指令，没有 threadgroup memory，没有
barrier。这条 qw36 也用，没问题。

---

## 2. mul_mv_ext —— 小 batch 专用 kernel（qw36 没有）

`ggml-metal-ops.cpp:2061-2161` 的 dispatch 决策：

```cpp
if (op->src[1]->type == GGML_TYPE_F32 && ne00%128 == 0 && (
        ((Q4_0/Q5_0/Q8_0/...) && ne11 >= 2 && ne11 <= 8) ||
        ((Q4_K/Q5_K/Q6_K/...) && ne11 >= 4 && ne11 <= 8)
    )) {
    // 走 mul_mv_ext 小 batch kernel
}
```

**ne11 ∈ [4,8]** 用一套专门的 kernel：threadgroup 几何按 `nxpsg, nypsg, r1ptg`
动态调（`:2099-2129`）：

```cpp
nxpsg = (ne00 % 256 == 0 && ne11 < 3) ? 16 :
        (ne00 % 128 == 0)              ? 8 : 4;
nypsg = 32/nxpsg;             // SIMD group 内沿 row 的 lane 数
r0ptg = nypsg * nsg;          // 每 TG 几 row
r1ptg = (ne11 switch case)    // 每 TG 几 column（即 batch 维度）
```

也就是说 batch=2 vs batch=8 走**不同 TG 拓扑** —— 当 batch=2 时
`nxpsg=16, nypsg=2`，每 SIMD group 算 2 row × 2 col；batch=8 时
`nxpsg=8, nypsg=4`，每 SIMD group 算 4 row × 4 col。**同一段 K 维 inner loop
被 batch 维度的多个 token amortize 掉**。

**qw36 在 batch ∈ [2,8] 时完全没有专门路径** —— 我们要么是 ne11=1 GEMV，要么是
MPS GEMM。prefill 一旦走 chunked 长度（比如每次 4 token），llama.cpp 这条
kernel 直接吃下来，我们没有。

---

## 3. mul_mm —— 大 batch 用 Apple MMA tensor core

`ggml-metal.metal:9387-9503`，激活条件 `ne11 > ne11_mm_min`（默认 8）。

关键代码 (`:9435-9442`)：

```cpp
mpp::tensor_ops::matmul2d<
    mpp::tensor_ops::matmul2d_descriptor(
        NRB, NRA, N_MM_NK_TOTAL, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
    execution_simdgroups<N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y>> mm;
```

`mpp::tensor_ops::matmul2d` 是 Apple 的 **MetalPerformancePrimitives**（M1+ 才
有），底层吃 `simdgroup_half8x8` MMA 指令。

**Quant 怎么进 MMA？** ——
1. Threadgroup 内的 thread 并行 dequant 一个 16-element block 到 threadgroup
   memory (`:9465-9476`)：`dequantize_func(row_ptr + block_idx, il, temp_a)`
   是模板参数（per-dtype，例如 `dequantize_q4_K`，在 `:680-697`）。
2. `threadgroup_barrier` 同步。
3. `mm.run(mB, mA, cT)` —— MPP 在 dequanted tile 上跑 MMA。

**就是说 GGUF quant 权重通过 on-the-fly tile dequant 进 MMA 单元**。这套
qw36 完全没有 —— codex 试过的 Q (MMA fp16 GEMV) 是 M=1 GEMV 用 MMA，
8×8 tile 浪费 7/8 所以更慢；而这里 llama.cpp 只在 **M > 8 prefill 时**才用
MMA，刚好绕开 M=1 浪费的问题。

---

## 4. Flash Attention —— 量化 KV 直接送进 kernel

`kernel_flash_attn_ext` (`:6491-6516`) + impl 模板 `:5853-6490`。

### 4.1 模板参数化的核维度 + K/V dtype

```
template <typename q_t, q4_t, q8x8_t, k_t, k4x4_t, k8x8_t, v_t, v4x4_t, v8x8_t,
          qk_t, qk8x8_t, s_t, s2_t, s8x8_t, o_t, o4_t, o8x8_t,
          typename kd4x4_t, short nl_k, void(*deq_k)(...),
          typename vd4x4_t, short nl_v, void(*deq_v)(...),
          short DK, short DV, short Q, short C>
```

`deq_k` / `deq_v` 是**模板函数指针** —— 编译期固化。一次性预实例化 50+ 个
host-named 变体：

```
kernel_flash_attn_ext_f16_dk128_dv128
kernel_flash_attn_ext_q4_0_dk128_dv128  ← Q4_0 KV
kernel_flash_attn_ext_q8_0_dk128_dv128  ← Q8_0 KV
kernel_flash_attn_ext_bf16_dk128_dv128
...
```

DK/DV ∈ {32, 40, 48, 64, 72, 80, 96, 112, 128, 192, 256, 320, 512, 576}，
K/V dtype ∈ {f32, f16, bf16, q4_0, q4_1, q5_0, q5_1, q8_0}。

**KV 存什么 dtype，kernel 就直接读什么 dtype** —— 不下来 dequant 再上去。
inline `deq_k(K_block, ...)` 在 simdgroup load 之后 dequant 到 threadgroup
memory，紧接着 `simdgroup_half8x8` MMA 算 Q·K^T。

### 4.2 真 MMA + 8 queries per TG

```
constexpr short Q = 8;        // 一次处理 8 个 query (prefill 时多 token)
constexpr short C = 64;       // K/V chunk size
constexpr short NW = 32;      // simdgroup width
```

Q·K^T 用 `simdgroup_half8x8` —— **每 SIMD group 8×8 fp16 tensor 乘累加，一条
硬件指令**。Online softmax 维护 `m, l, o`。这是真 FlashAttention v2 实现。

**qw36 的 flash-attn** (`qw36_attn_decode_flash_f16kv_f32`) 是 Q=1 单 token
decode 路径，没有 MMA tile —— 自己写的 fp32 累加。在长 context 上我们已经追上
了（n=2048 105 tok/s），但 prefill 长度 ≥ 8 时 llama.cpp 的 Q=8 MMA 路径会拉
开差距。

### 4.3 量化 KV cache 是头等公民

qw36 现在 KV 有 fp32/fp16/bf16/Q8_0，但 **flash-attn kernel 只识别其中一种
buffer-index layout**（kv16 vs Q8 已经写过分支）。要再加 Q4_0 KV 需要扩展同
一个 kernel 的分支。llama.cpp 的做法是**编译期 50 个独立 kernel**，每个 dtype
完全没有分支开销。

---

## 5. MoE —— kernel_mul_mv_id 即 SwitchGLU

`ggml-metal.metal:10331-10434`。

```cpp
kernel void kernel_mul_mv_id(...) {
    const int iid1 = tgpig.z/args.nei0;
    const int idx  = tgpig.z%args.nei0;
    const int32_t i02 = ((device const int32_t *) (ids + iid1*args.nbi1))[idx];
    //                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                  这一行 = "拿这个 (iid1, idx) 槽位选中的专家 ID"

    device const char * src0_cur = src0s + i02*args.nb02;
    // 把 src0 偏移到该专家的权重块
    ...
    disp_fn(args0, src0_cur, src1_cur, dst_cur, ...);
    // 调用 per-dtype mul_mv kernel（template param）
}
```

实例化（`:10421-10423`）：
```cpp
template [[host_name("kernel_mul_mv_id_q4_K_f32")]]
  kernel kernel_mul_mv_id_t kernel_mul_mv_id<
    mmv_fn<kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>>>;
```

**MoE = 一次 dispatch，z-grid 每个 (token, slot) 用 ids buffer 索引到专家
权重，然后调同一个 Q4_K GEMV kernel**。没有写专门的 MoE kernel，没有 Q8
detour，没有 1-row-per-TG。

跟 MLX SwitchGLU 完全是同一形状（只读 top-k 选中的专家行），而且**复用所有
GEMV 的 6 条优化**（仿射 factoring、bit-pack、scale register 化、ROWS_PER_SIMD
= 2、simd_sum 收尾）。

这正是 qw36 `docs/moe_design.md` 在追的设计。**llama.cpp 已经在那里了**，我们
只是没意识到 `kernel_mul_mv_id` 在做 SwitchGLU 的事。

---

## 6. 命令缓冲区 + 多线程编码

`ggml-metal-context.m:507-550`，是 qw36 没有的并发模型。

```objc
// n_cb 默认 8（可调）
for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
    id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
    [cmd_buf retain];
    ctx->cmd_bufs[cb_idx].obj = cmd_buf;
    if (cb_idx < 2 || abort_callback == NULL)
        [cmd_buf enqueue];   // 提早 enqueue，让 GPU 可以开始
}

dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async);
//             ^ GCD: n_cb 个 CPU 线程并发往 cmd_bufs[0..n] 编码
```

**关键点**：

1. **`n_cb` 个独立 MTLCommandBuffer**（默认 8），不是单 buffer 单 encoder。
2. **`dispatch_apply` 多 CPU 线程并发编码** —— 一边 GPU 在跑 cb[0]，另一边
   CPU 在编码 cb[1..n]。
3. **早 `enqueue` 晚 `commit`** —— enqueue 把 buffer 加入队列但不 commit；
   GPU 看到 enqueued buffer 后准备好资源。等编码完成 commit 时基本零延迟。

**qw36 的模型**：单 persistent compute encoder（codex commit 112e85f），单
MTLCommandBuffer per token。decode 单流时这很好；**prefill 大 batch 时
llama.cpp 的多线程编码会快**，因为 CPU encode time 在 prefill 里非负（每 op
要 `set_pipeline + set_buffer*N + dispatch`）。

我们 `FINAL_STATUS.md` 写过 "CPU encode already <0.1% of budget" —— 那是
**decode 单 token** 的尺度。Prefill 8 token × 24 layer × 10 op = 1920 个
encode call，那才是 dispatch_apply 体现价值的地方。

---

## 7. 综合对比 —— qw36 vs llama.cpp

按 hot path 走：

| 阶段 | qw36 | llama.cpp | 差距来源 |
|------|------|-----------|----------|
| **Decode Q4_K GEMV** | qmv_fast on AFFINE32 repack (85 tok/s) | 6-trick K-quant GEMV no repack | (1.2)+(1.4)+(1.5) 仿射 factoring、register-scale、ROWS_PER_SIMD |
| **Decode Q5_K** | native dq_q5_K + GEMV (没 GPU 优化) | NR0=1 NSG=2 per-dtype-tuned | per-dtype NR0/NSG 调优 |
| **Decode Q6_K** | Q6K_SCALE16 repack + qmv_fast | NR0=2 NSG=2 + factoring | 同上 |
| **小 batch prefill (n_tokens ∈ [2,8])** | 走 MPS GEMM (没量化 fast path) | mul_mv_ext 专门 kernel | (2) 完全缺这一档 |
| **大 batch prefill (n_tokens > 8)** | 走 MPS GEMM | MPP MMA + tile dequant | (3) on-the-fly dequant 进 MMA |
| **Decode attention** | flash_attn_f16kv (Q=1) | flash_attn_ext (Q=8 MMA) | Q>=8 时 MMA 优势；Q=1 上等价 |
| **量化 KV cache** | Q8_0 通过分支接入 1 个 flash kernel | 50+ 编译期实例化 flash kernel | (4) 编译期 specialize 而非 runtime branch |
| **MoE** | 一行一 thread naive，79% GPU time | mul_mv_id 复用 GEMV + ids gather | (5) SwitchGLU 形状 |
| **多 token CPU 编码** | 单 persistent encoder | n_cb 个 buffer + dispatch_apply | (6) prefill 并发编码 |

---

## 8. 对 qw36 的具体可移植 lessons

按 ROI 排序，能直接搬上 qw36 的：

### L1 — **仿射 factoring** （最高 ROI，不动 layout）

把 qw36 `dq_q4_K` 系列 dequant 重写为：
- 每 sub-block 算 `sum(y_chunk)` 一次；
- 总和 = `sum_scale * dot(q, y) - sum_min * sumy`；
- 每 32 元素少 31 个 sub-mul。

**预期影响**：Q4_K 路径从 85 → 100+ tok/s（取消 affine32 repack 的必要性，
回到 native Q4_K layout 拿同等性能）。

`docs/q4k_kernel_design_v2.md` 里的 affine32 决策是"换 layout 换性能"；这条是
**不换 layout 也能拿同样性能**。如果验证成立，可以省掉 load-time repack 的
memory 开销和 startup 时间。

### L2 — **per-dtype ROWS_PER_SIMD** （已在 SwitchGLU 路上）

我们已经写在 `docs/moe_design.md` ROWS_PER_SIMD=4。llama.cpp 实际值更保守：
Q4_K=2、Q5_K=1（register pressure 限制）、Q6_K=2、Q8_0=2。

**Action**：MoE SwitchGLU 实现时按 llama.cpp 的值起步而不是 4，可能跑得更稳。

### L3 — **mul_mv_ext 小 batch kernel** （prefill 改善）

Prefill 用 chunked 4-8 token 跑时，qw36 现在 dispatch ne11 个 GEMV
（24 layer × 6 op × 8 token = 1152 个 dispatch），llama.cpp 一次 dispatch
搞定。

**Action**：写一个 `qw36_matmul_q4k_smallbatch` kernel，参数 `(ne11, nxpsg, nypsg,
r1ptg)`。优先级中等 —— 主要影响 TTFT，不影响 decode tok/s。

### L4 — **多 KV dtype 通过编译期 specialize** （清洁度而非性能）

qw36 现在 KV fp16/bf16/Q8 通过 runtime branch 进同一个 flash kernel。如果想加
Q4 KV（task #84），与其再加分支不如学 llama.cpp 模板化。

**Action**：把 `qw36_attn_decode_flash_*` 用宏展开成 per-dtype kernel。

### L5 — **prefill 多线程编码** （prefill 大 batch 时有用）

qw36 单 encoder 的设计在 decode 上没问题。如果未来 batch>1 prefill 成为主线
（比如服务化），可以考虑加 `dispatch_apply` 风格的并行编码。当前优先级低，
单 user 单流场景不需要。

### L6 — **MMA tile-dequant GEMM**（大 prefill 用，但 qw36 用 MPS）

qw36 prefill 走 MPS GEMM，已经把 quant → fp16 转一道再做。llama.cpp 这条是
在 quant kernel 内部直接 tile-dequant 喂 MMA。**对 qw36 不是 ROI 高的方向**
（MPS 已经接近天花板），除非要去掉 MPS 依赖。

---

## 9. 重要观察 —— 我们之前的失败实验，llama.cpp 怎么躲过去的

| qw36 失败实验 | llama.cpp 等价决策 |
|---------------|---------------------|
| `docs/q4k_qmv_quad_failed.md` —— port MLX qmv_quad 到 Q4_K 慢 50% | 不 port MLX，自己写**为 K-quant layout 设计的** GEMV（仿射 factoring + register-scale unpack） |
| `docs/moe_kernel_failed.md` Attempt 1 —— 1-row-per-TG 慢 24× | mul_mv_id + NR0=2 NSG=2 = **每 TG 4 row** + bridge to per-dtype GEMV |
| `docs/moe_kernel_failed.md` Attempt 2 —— Q8-only fast path | **直接操作 Q4_K 原生 dtype**，把 dequant 嵌进 GEMV |
| codex Q —— MMA fp16 GEMV for lm_head 慢 2× | **只在 M>8 时用 MMA**（mul_mm），M=1 走专门的 GEMV |

每一条"失败"的根因，llama.cpp 都已经写在源码里了。我们的失败不是方向错，是
**抄错了哪个抽象层**（去抄 MLX 的 affine 假设，没去抄 llama.cpp 的 K-quant
原生写法）。

---

## 10. 怎么验证 —— bench 设计

下一步如果实现 L1 (仿射 factoring) ：

**对照实验**：
1. Baseline：当前 `QW36_METAL_Q4K_AFFINE32=1` qmv_fast（85 tok/s 目标）。
2. New：`QW36_METAL_Q4K_LLAMACPP_GEMV=1` 在 native Q4_K 上跑新 kernel。
3. Reference：直接跑同台 llama.cpp `./llama-bench -m Qwen3.5-0.8B-Q4_K_M.gguf -n 64`。

**预期**：
- Baseline 与 New 大致相等（85 ± 5 tok/s）→ 证明仿射 factoring 等价
  AFFINE32 repack，可省 repack 时间。
- 三者相对 llama.cpp 的差距 = 剩下 (per-dtype NR0/NSG 调优 + mul_mv_ext 小
  batch) 的潜力。

**Wallclock framing**：per AGENTS §0，必须 3 次重跑 median，`uptime` 5 min
avg < 3。`QW36_METAL_PERF=1` 单独跑确认 kernel 真的 dispatched，不是 fallback
到 MPS。

---

## 11. 参考资料

- llama.cpp Metal 源码（本文档引用的所有行号都是这份）：
  https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-metal
- ggerganov 在 K-quant 上的设计原帖：
  https://github.com/ggerganov/llama.cpp/pull/1684 （Q4_K_M 引入）
- FlashAttention v2 paper：Dao 2023 (`OP_FLASH_ATTN_EXT_NQPSG = 8` 就是 Q=8)
- 相关 qw36 文档：
  - [`docs/q4k_kernel_design_v2.md`](q4k_kernel_design_v2.md) —— AFFINE32 决策
    的来源（**未来可能不再需要**，如果 L1 验证通过）
  - [`docs/q4k_qmv_quad_failed.md`](q4k_qmv_quad_failed.md) —— MLX qmv_quad
    port 失败
  - [`docs/moe_kernel_failed.md`](moe_kernel_failed.md) —— MoE 两条死路
  - [`docs/moe_design.md`](moe_design.md) —— SwitchGLU 设计（与
    llama.cpp `mul_mv_id` 同形）
- 调研日期：2026-05-21；llama.cpp commit：master HEAD（sparse clone）。
