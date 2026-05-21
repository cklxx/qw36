# MLX Metal 推理深度调研

调研对象：[`ml-explore/mlx`](https://github.com/ml-explore/mlx) 主线
（2026-05-21 拉取），`mlx/backend/metal/` 目录。

**核心问题**：MLX 在同一台 Apple Silicon 上跑 Qwen3.5-0.8B-4bit 拿到
**244 tok/s**（`agent-infer/AGENTS.md` 记录的同台机器参考），qw36 当前最好
122 tok/s（fp16 MPS），Q4_K affine32 85 tok/s。差 2-3×。MLX 用的是 **affine
quant**（而非 GGUF K-quant），跟我们 `Q4K_AFFINE32` repack 后的内存 layout
基本一样。**那为什么我们没追平？**

**结论先行**：MLX 跟 llama.cpp 用了**几乎相同的 GEMV factoring 技巧**（仿射
分解 + bit-pack pre-scale + multi-row-per-thread），但 MLX 多出 3 块 qw36
没有的硬实力：
1. **quad-subgroup GEMV** (4-lane subgroup) for K∈{64,128} —— 比 simdgroup
   更省 launch；
2. **三档 GEMV (qmv_quad / qmv_fast / qmv)** 自动选 + **qmm_splitk** 小 M
   GEMM；
3. **NAX tensor cores** (M3/M4 metal 4.1 MMA) —— qw36 / llama.cpp 都没接。

外加：
- **sdpa_vector**（Q=1 decode 路径，1024 threads/TG）+ **sdpa_vector_2pass**
  （长 context 的 split-K）+ **sdpa_full_nax**（Q>=64 prefill）三档分流；
- **gather_qmm** (MoE) 直接复用 qmm，加 `lhs_indices/rhs_indices` 走索引。

参考 path：

```
mlx/backend/metal/
├── quantized.cpp                  (1751 行)  dispatch / 分档逻辑
├── scaled_dot_product_attention.cpp           sdpa_vector / sdpa_full / nax
├── kernels/
│   ├── quantized.h                (2603 行)  qmv_quad / qmv_fast / qmv / qmm / gather_qmm 模板
│   ├── quantized.metal            (158 行)   host_name instantiations
│   ├── quantized_nax.h            STEEL NAX 路径
│   ├── gemv.metal / gemv.h        non-quant GEMV (fp16/bf16/fp32)
│   ├── steel/                     STEEL = CUTLASS 风格 tile loader
│   └── sdpa_vector.h              Q=1 decode 内核
└── matmul.cpp                                 dense matmul dispatch
```

---

## 1. MLX affine quant layout —— 跟我们 AFFINE32 repack 一模一样

`kernels/quantized.h:17-26`：

```cpp
template <int bits, int wsize = 8>
inline constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}
// bits=4 → pack_factor=2 (per byte), 或 32/4=8 per uint32
// bits=8 → pack_factor=1
```

**4-bit affine layout**：

```
struct AffineBlock<bits=4, group_size=32 or 64> {
    half scale;                  // 一个 group 一个 scale
    half bias;                   // 一个 group 一个 bias (注意是 +bias 而非 -min)
    uint8 qs[group_size / 2];    // 4-bit packed
};
```

- `group_size` 是模板参数，默认 32 或 64（dispatch 时选）。
- Dequant: `(q * scale + bias)`（注意 **+ bias**，是 asymmetric signed
  quant，跟 GGUF Q4_K 的 `-min` 等价但符号反）。
- Quant kernel：`quantized.h:2439-2496`，标准 max/min scan + uniform bin。

这跟 qw36 的 `Q4K_AFFINE32` 完全是同一形状，包括 group_size=32 都对得上。
**Layout 不是 MLX 占便宜的地方。**

---

## 2. qdot —— 仿射 factoring 的关键

`kernels/quantized.h:192-290`，所有 GEMV kernel 都调它：

```cpp
template <typename U, int values_per_thread, int bits>
inline U qdot(const device uint8_t* w, const thread U* x_thread,
              U scale, U bias, U sum) {
  U accum = 0;
  if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4*i+0] * (ws[i] & 0x000f) +
           x_thread[4*i+1] * (ws[i] & 0x00f0) +
           x_thread[4*i+2] * (ws[i] & 0x0f00) +
           x_thread[4*i+3] * (ws[i] & 0xf000));
    }
  }
  // ... bits=2/3/5/6/8 等价分支
  return scale * accum + sum * bias;
  //     ^^^^^^^^^^^^^^^^^^^^^^^^^^
  //     仿射 factoring：dot(q, y_pre_scaled) * scale + sum(y) * bias
}
```

**与 llama.cpp Q4_K 的同一招**：

| 操作 | MLX qdot | llama.cpp Q4_K |
|------|----------|----------------|
| 仿射 factoring | `scale*accum + sum*bias` | `d_all*(acc * sc) - dmin*(sumy * sm)` |
| 单 bit-pack 读 4 值 | `(ws[i] & 0x000f, 0x00f0, 0x0f00, 0xf000)` | `(q1[i] & 0x000F, 0x0F00, 0x00F0, 0xF000)` |
| `x_thread` pre-scale 抵消位移 | `load_vector`：`x_thread[i+1] = x[i+1] / 16` | 内嵌 `1.f/256.f` 抵消 |
| sum(y) 一次性算 | `load_vector` 返回 `sum` | `sumy[0..3]` 在外层求和 |

**核心 insight 完全一样**：affine 量化的 `bias`/`-min` 项只跟 `sum(y)` 有关，
跟 `q[]` 无关 → 每 group 只算 **1 次 mul**，不是 `group_size` 次。

qw36 的 `Q4K_AFFINE32 qmv_fast` 当时**没用这条 factoring**，逐元素做的
`(q * scale + bias) * y` —— 这是 85 vs 244 差距的一部分。直接补上这条
factoring 是 L1 lever。

---

## 3. 三档 GEMV —— quad / fast / 普通

`quantized.cpp:1365-1385` 的 `dispatch_qmv` 是入口：

```cpp
void dispatch_qmv(...) {
  if ((K == 128 || K == 64) && is_power_of_2(bits)) {
    qmv_quad(...);   // K 小时走 quad 4-lane 路径
    return;
  }
  qmv(...);          // 其他走 simdgroup 32-lane 路径
}
```

但 `qmv_fast` 还有一档（lines 1497-1547），在 `qmv` 内部根据 power-of-2 bits
自动 fallback 到 fast 还是 generic。

### 3.1 qmv_quad —— 4-lane subgroup

`kernels/quantized.h:692-747`：

```cpp
template <typename T, int group_size, int bits, int D>
METAL_FUNC void qmv_quad_impl(...,
    uint quad_gid [[quadgroup_index_in_threadgroup]],
    uint quad_lid [[thread_index_in_quadgroup]]) {

  constexpr int quads_per_simd = SIMD_SIZE / QUAD_SIZE;  // 32/4 = 8
  constexpr int values_per_thread = D / QUAD_SIZE;       // D=128 → 32 vals/thread
  constexpr int results_per_quadgroup = 8;                // 一个 quad 算 8 个 output
  ...
  U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

  for (int row = 0; row < results_per_quadgroup; row++) {
    auto wl = (const device uint8_t*)(w + row * in_vec_size_w * quads_per_simd);
    ...
    result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
  }

  for (int row = 0; row < results_per_quadgroup; row++) {
    result[row] = quad_sum(result[row]);  // 4-lane sum，比 simd_sum 便宜
    if (quad_lid == 0) y[row * quads_per_simd] = result[row];
  }
}
```

**Key idea**：
- **4 lanes = 1 quad subgroup**（Metal 的最小 SIMD 单位，硬件保证 lockstep）。
- **8 quads / SIMD group**（32/4），8 quads × 8 results = 64 output rows per
  TG（如果用一个 simdgroup）。
- 用 `quad_sum`（仅 4-lane reduction）代替 `simd_sum`（32-lane），**少 28 lane
  的 reduction 开销**。
- 触发条件：K=64 或 128，bits 是 2 的幂。Qwen3.5 0.8B 的 head_dim=64
  正好命中（attn QKV 投影、down_proj 都可能跑这条）。

这一档 qw36 完全没有 —— 我们 `qmv_fast` 是 32-lane simdgroup 一档到底。

### 3.2 qmv_fast —— 32-lane simdgroup, packs_per_thread=2

`kernels/quantized.h:749-814`：

```cpp
constexpr int packs_per_thread = bits == 2 ? 1 : 2;       // 每 thread 处理 2 个 pack
constexpr int num_simdgroups = 2;                          // 2 SG / TG
constexpr int results_per_simdgroup = 4;                   // 每 SG 算 4 row
constexpr int values_per_thread = pack_factor * packs_per_thread;
constexpr int block_size = values_per_thread * SIMD_SIZE; // 一次内层 step 处理 block_size 个 K
```

- 一 TG = 2 SG × 32 lanes = 64 thread。
- 每 TG 一次算 8 个 output row（2 × 4 = 8）。
- 内层 loop 每 step 处理 `block_size = 8 * 32 = 256` 个 K 维元素（4-bit）。
- 触发条件：K ≥ 256 且 K 不是 64/128。lm_head（K=896 hidden）和 MLP
  (gate/up/down) 都走这条。

### 3.3 qmv —— fallback 单 pack/thread

`kernels/quantized.h:816-...`，`packs_per_thread = 1`，触发：K 太小或 bits 非
2 的幂（如 5-bit、6-bit）。Qwen3.5 q4 不太用到。

### 3.4 dispatch_qmv 三档总览

| 触发条件 | 选哪档 | TG 几何 | 每 TG row 数 |
|----------|--------|---------|---------------|
| K∈{64,128} ∧ bits 是 2 幂 | qmv_quad | (32, 1, 1) | 64 (8 quads × 8 res) |
| K≥256 ∧ bits 是 2 幂 | qmv_fast | (32, 2, 1) | 8 (2 SG × 4 res) |
| 其他 | qmv | (32, 2, 1) | 8 |

---

## 4. qmm + qmm_splitk —— 大 M GEMM

`quantized.cpp:1412-1418`：

```cpp
if (M >= vector_limit) {
  int B = out.size() / M / N;
  if (transpose_ && B == 1) {
    qmm_splitk(...);   // 小 M、transpose、non-batched 走 split-K
    return;
  }
  qmm(...);
}
```

`vector_limit = get_qmv_batch_limit(K, N, d)` —— 按 K/N/硬件选 4-8 区间。也就
是说 **M < 4 走 GEMV，M ≥ 4 走 GEMM 或 splitk-GEMM**。

### 4.1 qmm —— 标准 tiled

`kernels/quantized.h:1722-1850` (`affine_qmm_t` transposed)。BM=BN=32, BK=32
的 tile，threadgroup 内 wm × wn 个 simdgroup 协作。Dequant tile 加载到
threadgroup memory，然后 `simdgroup_matrix<T,8,8>` MMA 算 8×8 fp16 tile。

### 4.2 qmm_splitk —— 小 M 时的关键武器

Comment line 788 提到 "target ~512 threadgroups"。**当 M 很小（比如 prefill
chunk 8 token）但 N 很大（vocab 152K）时**，标准 tiled GEMM 的 grid =
`(N/BN, M/BM)` 只有几个 TG，GPU 占用率低。Split-K 把 K 维切片到多个 TG，每个
TG 算 partial sum，再 reduce。这给小 M 工况拉满 occupancy。

qw36 没有 split-K。我们 prefill 走 MPS 时被 Apple 的 GEMM 调度决定；自己写
quant GEMM 也没考虑这一档。

### 4.3 NAX —— M3/M4 真 MMA tensor core

`kernels/quantized_nax.h` + `kernels/steel/gemm/nax.h`。触发：
`is_nax_available() && transpose && (K % 64 == 0)` (`quantized.cpp:695`)。

- **NAX = Apple Neural Accelerator extensions for Metal 4.1**，M3 及之后有
  专门的 tensor core，跟 Nvidia 的 wmma 类似。
- `NAXTile<T, TM=16, TN=16>` 是 16×16 fp16 fragment。
- async block loader (`QuantizedBlockLoader`) 把 K×BN block 异步加载到 LDS。
- 触发后 GEMM 用真 tensor core 而不是 simdgroup MMA —— 理论吞吐 **2-4×**
  simdgroup_matrix。

llama.cpp 用的是 MPP `tensor_ops::matmul2d`（也走 MMA 但走 Apple 的高层 API
而不是 NAX raw 指令）。**MLX 在 M3/M4 上比 llama.cpp 这一档更快**，是 244
tok/s 后面的硬件加成。

qw36 完全没有 NAX / steel。我们 prefill 走 MPS（也基于 MMA 但中间隔一层
framework），decode 走自己写的 GEMV。

---

## 5. SDPA —— Q=1 decode 走 sdpa_vector

`scaled_dot_product_attention.cpp` 有三个入口：

| 路径 | 触发 | 文件:行 |
|------|------|---------|
| `sdpa_vector` | Q=1, N (KV len) < 阈值 | `:329-415` |
| `sdpa_vector_2pass` | Q=1, N 大（长 context） | `:418-617` |
| `sdpa_full_self_attention_nax` | Q ≥ 64 prefill, NAX 可用 | `:18-... ` |
| `sdpa_full_self_attention`（无 NAX 时） | Q ≥ 64 prefill | `:257+` |

### 5.1 sdpa_vector (Q=1 decode)

```cpp
MTL::Size group_dims(1024, 1, 1);                          // ← 1024 threads/TG！
MTL::Size grid_dims(q.shape(0) * q.shape(1), q.shape(2), 1);
//                  ^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^
//                  batch * head             q_seq (decode = 1)
```

**1024 thread per TG** 是 Apple Metal 单 TG 上限。这意味着 32 个 simdgroup
协作算一个 (head, query)，整个 KV cache 沿 N 维并行 reduce。

Kernel name 由 `q_dtype × Dhead × Dvalue` 编译期固化，但 **KV 必须是
fp32/fp16/bf16** —— 没有 `_q4_0` / `_q8_0` 变体。

### 5.2 sdpa_vector_2pass (Q=1 长 context)

当 KV 序列长（N 大到一个 TG 1024 thread 也吃不下），切成 N/CHUNK 个
partial pass，每 chunk 一个 TG，最后再 reduce。这是 split-N decode。
qw36 的长 context (n=2048) 路径相当于这一档，但我们没拆 KV 维度。

### 5.3 vs llama.cpp 的对比

| 维度 | MLX | llama.cpp |
|------|-----|-----------|
| Q=1 decode 专用 kernel | `sdpa_vector` 1024 thr/TG | `kernel_flash_attn_ext_vec` Q=1 模板 |
| 长 context split | `sdpa_vector_2pass` | TG 内多 chunk loop |
| Q≥8 prefill | `sdpa_full_nax` NAX MMA | `kernel_flash_attn_ext` Q=8 simdgroup MMA |
| **量化 KV cache** | ❌ 不支持 | ✅ Q4_0/Q5_0/Q8_0 各一份 kernel |

**这是 llama.cpp 反超 MLX 的唯一维度** —— 长 context decode 内存压力下，
量化 KV cache 是真正的省 bandwidth。MLX 在这一档比 llama.cpp 慢。qw36 当前
有 Q8 KV，flash kernel 也接了，**这条优势我们可以保住**。

---

## 6. gather_qmm —— MoE 入口

`kernels/quantized.h:2092-2240`。`affine_gather_qmm_t/n` 几乎是 `qmm_t/n`
的薄封装：

```cpp
[[kernel]] void affine_gather_qmm_t(
    const device uint32_t* w, scales, biases, x,
    const device uint32_t* lhs_indices,   // 每个 output slot 走哪个 token
    const device uint32_t* rhs_indices,   // 每个 output slot 用哪个 expert weight
    device T* y, ...) {
  adjust_matrix_offsets<T>(  // 看 lhs/rhs_indices 调整指针
      x, w, scales, biases, lhs_indices, rhs_indices, y, ...);
  qmm_t_impl<...>(w, scales, biases, x, y, ...);   // ← 直接复用 qmm
}
```

**MoE = top-k 在 Python (mlx-lm) 侧算 indices → 调一次 gather_qmm → 内部走
qmm/qmv 的所有优化**。这是 qw36 `docs/moe_design.md` SwitchGLU 的精确参考。

`GatherQMM::eval_gpu` (`quantized.cpp:1456-...`) 同样分档：
- M ≥ vector_limit → gather_qmm (qmm-based)
- M < vector_limit → 走 gather_qmv (qmv-based)

**SwitchGLU 不在 mlx 仓库**，在 `mlx-lm`（model 层 Python）。仓库这边只提供
原语，组装方式在 model file 里。

---

## 7. STEEL —— MLX 的"CUTLASS"

`mlx/backend/metal/kernels/steel/` 整套是 MLX 的 tile loader / async copy /
MMA 抽象，相当于他们自家的 CUTLASS。包含：

- `gemm/nax.h` —— NAX MMA tile 操作（M3/M4 only）
- `gemm/mma.h` —— 通用 simdgroup MMA（fp16/bf16）
- `attn/loader.h` —— FlashAttention KV tile loader
- `attn/transforms.h` —— softmax/mask transform
- `defines.h` —— 编译期常量

这套是 MLX 跟 qw36 / llama.cpp 最大的工程差距。**llama.cpp 是手写 metal，MLX
是抽象成 reusable tile op 后再写 kernel**。我们短期没必要复制 STEEL，但需要
认识到这是为啥 MLX 加 NAX、加新 dtype、加新 shape 比 llama.cpp 快。

---

## 8. 综合三方对比 —— qw36 vs llama.cpp vs MLX

| 项目 | qw36 | llama.cpp | MLX |
|------|------|-----------|-----|
| Quant 格式 | GGUF K-quant + AFFINE32 repack | GGUF K-quant 原生 | MLX affine (group32/64) |
| Q4 GEMV 仿射 factoring | ❌ | ✅ | ✅ |
| bit-pack pre-scale 抵消 shift | 部分 | ✅ | ✅ |
| ROWS_PER_SIMD multi-row GEMV | ❌（1 行） | ✅ (NR0 per dtype) | ✅ (4/8 per SG) |
| Q=64/128 quad 4-lane GEMV | ❌ | ❌ | ✅ qmv_quad |
| 小 batch GEMV (M∈[2,8]) | ❌（走 MPS） | ✅ mul_mv_ext | ✅ qmm + splitk |
| 真 MMA (M>vector_limit) | MPS GEMM | MPP matmul2d | **NAX tensor cores** (M3/M4) |
| Q=1 decode SDPA | flash_attn_f16kv | flash_attn_ext_vec | sdpa_vector 1024 thr |
| 长 context split | flash_attn 内 chunk | flash_attn_ext_pad | sdpa_vector_2pass |
| 量化 KV cache | Q8_0 ✅ | Q4_0/Q5_0/Q8_0 ✅✅ | ❌ |
| MoE 入口 | naive 1-row-per-thread (79% GPU) | mul_mv_id 复用 mul_mv | gather_qmm 复用 qmm |
| 并发命令缓冲 | 单 encoder | n_cb × dispatch_apply | 多 stream |

---

## 9. 对 qw36 的 lessons

**已经在 llama.cpp 调研里列过的（[`llamacpp_metal_research.md`](llamacpp_metal_research.md) §8）**：

- L1 仿射 factoring（**两份调研都指向同一条**）
- L2 per-dtype ROWS_PER_SIMD
- L3 mul_mv_ext / qmm_splitk 小 batch kernel
- L4 多 KV dtype 编译期 specialize

**MLX 独有的、qw36 可以考虑的**：

### L7 — **qmv_quad 4-lane 路径**（head_dim=64 时有用）

Qwen3.5 / Qwen3.6 vanilla attention 的 head_dim 一般 64 或 128。**Q/K/V
投影里的 K 维就是 head_dim** —— 命中 MLX 的 qmv_quad 路径。

qw36 现在统一走 32-lane qmv_fast。在 head_dim=64 时，**quad 路径 8 quads ×
8 results = 64 output rows per TG，simd 路径 2 SG × 4 results = 8 rows
per TG**。**8× 的 row-per-TG 密度差距**。

**Action**：head_dim ≤ 128 的 matmul 加 quad-subgroup 变体。先 bench 验证
QKV 投影是不是瓶颈，再决定优先级。

### L8 — **qmm_splitk 小 M GEMM**（prefill 8-token chunk）

qw36 prefill 走 MPS GEMM。M=8、N=152K 时 MPS 大概率没切 K 维 → grid =
(N/BN, M/BM) = (1188, 1) 一行 TG，GPU 占用 < 10%。

**Action**：自己写一个 quant GEMM split-K，或者让 prefill chunk 大小调整避开
这一档（chunk 32 而不是 chunk 8，让 M/BM ≥ 几个 SG）。后者更便宜。

### L9 — **STEEL 风格的 tile loader**（长期）

只有要在 qw36 上加 NAX/MMA quant GEMM 时才需要。**短期不做**。

### L10 — **sdpa_vector 1024 thread/TG decode kernel**

qw36 的 flash kernel 现在大概是 256 threads/TG（2 SG）。MLX 用 1024（32 SG）。
但要注意：1024 threads 需要 KV 沿 N 维并行 reduce，**threadgroup memory 占用
极大**，对 batch=1 单 head 才划算。

**Action**：等长 context 性能再压不动的时候再试。当前 105 tok/s @ n=2048 已
经接近合理上限，先看清还有多少剩余收益。

---

## 10. 三方共识 vs 分歧

**三方共识（所有人都做）**：
1. 仿射 factoring（`scale*accum + sum*bias`）
2. bit-pack 4-element-per-uint16 + pre-scale 活化向量
3. 一个 TG 算多个 output row (ROWS_PER_SIMD)
4. simd_sum (或 quad_sum) 跨 lane reduction
5. Q=1 vs Q>1 走不同 SDPA kernel

**llama.cpp 独有**：
- K-quant 原生 layout（不 repack）
- 量化 KV cache (Q4_0/Q5_0/Q8_0)
- per-dtype 调优 NR0/NSG

**MLX 独有**：
- qmv_quad 4-lane (K=64/128)
- qmm_splitk 小 M
- NAX tensor core (M3/M4)
- STEEL tile abstraction

**qw36 独有（且应该保住）**：
- Q4_K AFFINE32 repack（虽然 llama.cpp 不需要，但我们已经付了成本）
- KV prefix cache tier composition (ram_lru + disk)
- 单文件可移植 / 三 backend 共用 vtable

---

## 11. 参考资料

- MLX Metal 源码（本文档引用的所有行号都是这份）：
  https://github.com/ml-explore/mlx/tree/main/mlx/backend/metal
- mlx-lm SwitchGLU（model 层）：
  https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/switch_layers.py
- MLX 量化算法 blog post：
  https://ml-explore.github.io/mlx/build/html/quantization.html
- 调研日期：2026-05-21；MLX commit：master HEAD（sparse clone）

**相关 qw36 文档**：

- [`docs/llamacpp_metal_research.md`](llamacpp_metal_research.md) —— 姊妹篇，
  llama.cpp Metal 调研
- [`docs/q4k_kernel_design_v2.md`](q4k_kernel_design_v2.md) —— AFFINE32 决策
  来源
- [`docs/q4k_qmv_quad_failed.md`](q4k_qmv_quad_failed.md) —— port MLX
  qmv_quad 失败 root cause（**重读**：我们当时试图把 quad port 到 GGUF Q4_K
  layout 失败；但 AFFINE32 layout 上 quad 是可行的，只是我们当时还没 repack。
  这条 lesson 现在可以重新评估）
- [`docs/moe_design.md`](moe_design.md) —— SwitchGLU 设计（gather_qmm 在 MLX
  叫这个名字）
- `agent-infer/AGENTS.md` —— MLX 244 tok/s reference
