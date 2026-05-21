# qw36 — Agent Contract

Assisting **cklxx** (q1293822641@gmail.com). **Project-specific** rules only;
generic C / Metal / CUDA / git knowledge is intentionally absent.

---

## §0 第一原则 — SOLID（求真务实，追求极致）

**所有事必须 SOLID。不够 SOLID 就不断深入，不断突破。**

- **推断 ≠ SOLID**：source grep、callgraph 推断、design doc 都是 *hypothesis*，
  不是 evidence。Evidence = 跑过的 bench 数字 / `QW36_METAL_PERF=1` per-kernel
  trace / `precision_cpu_vs_metal.sh` 对照 / 实测 wallclock。没有 evidence
  不下结论。Hypothesis 必须显式标。
- **混淆变量必须隔离**：一个实验同时改 N 个变量（kernel fusion + dtype + env
  flag + persistent encoder）→ 任何结果都不能归因。每次只改一个变量，或显式
  跑对照实验。
- **Wall-clock framing 是 ground truth**：per-kernel `gpu_ms` 累加 ≠ wallclock。
  Persistent encoder 下 kernel 是并发的，profile 累计会高估。验证收益必须用
  `tok/s decode (n>=64)` wallclock，不用 PERF gpu_ms 占比自欺。
- **可回退 + 可验证**：fp16 state、quant 路径都已经掉过坑（logits drift / 异常
  字符）。任何方向先想"怎么回退、怎么 verify"，再开始改。
- **80% SOLID 不够**：发现 gap 必须深入到 95%+，或显式声明 "deferred, accepting
  uncertainty"。禁止 silent skip。

实证经验：
- **2026-05-19 fp16 state drift**：把 `q_dev` 切 fp16，24 层累积导致 logits
  发散（`Hello` → `uela` 而非 `,`）。Bisect 才能定位。Lesson：任何 dtype 收
  缩必须先做单层 vs 多层 diff（`docs/fp16_state_root_cause.md`）。
- **2026-05-20 Q4_K qmv_quad fail**：直接把 MLX qmv_quad 端口到 GGUF Q4_K
  layout → 29 tok/s 比 baseline 58 还慢。Root cause: Q4_K 每 32 元换 scale，
  MLX affine 全 group 同 scale；4-lane quad K/4 decode 太长。Lesson：参考实
  现的前提（数据 layout）必须先验证（`docs/q4k_qmv_quad_failed.md`）。
- **2026-05-21 Q5_K promote-to-fp16 regression**：在 affine32 mode 把 Q5_K 切
  成 fp16 lazy weights，从 85 → 50 tok/s。Per-PERF 看 `qw36_rmsnorm_f32` 调用
  量爆 30 倍，怀疑 fusion fast-path dispatch 失效。**未深入即回退** — 标记
  deferred，下次需要先排查 dispatch 路径再动。

---

## Project shape

`qw36` 是 **纯 C 的 Qwen 3.5 / 3.6 推理引擎**，单文件可移植，三 GPU backend 通
过 frozen vtable 接入。无 Python / Rust / C++ 在 hot path 上。

```
qw36/
├── common/                 ← engine + 抽象层（CPU + GPU 共用）
│   ├── qw36.{h,c}          ← 引擎 lifecycle + per-token forward orchestrator
│   ├── qw36_internal.h     ← 私有 API（lazy_w / forward_ctx / dispatch_dev）
│   ├── qw36_gpu.h          ← backend vtable 契约（不要随便改 ABI）
│   ├── qw36_gguf.{h,c}     ← GGUF v3 loader (mmap)
│   ├── qw36_tokenizer.{h,c}← BBPE + Qwen3 chat-template specials
│   ├── qw36_dequant.c      ← 所有 dq_q*_K / repack（Q4_K → affine32 在这里）
│   ├── qw36_ops.c          ← rmsnorm / rope / residual / silu (CPU 原语)
│   ├── qw36_attn_vanilla.c ← 普通 GQA 注意力 + Q-gate (Qwen3.5/3.6)
│   ├── qw36_attn_deltanet.c← Gated DeltaNet decode (hybrid 模型)
│   ├── qw36_mlp.c          ← SwiGLU dense MLP
│   ├── qw36_moe.c          ← Top-K MoE + shared expert
│   └── qw36_cli.c          ← CLI entry / sampler
├── cpu/Makefile            ← CPU-only build → qw36_cpu
├── metal/                  ← Apple Metal backend（主力）
│   ├── qw36_metal.m        ← MTL pipelines + dispatch + MPS GEMV
│   └── qw36_metal.metal    ← MSL kernels（fused decode / rmsnorm / quant）
├── cuda/                   ← CUDA backend（off-host build only）
├── amd/                    ← HIP backend（off-host build only）
├── tests/                  ← bench / precision 对比脚本
├── docs/                   ← research briefs + design docs（不是产品文档）
├── models/                 ← HF snapshot cache（lazy fetch）
└── Makefile                ← 顶层调度 cpu/ metal/
```

**Canonical bench model — Qwen3.5-0.8B-Q4_K_M**：
`/Users/bytedance/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf`

35B-A3B MoE 模型 (`~/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`) 仅用于功能
冒烟，不用于 perf 主线 —— 用户明示 "测试还是用 0.8B 的就行"（2026-05-20）。

**Perf target / ceiling**（M-class GPU, Apple Silicon 统一内存）：
- 1.6 GB fp16 weights / 200 GB/s ≈ **8 ms/token = 125 tok/s 理论**
- 实测最好 **122 tok/s** (fp16 path, short generation)，**109 sustained @ n=256**
- MLX reference: 244 tok/s（他们走 affine 4-bit + 自研 qmv_quad，hardware
  ceiling 同一台机器约 500 tok/s 4-bit / 250 tok/s fp16）
- 当前 Q4_K affine32 path：85 tok/s opt-in (`QW36_METAL_Q4K_AFFINE32=1`)

详见 [`FINAL_STATUS.md`](FINAL_STATUS.md) 完整 ladder + 失败实验清单。

---

## Rules

### Execution phases

| Phase | Exit condition |
|-------|----------------|
| **Explore** | 能说出每个要改的文件路径 + 当前代码做什么 |
| **Plan**（>3 文件或不可逆 → 先 confirm） | 用户接受的简短方案 |
| **Implement** | `make` 通过，warning 不增加 |
| **Verify** | bench `tests/quant_kernel_bench.sh` 或自定义 wallclock 3× 重跑；correctness 用 `Hello! How can I help you today?` 句尾收敛作 sanity（任何 dtype 改动都跑）|
| **Reflect**（失败 / 反直觉 / 反复出错 → `docs/` 留 design 或 failed entry） | 经验 commit |

简单 mechanical change：Implement + Verify 即可。

### Editing

- **保留为主。** 不要删除范围外内容；废弃的 opt-in env flag 留着（已有 G/Q/J
  failed experiment 是历史 evidence）。
- **不要半成品。** Refactor 一刀切完或回滚；不要并存新旧 dispatch path。
- **AGENTS.md 是 canonical。** `CLAUDE.md` 是 symlink。两份要保持等同。
- **Plan-first** 当改动 > 3 文件、动 vtable / ABI / state layout / kernel
  dispatch 流程 时，先说方案再动手。

### Backend isolation

- `common/qw36_gpu.h` 是 backend vtable，**ABI 是冻结的**。要新增 slot 三个
  backend 都得加（即使是 stub），否则 CUDA/AMD off-host build 会断。
- 跨 backend 模块（`common/`）**禁止 `#ifdef __APPLE__` / `#ifdef __CUDACC__`**
  分支。Backend 特定的逻辑走 vtable dispatch（`qw36__*_dispatch_dev`）。
- Metal 上：`MPSMatrixVectorMultiplication` 是 fp16 / fp32 GEMV 的 baseline。
  自写 kernel **必须先 bench 过 MPS** 才算赢，不许只看 spec 估算。

### Delegation (Codex + Claude 双人协作)

历史协作模式（保持）：
- **Claude** 负责：方向 + 集成 + plan + final commit + 文档。
- **Codex** 负责：执行（kernel 重写 / refactor / 调研深 dive），通过 tmux
  session 或 `codex review` shell 调用。
- 并行原则：Codex 写 kernel 的同时 Claude 写 fusion / orchestration / docs。
- **Codex hangs warning**：`codex:codex-rescue` 和 `mcp__openmax__execute_with_codex`
  有挂死历史 — 优先用 tmux + brief markdown 派活。
- 失败两次 → Claude 自己写小 diff 或换一个 brief 重派 codex。

### Benchmarks

- **每个 perf-affecting commit 必须带数据。** 仅 docs / comment / `AGENTS.md`
  / `memory/` 例外（commit body 说明 exempt）。
- 标准 bench：`tests/quant_kernel_bench.sh` —— side-by-side fp16 / QUANT_GPU /
  QUANT_GPU+AFFINE32，每条 3 次。
- 自定义 bench 必须报告：commit、env、`-n` 长度、3 次重跑结果（拒绝单点 noise）。
- **Wallclock framing 优先**：profile per-kernel `gpu_ms` 用来定位瓶颈，但
  validation 必须用 end-to-end wallclock tok/s。Persistent encoder 下 GPU 并
  发会让 `gpu_ms` 累计 > wallclock。
- 失败实验记 `docs/` 一份 brief（什么数据布局 / 为什么慢 / root cause hypothesis），
  不要 silent delete。
- bench 结果同步进 `FINAL_STATUS.md` 的 ladder 表。

### Git

- Commitizen `<type>(<scope>): <subject>`。Scope：`metal`、`cuda`、`amd`、
  `common`、`tokenizer`、`gguf`、`docs`、`tests`。
- 直接 commit 到 `main`，不开 feature branch（小项目 single-author）。
- 小切片立即 commit，verification 失败 → 后续 commit 修复，不要 amend 历史。
- **不要 push 到 main 除非用户明确说**。Auto mode 默认会 block，符合预期。

### Code conventions

- C99/C11，`-Wall -Wextra` 不能新增 warning。
- 私有 API 用 `qw36__` 双下划线前缀（package-private 约定）。
- 不引入新依赖（mmap / pthread / Metal framework 已经是底线）。
- Lazy weight 走 `qw36_lazy_w` 描述符，**materialize 是惰性 + cached** —
  `lazy_materialize_f16` / `_f32` 不会重复执行。
- State buffer (`x_dev` / `x_rms_dev` / `q_dev` / kv_cache_dev) 默认 fp32；
  fp16 化是 opt-in（`QW36_METAL_FP16_*`），有 logits drift 历史。
- Forward fusion 用 `fc->post_attn_rmsnorm_done` / `fc->input_rmsnorm_done`
  这种 flag 跨模块传递；下一层 dispatch 看 flag 决定是否 skip rmsnorm。

### GPU kernel work

动 `metal/qw36_metal.metal` 或 `cuda/*.cu` 之前：
1. 读 [`docs/q4k_kernel_design_v2.md`](docs/q4k_kernel_design_v2.md) —— Q4_K
   kernel 失败 + 成功 attempt 全记录。
2. 跑 `QW36_METAL_PERF=1` 拿到 baseline 每 kernel us。
3. MPS GEMV 是 fp16 / fp32 的 hard floor —— 自写 kernel 必须 bench 过 MPS。
4. M=1 GEMV **不会**从 MMA / 多 row tile 受益（lm_head MMA 实验比 MPS 慢
   2× — 因为 8×8 tile 浪费 7/8）。
5. Persistent compute encoder 已经默认开（commit 112e85f）—— 新 kernel 走
   `metal_compute_encoder_for_op` API 而不是直接 makeComputeCommandEncoder。

---

## Build & run

```bash
make                  # 全 backend (cpu + metal)
make -C cpu           # CPU only → ./qw36_cpu
make -C metal         # Metal only → ./qw36_metal

./qw36_metal -m <gguf> -p "Hello" -n 64       # 单次推理
tests/quant_kernel_bench.sh                   # canonical bench

# 关键 env flag
QW36_METAL_FP16_WEIGHTS=1     # fp16 weights via MPS（**最快 path: 115-122 tok/s**）
QW36_METAL_FP16_KV=1          # fp16 KV cache（已默认）
QW36_METAL_QUANT_GPU=1        # 保留 Q4_K/Q5_K/Q6_K/Q8_0 packed weights on GPU
QW36_METAL_Q4K_AFFINE32=1     # 把 Q4_K repack 成 affine32 走 qmv_fast（+45% over native Q4_K, 85 tok/s）
QW36_METAL_PERF=1             # per-kernel gpu_ms profile（dev only, 关掉 persistent encoder）
QW36_METAL_TIMING=1           # per-forward wallclock
```

详细 ladder + 失败实验：[`FINAL_STATUS.md`](FINAL_STATUS.md)。

---

## Core docs (on-demand)

- [`FINAL_STATUS.md`](FINAL_STATUS.md) — perf ladder + ceiling 分析 + 失败实验
- [`docs/q4k_kernel_design_v2.md`](docs/q4k_kernel_design_v2.md) — Q4_K kernel
  深度调研 + affine32 决策 + qmv_fast 设计
- [`docs/q4k_qmv_quad_failed.md`](docs/q4k_qmv_quad_failed.md) — Q4_K qmv_quad
  直接 port 失败 root cause
- [`docs/fp16_state_root_cause.md`](docs/fp16_state_root_cause.md) — fp16 q_dev
  导致 logits drift 的 bisect
- [`DIVISION_OF_WORK.md`](DIVISION_OF_WORK.md) — Codex / Claude 历史分工
- [`CODEX_TASKS.md`](CODEX_TASKS.md) — codex 派活 brief 模板
- `../agent-infer/AGENTS.md` — perf 参考实现 (MLX, 244 tok/s 同台机器)
