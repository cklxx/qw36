# Backend parity audit

Where each backend stands against the Metal reference. Updated
2026-05-21 (commit `137eaa9`).

Metal is the perf-leader on this host; CUDA and AMD lag because they
were compile-checked but not runtime-iterated. This doc is the
canonical "what's still NULL" so Roadmap theme 1.2 ports can attack
the right slots first.

## Per-vtable-slot status

Slot list from `common/qw36_gpu.h`. ✓ = implemented and verified;
🟡 = implemented but unverified (no GPU host); ❌ = stub (NULL); — =
intentionally optional.

| slot                          | Metal | CUDA | AMD  | notes |
|-------------------------------|:-----:|:----:|:----:|-------|
| `init`                        | ✓     | 🟡    | 🟡    | device acquisition |
| `destroy`                     | ✓     | 🟡    | 🟡    | cleanup |
| `begin_batch` / `end_batch`   | ✓     | —    | —    | Metal persistent encoder. CUDA/AMD don't currently need it. |
| `upload` / `download`         | ✓     | 🟡    | 🟡    | host↔device |
| `copy_from_host`              | ✓     | 🟡    | 🟡    | |
| `alloc` / `free`              | ✓     | 🟡    | 🟡    | |
| `rmsnorm`                     | ✓     | 🟡    | 🟡    | |
| `matmul`                      | ✓     | 🟡    | 🟡    | fp32/fp16/Q4_K/Q5_K/Q6_K/Q8_0; CUDA/AMD likely cover only fp32/fp16 |
| `attention`                   | ✓     | 🟡    | 🟡    | Q-gate + RoPE + KV append; CUDA/AMD likely don't have Q-gate yet |
| `swiglu_mlp`                  | ✓     | 🟡    | 🟡    | |
| `residual_add`                | ✓     | 🟡    | 🟡    | |
| `residual_rmsnorm`            | ✓     | ❌    | ❌    | fused residual + next-rmsnorm. CUDA/AMD never got this. |
| `embedding_lookup`            | ✓     | 🟡    | 🟡    | |
| `dn_conv1d_silu`              | ✓     | ❌    | ❌    | DN block step 1 |
| `dn_gated_delta`              | ✓     | ❌    | ❌    | DN block step 2 |
| `dn_gated_delta_conv1d`       | ✓     | ❌    | ❌    | fused (1+2) — Metal only optimization |
| `dn_gated_rmsnorm`            | ✓     | ❌    | ❌    | DN block step 3 |
| `dn_gated_rmsnorm_matmul`     | ✓     | ❌    | ❌    | fused (3+4) — Metal only optimization |
| `moe_forward`                 | ✓     | ❌    | ❌    | route + top-K + experts + shared expert |

### Counts

| backend | filled | stubbed | optional |
|---------|-------:|--------:|---------:|
| Metal   |     20 |       0 |        — |
| CUDA    |     12 |       8 |        2 |
| AMD     |     12 |       8 |        2 |

CUDA and AMD are missing the same 8 slots (DN family + MoE + fused
residual+rmsnorm). They have identical surface — porting one almost
gives you the other.

## What "unverified" means

🟡 entries compile against the vtable contract; the host running this
audit has no NVIDIA / AMD GPU to actually load and run the binaries.
On the first run on a real GPU:

1. The CPU reference forward (per `tests/precision_cpu_vs_metal.sh`)
   must reproduce on the CUDA / AMD binary at fp32, step 0, bitwise.
2. The kernel golden harness (`tests/golden_kernels.sh`) must pass
   when extended with cross-backend `check_goldens_cuda`.
3. `tests/quant_fastest_smoke.sh` reaches "Hello! How can I help you
   today?" on the 0.8B model.

Until those three gates pass, the 🟡 status is provisional.

## Hardware needs

To take CUDA from 🟡 to ✓ on the 8 implemented slots plus port the
8 missing slots:

- A Linux host with a modern NVIDIA GPU (sm_80+ recommended; sm_70
  works for fp16). 0.8B model is ~500 MB so even an 8 GB card is
  fine.
- CUDA toolkit 12.x with cuBLAS.
- `make cuda` produces `./qw36_cuda`; `./qw36_cuda -m <gguf>` should
  run; precision smoke gates it.

Same for AMD with HIP / ROCm 6.x and a Radeon GPU.

We don't currently have either on this host. CI does compile-check
for CUDA (`linux-cuda-build-check` job in `.github/workflows/ci.yml`),
but cannot exercise the binary. AMD doesn't even have a compile-check
job because the ROCm toolkit installation is fragile in GH runners.

## Porting order (recommended)

If you have GPU hardware to spend on this:

1. Get `precision_cpu_vs_metal.sh` equivalent working on your backend
   — bitwise CPU vs your-backend at fp32 step 0. This validates init
   / upload / matmul / rmsnorm / attention / residual_add /
   embedding_lookup in one shot.
2. Port `residual_rmsnorm` (the easy fused kernel). +3-5% wallclock.
3. Port DN family in this order: `dn_conv1d_silu` → `dn_gated_delta`
   → `dn_gated_rmsnorm`. Each is a small kernel; the recurrent state
   is the only tricky part.
4. Port `moe_forward`. Larger surface (route + top-K + expert combine
   + shared expert). Skip until 35B is actually a target on your
   hardware.
5. Fuse only when the unfused path is solid. `dn_gated_delta_conv1d`
   and `dn_gated_rmsnorm_matmul` are perf wins that can wait.
6. Add quant matmul kernels (Q4_K, Q5_K, Q6_K, affine32 variants,
   Q6K_SCALE16). This is most of the long tail of Metal's kernel set.

Each step lands as its own commit with bench data; the perf-gate
script (`tests/perf_gate.sh`) catches regressions.

## See also

- `common/qw36_gpu.h` — the frozen vtable contract.
- `metal/qw36_metal.m` — the perf reference for every slot.
- [`docs/architecture.md`](architecture.md) § "The vtable contract".
- [`ROADMAP.md`](../ROADMAP.md) theme 1.2 — CUDA/AMD catch-up.
