# Q4_K qmv_quad Metal GEMV Design

This pass adapts MLX's `qmv_quad_impl` scheduling pattern to the GGUF
Q4_K block layout used by qw36. The goal is an opt-in decode GEMV path for
`M=1`, where one 32-thread SIMD threadgroup emits up to 64 output rows and
uses only quad-level reductions.

## Reference Shape

MLX affine qmv_quad:

- 32 threads per threadgroup, split into 8 quadgroups.
- Each quadgroup computes 8 output rows.
- Each quad lane owns `K / 4` input elements.
- `quad_sum` reduces the 4 lane partial sums for each output row.
- No threadgroup memory, no threadgroup barriers.

qw36 Q4_K differences:

- One row is `K / 256` GGUF Q4_K blocks.
- Each Q4_K block is 144 bytes: `d`, `dmin`, 12 scale/min bytes, and 128
  nibble payload bytes.
- Scale/min changes every 32 values, so MLX's affine `qdot(sum * bias)`
  shortcut does not apply directly.
- The first implementation decodes Q4_K directly in each quad lane and keeps
  the current block constants in thread-local registers.

## Thread Mapping

For each threadgroup:

```text
threadgroup tg.y = output-row tile

32 threads
  quad 0 -> rows base + 0,  8, 16, 24, 32, 40, 48, 56
  quad 1 -> rows base + 1,  9, 17, 25, 33, 41, 49, 57
  ...
  quad 7 -> rows base + 7, 15, 23, 31, 39, 47, 55, 63

inside each quad:
  lane 0 -> k [0*K/4, 1*K/4)
  lane 1 -> k [1*K/4, 2*K/4)
  lane 2 -> k [2*K/4, 3*K/4)
  lane 3 -> k [3*K/4, 4*K/4)
```

Each lane computes 8 fp32 accumulators, one per row owned by its quad. Lane
partials are reduced with `quad_sum`. Lane 0 writes the 8 final fp32 outputs.

## Data Flow

```text
host metal_matmul
  |
  | QW36_METAL_QUANT_GPU=1
  | QW36_METAL_Q4K_QUAD=1
  | dtype(Q4_K), batch=1, x/y=f32
  v
qw36_matmul_q4_k_qmv_quad_f32
  |
  | per TG: 64 output rows
  | per quad: 8 rows
  | per lane: contiguous K/4 slice
  v
for each row and lane-local k
  |
  | cache current Q4_K block constants:
  |   d, dmin, sc[8], mn[8]
  | decode nibble for k
  | acc[row] += x[k] * (q * d * sc[sub] - dmin * mn[sub])
  v
quad_sum(acc[row])
  |
  v
y[row] = fp32 result
```

## Correctness Notes

- The kernel keeps the old `qw36_matmul_q4_k_f32` path intact and is selected
  only by `QW36_METAL_Q4K_QUAD=1`.
- Output remains fp32, matching the existing quant matmul contract.
- The row layout and Q4_K value formula are intentionally identical to the
  existing scalar helper:

```text
iter = (k & 255) >> 6
h    = ((k & 255) >> 5) & 1
lane = (k & 255) & 31
sub  = iter * 2 + h
byte = payload[iter * 32 + lane]
q    = h == 0 ? byte & 0x0f : byte >> 4
val  = q * (d * sc[sub]) - dmin * mn[sub]
```

## Performance Risks

- This first kernel is direct-decode, not a fully packed `qdot` equivalent.
  It removes threadgroup barriers and improves row tiling, but it may still be
  limited by per-lane Q4_K unpack work.
- K is runtime in the initial implementation to keep host integration simple.
  If benchmarks are promising, a follow-up can add K-specialized entrypoints
  for 1024, 2048, and 3584 so the compiler can unroll lane-local loops.
- Q5_K and Q6_K are not changed here. If Q4_K wins materially, they should get
  matching qmv_quad kernels instead of falling back to the 256-thread path.
