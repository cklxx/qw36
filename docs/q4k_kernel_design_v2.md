# Q4_K Kernel Design v2

Research-only pass for task R2. No production kernel is proposed here until
the design is reviewed.

## Sources Read

- `docs/q4k_kernel_research_task.md`
- `docs/q4k_qmv_quad_failed.md`
- `metal/qw36_metal.metal` current `qw36_matmul_q4_k_f32` and failed
  `qw36_matmul_q4_k_qmv_quad_f32`
- `metal/qw36_metal.m` MPS and quant matmul dispatch paths
- MLX:
  `/Users/bytedance/code/agent-infer/crates/mlx-sys/vendor/mlx/mlx/backend/metal/kernels/quantized.h`
  (`load_vector`, `qdot`, `qmv_quad_impl`, `qmv_fast_impl`, `qmv_impl`)
- MLX dispatch:
  `/Users/bytedance/code/agent-infer/crates/mlx-sys/vendor/mlx/mlx/backend/metal/quantized.cpp`
- agent-infer GGUF helpers:
  `/Users/bytedance/code/agent-infer/crates/mlx-sys/src/mlx_bridge.cpp`
- Apple SDK headers:
  `MPSMatrixMultiplication.h` and `MPSMatrix.h`

## S2 Implementation Result

Phase S2 landed as an opt-in path behind
`QW36_METAL_QUANT_GPU=1 QW36_METAL_Q4K_AFFINE32=1`. It repacks GGUF Q4_K into
160-byte affine32 blocks at load time, then uses
`qw36_matmul_q4k_affine32_qmv_fast_f32` for decode GEMV.

Correctness gates passed:

```sh
make metal
./tests/precision_cpu_vs_metal.sh
QW36_METAL_QUANT_GPU=1 QW36_METAL_Q4K_AFFINE32=1 ./qw36_metal ... -p Hello -n 16
QW36_METAL_QUANT_GPU=1 QW36_METAL_Q4K_AFFINE32=1 ./qw36_metal ... -p 写一首关于秋天的古诗 -n 32
```

The affine32 kernel is a clear Q4_K kernel win, but not a whole-model 150+
tok/s win because this model still spends most quant-path time in Q5_K/Q6_K
and the fp16 lm_head. Standalone-profile comparison on `Hello -n 16`:

| shape (rows x K) | old Q4_K avg us | affine32 avg us |
|---|---:|---:|
| 3584x1024 | 126.3 | 22.8 |
| 2048x1024 | 82.4 | 14.9 |
| 1024x3584 | 70.3 | 27.3 |
| 4096x1024 | 146.4 | 22.0 |
| 1024x2048 | 49.5 | 17.4 |

End-to-end decode on this host:

| mode | prompt | generated speed |
|---|---|---:|
| fp16 weights | `Hello` | 115-120 tok/s |
| QUANT_GPU old | `Hello` | 57-59 tok/s |
| QUANT_GPU + affine32 | `Hello` | 75-85 tok/s |
| QUANT_GPU + affine32 | `The capital of France is` | 86 tok/s |

Conclusion: keep affine32 opt-in for the low-memory path and use the same
repack/qmv_fast pattern for Q5_K next. Do not make it the default speed path
until the remaining Q5_K/Q6_K bottlenecks are addressed.

## Follow-on: Q5_K/Q6_K Repack

The Q5_K follow-on is implemented behind
`QW36_METAL_QUANT_GPU=1 QW36_METAL_QK_REPACK=1`. It uses the same affine
per-32 scheme as Q4_K, except each 32-element group stores four little-endian
5-byte packs (8 values per pack). Load-time sanity checks compare sampled
rows against the original GGUF Q5_K dequantizer; the observed max absolute
delta on Qwen3.5-0.8B-Q4_K_M is `0.000202417`.

Q6_K was also implemented as an explicit diagnostic path,
`QW36_METAL_Q6K_SCALE16=1`, with `half scale[16]` plus 16 packed 6-bit
groups per 256 elements. It is faster than the old Q6_K row kernel in
standalone profiles, but it is intentionally not enabled by
`QW36_METAL_QK_REPACK=1`: the Q6 path changes enough fp32 weight values
through fp16 scale rounding that it needs golden-logit validation before it
can be called correctness-safe.

Current correctness-safe low-memory path:

```sh
QW36_METAL_QUANT_GPU=1 QW36_METAL_QK_REPACK=1 ./qw36_metal ...
```

Bench snapshot on this host with `tests/quant_kernel_bench.sh`:

| mode | Hello decode speed |
|---|---:|
| QUANT_GPU old | 35-38 tok/s |
| QUANT_GPU + Q4K_AFFINE32 | 53-57 tok/s |
| QUANT_GPU + QK_REPACK (Q4_K+Q5_K) | 99-103 tok/s |
| legacy Q4K_QUAD | 21 tok/s |

## Measurement Notes

`QW36_METAL_PERF=1` disables batch command buffers so absolute per-token speed
is lower than normal decode. The per-kernel averages are still useful for
relative shape comparisons.

Model:

```text
/Users/bytedance/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf
```

Commands:

```sh
QW36_METAL_PERF=1 QW36_METAL_FP16_WEIGHTS=1 ./qw36_metal ... -p Hello -n 16
QW36_METAL_PERF=1 QW36_METAL_QUANT_GPU=1 ./qw36_metal ... -p Hello -n 16
```

### Current fp16 MPS Path

| shape (rows x K) | avg us | fp16 weight bytes | effective read BW |
|---|---:|---:|---:|
| 7168x1024 | 82.7 | 14.7 MB | 177 GB/s |
| 8224x1024 | 101.1 | 16.8 MB | 167 GB/s |
| 248320x1024 | 2021.5 | 508.6 MB | 252 GB/s |
| 1024x3584 | 40.2 | 7.3 MB | 183 GB/s |
| 1024x2048 | 28.2 | 4.2 MB | 149 GB/s |

Earlier batch-mode profiles were faster for small/medium shapes
(`7168x1024` around 61 us, `1024x3584` around 34 us), so use the table as a
conservative standalone-profile view.

### Current Native Quant Path

Q4_K raw row bytes are `(K / 256) * 144`. For `K=1024`, one row is only
576 bytes; for `K=3584`, one row is 2016 bytes.

| kernel | shape | avg us | raw Q4 bytes | raw BW | fp16-equivalent BW |
|---|---:|---:|---:|---:|---:|
| Q4_K old | 3584x1024 | 166.4 | 2.06 MB | 12 GB/s | 44 GB/s |
| Q4_K old | 2048x1024 | 98.2 | 1.18 MB | 12 GB/s | 43 GB/s |
| Q4_K old | 1024x3584 | 90.3 | 2.06 MB | 23 GB/s | 81 GB/s |
| Q4_K old | 1024x2048 | 64.5 | 1.18 MB | 18 GB/s | 65 GB/s |
| Q4_K qmv_quad attempt | 3584x1024 | 235 | 2.06 MB | 9 GB/s | 31 GB/s |
| Q4_K qmv_quad attempt | 1024x3584 | 844 | 2.06 MB | 2 GB/s | 9 GB/s |

The key correction: using qw36 labels (`rows x K`), the Q4_K physical bytes
for `1024x3584` are about 2.06 MB, not 7.2 MB. The 7.3 MB number is the
fp16-equivalent matrix size.

## Q1: Why Is the Old 256-thread Q4_K Kernel So Inefficient?

The old kernel is not saturating memory. Even on the larger `1024x3584`
case it reads only about 2 MB of Q4_K payload in 90 us, which is roughly
23 GB/s physical bandwidth. That is too low to call it bandwidth-bound on an
M4 Pro where a trivial GPU copy reaches about 220 GB/s physical traffic.

The likely bottleneck is instruction and scheduling overhead:

1. **One output row per threadgroup.** For `K=1024`, each of 256 threads
   processes only 4 elements. Kernel launch/row scheduling and two reductions
   dominate useful work.
2. **Two barriers per row.** The optimized old kernel already uses
   `simd_sum`, but it still has one barrier after threadgroup scale-cache fill
   and one barrier for the cross-SIMD partials.
3. **Too little per-thread work for small K.** The `K=1024` hot path has
   high thread count and low arithmetic per thread. Q4_K matmuls with
   `K=1024` dominate q/k/v/gate_up shapes.
4. **Threadgroup-memory scale cache helps but does not fix layout.** It saves
   repeated 6-bit scale unpack, but every element still does dynamic sub-block
   indexing and scalar nibble extraction.
5. **No cross-row reuse.** The same `x[k]` vector is logically used for every
   output row. Hardware cache may catch some of it, but the kernel structure
   does not explicitly reuse `x` across rows the way MLX's qmv variants do.

I did not get useful CLI counter samples. `xctrace list templates` shows
`Metal System Trace`, but there is no `metal-capture` command available in
this Xcode install. A GUI/Xcode capture could still classify barrier vs memory
stalls more directly, but the timing/byte math already rules out pure memory
bandwidth as the primary issue.

## Q2: How Does MPS GEMV Reach High Effective Bandwidth?

MPS internals are private. The public headers confirm only the operation and
layout contract:

- `MPSMatrixVectorMultiplication` computes `y = alpha * op(A) * x + beta * y`.
- It is initialized by `rows` and `columns`.
- `MPSMatrixDescriptor` explicitly says optimal row stride can differ from
  tight rows and exposes `rowBytesForColumns:dataType`.

For the current hot shapes, `rowBytesForColumns` returns the tight fp16 stride:

```text
1024 cols -> 2048 bytes
2048 cols -> 4096 bytes
3584 cols -> 7168 bytes
```

So MPS is not getting a win from hidden padding on these shapes. The likely
strategy is a proprietary row-tiled GEMV:

- many output rows per threadgroup or SIMD tile;
- vectorized fp16 row loads;
- input vector staged in registers/cache and reused across a row tile;
- reduction kept within SIMD/SIMDgroup where possible;
- shape-specialized kernels selected by MPS.

The observed `lm_head 248320x1024` time is the cleanest memory-bandwidth
signal: 508 MB of fp16 weights in about 2.02 ms, about 252 GB/s. That is close
to the copy microbench physical-traffic bound below.

## Q3: What Is the True GGUF Q4_K Decode Cost?

The original rough estimate ("GGUF 5 cycles vs MLX affine 4 cycles") is too
optimistic for direct GGUF layout.

After the old kernel caches `d`, `dmin`, `sc[8]`, and `mn[8]`, one element
still needs:

```text
sb/local/iter/h/lane/sub index math
1 payload byte load
mask or shift to select low/high nibble
threadgroup loads: d, dmin, sc[sub], mn[sub]
q * d * sc - dmin * mn
x[k] load
acc += x * value
```

The scale unpack cost is amortized per 256-element block, but the per-element
sub-block selection and min subtraction remain.

MLX affine `qdot` is not doing the same work:

- for 4-bit it loads a `uint16_t`, covering 4 packed nibbles;
- `load_vector` pre-scales `x_thread` by powers of 16, so `qdot` can multiply
  masked bitfields directly instead of shifting each nibble down;
- bias is applied once as `sum * bias`;
- scale is applied once as `scale * accum`;
- for group-size affine quant, scale/bias are simple arrays indexed by group.

GGUF Q4_K is mathematically affine per 32-element sub-block:

```text
scale = d * sc[sub]
bias  = -dmin * mn[sub]
value = q * scale + bias
```

But the physical payload layout is not MLX-affine packed. Each 64-element
iteration stores low nibbles for sub-block A and high nibbles for sub-block B
in the same 32 bytes. A 32-element sub-block is therefore not stored as the
16 contiguous packed bytes that MLX `qdot` wants.

Conclusion: direct GGUF decode is materially more expensive than MLX affine
qdot. The cost is not just 1 extra cycle; it is both extra instructions and an
unfriendly packing layout.

## Q4: Where Is MLX qmv/qmv_quad Actually Fast?

Important details from MLX:

1. `qmv_quad_impl` uses 32 threads, 8 quads, 8 rows per quad, and `quad_sum`.
   This removes threadgroup barriers entirely.
2. `qdot` is the real fast path. For 4-bit it processes 4 values per
   `uint16_t` load and avoids explicit per-nibble shift-down by pre-scaling x.
3. `load_vector` also computes `sum(x)` so affine bias can be applied as
   `sum * bias`, not as one add per element.
4. The current vendored MLX dispatch uses `qmv_quad` only for small
   `K == 64 || K == 128` with power-of-two bits. For larger K it routes to
   `qmv` / `qmv_fast`, which use 2 or 4 SIMDgroups and 4 rows per SIMDgroup.
5. The common thread is not specifically "quad"; it is **packed affine qdot +
   multi-row per threadgroup + SIMD-local reduction**.

This explains task R's result. The failed qmv_quad Q4_K kernel copied the
32-thread/64-row scheduling but not the packed affine qdot. It serialized too
much GGUF decode inside each quad lane and lost the old kernel's intra-row
parallelism.

## Q5: M=1 GEMV Bandwidth Upper Bound

I used a temporary standalone Metal microbench in `/tmp/qw36_metal_bwbench.m`
(not committed) to measure GPU-to-GPU copy behavior on this host.

Commands:

```sh
clang -fobjc-arc -framework Foundation -framework Metal \
  /tmp/qw36_metal_bwbench.m -o /tmp/qw36_metal_bwbench
/tmp/qw36_metal_bwbench 256 12
/tmp/qw36_metal_bwbench 512 12
/tmp/qw36_metal_bwbench 1024 8
```

Device: Apple M4 Pro.

| size | blit best | compute copy best |
|---:|---:|---:|
| 256 MB | 2381 us, 112.7 GB/s copied | 2373 us, 113.1 GB/s copied |
| 512 MB | 4796 us, 111.9 GB/s copied | 4770 us, 112.5 GB/s copied |
| 1024 MB | 9636 us, 111.4 GB/s copied | 9617 us, 111.7 GB/s copied |

These numbers report one-way copied bytes. Physical memory traffic is read +
write, so the observed physical traffic is roughly 222-226 GB/s. MPS GEMV reads
large weights and writes tiny outputs, so its 180-250 GB/s effective weight
read bandwidth is consistent with the copy bound.

## Target Numbers

Use current MPS as the bar. A Q4_K path that does not beat these per-shape
numbers is not worth replacing fp16 materialization for speed.

| shape | MPS target | acceptable Q4_K v2 target | stretch target |
|---|---:|---:|---:|
| 3584x1024 | <= 60-80 us standalone, <= 60 us batch | <= 70 us | <= 45 us |
| 2048x1024 | <= 50 us standalone | <= 45 us | <= 30 us |
| 1024x3584 | <= 40 us standalone, <= 34 us batch | <= 45 us | <= 30 us |
| 1024x2048 | <= 30 us standalone | <= 35 us | <= 22 us |
| 248320x1024 lm_head | ~2.0 ms fp16 MPS | quant path must be < 1.0 ms to matter | <= 0.8 ms |

For Q4_K `1024x3584`, the physical payload is about 2.06 MB. A perfect
read-only 220 GB/s kernel would read that in about 9.4 us. The realistic target
is therefore set by decode/reduction overhead, not raw memory.

## TG / Thread / Memory Layout Decision Tree

### Reject: Direct qmv_quad over GGUF layout

This was task R:

```text
32 threads / TG
8 quads / TG
8 rows / quad
lane owns K/4 contiguous elements
direct Q4_K decode inside lane
quad_sum
```

Result: correct, but 2-10x slower than old kernel. Do not pursue further.

### Weak Maybe: Direct GGUF multi-row, wider TG

Potential layout:

```text
128 or 256 threads / TG
2-4 output rows / TG
each SIMD owns one row or one row segment
threadgroup cache per-row block constants
simd_sum then one cross-SIMD reduction per row
```

This may reduce row scheduling overhead while retaining intra-row parallelism.
However it still cannot use MLX `qdot`, still does dynamic sub-block decode,
and still needs barriers. I do not expect it to beat MPS reliably.

### Recommended: Load-time Q4_K -> affine32 repack, then MLX-style qmv

Convert each Q4_K row into:

```text
packed_weight_affine32: 16 packed bytes per 32-element group
scale[group] = d * sc[sub]
bias[group]  = -dmin * mn[sub]
```

Then implement a qmv/qmv_fast-style kernel over this repacked layout:

```text
if K <= 128:
    qmv_quad: 32 threads, 8 quads, 64 rows/TG
else:
    qmv_fast/qmv: 2-4 SIMDgroups/TG, 4 rows/SIMDgroup,
                  block_size = values_per_thread * SIMD_SIZE
```

This is no longer "GGUF direct decode"; it is "GGUF-origin weights repacked
into affine32". It keeps Q4 storage compact enough for bandwidth wins while
making the inner loop match MLX's proven design.

Storage cost:

```text
Q4_K original per 256 elems: 144 bytes
affine32 repack per 256 elems:
  8 groups * 16 packed bytes = 128 bytes
  8 scales + 8 biases as fp16 = 32 bytes
  total = 160 bytes
```

That is 11% more than Q4_K but still 3.2x smaller than fp16.

## Inner Loop Pseudo-code: Recommended Repack Path

Load-time repack:

```c
for row in rows:
  for block in K / 256:
    d = f16(block[0:2])
    dmin = f16(block[2:4])
    sc[8], mn[8] = q4_k_unpack_scales(block + 4)
    for sub in 0..7:
      scale[row, group] = half(d * sc[sub])
      bias[row, group] = half(-dmin * mn[sub])

      // write the 32 q values for this sub-block as 16 contiguous bytes
      for j in 0..31 step 2:
        q0 = q4_k_nibble(block, sub, j)
        q1 = q4_k_nibble(block, sub, j + 1)
        packed[row, group][j / 2] = q0 | (q1 << 4)
```

Kernel inner loop:

```metal
// qmv_fast-style for K >= 1024
for k0 in 0..K step block_size:
    load_vector(x + k0 + lane_offset, x_thread)
    sum = sum(x_thread)

    for row_slot in rows_per_simd:
        group = (k0 + lane_offset) / 32
        s = scale[row, group]
        b = bias[row, group]
        wl = packed + row * packed_row_bytes + packed_offset
        acc[row_slot] += qdot_4bit(wl, x_thread, s, b, sum)

for row_slot:
    acc = simd_sum(acc)
    if simd_lid == 0:
        y[row] = acc
```

The important property is that `qdot_4bit` receives contiguous 4-bit groups,
so it can use the MLX trick:

```metal
const device uint16_t *ws = (const device uint16_t *)w;
acc += x0 * (ws[i] & 0x000f)
     + x1 * (ws[i] & 0x00f0)
     + x2 * (ws[i] & 0x0f00)
     + x3 * (ws[i] & 0xf000);
return scale * acc + sum * bias;
```

## Reduction Strategy

Use MLX's split:

- `K <= 128`: `qmv_quad` with `quad_sum`, 64 rows/TG.
- `K >= 1024`: `qmv_fast`/`qmv` style with 2-4 SIMDgroups/TG and
  `simd_sum`, 4 rows per SIMDgroup.

Avoid one-row-per-TG unless `N` is very small. It is the core weakness of the
current old kernel on decode workloads.

## Expected Problems and Mitigations

1. **Repack time at model load.** Mitigation: only repack when
   `QW36_METAL_QUANT_GPU=1` and a new opt-in env is set; cache buffers in the
   existing GPU weight materialization path.
2. **Extra memory vs GGUF Q4_K.** 160 bytes vs 144 bytes per 256 elems is +11%.
   This is acceptable if speed improves and still much smaller than fp16.
3. **Q5_K does not map as cleanly.** Q5_K can be repacked to affine32 with
   5-bit payload, but packing and `qdot` are more complex. Do Q4_K first.
4. **lm_head dominates long decode.** If tied lm_head is Q6_K or Q8_0 in a
   given model, Q4_K work will not solve the largest kernel. Need per-model
   dtype-aware prioritization.
5. **MPS private kernels remain hard to beat.** Any direct user MSL kernel must
   approach 180-250 GB/s effective read bandwidth to beat fp16 MPS. Repacking
   is required to make that plausible.

## Feasibility Conclusion

**Direct GGUF-layout Q4_K kernel: no.** The old kernel is instruction/scheduling
bound, not memory-bound, and the qmv_quad direct port made that worse by
serializing GGUF decode inside 4-lane quads. More tuning of that shape is
unlikely to beat MPS.

**Q4_K-origin affine32 repack + MLX-style qmv: yes, conditionally.** This is
the only path that preserves quant bandwidth while giving the inner loop a
packed `qdot` comparable to MLX. It costs +11% weight storage over GGUF Q4_K
and requires load-time repacking, but it is technically plausible and gives a
clear benchmark gate: beat fp16 MPS on `1024x3584` and `7168/8224x1024`, or
close the quant-speed line and keep native quant only as a low-memory path.
