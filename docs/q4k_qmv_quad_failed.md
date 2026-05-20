# Q4_K qmv_quad Experiment Result

The first GGUF Q4_K qmv_quad-style Metal kernel is correct but too slow to
enable by default.

## Benchmark

Command:

```sh
./tests/quant_kernel_bench.sh
```

Results on Qwen3.5-0.8B-Q4_K_M:

```text
fp16 MPS path:              117-121 tok/s
QUANT_GPU old kernels:       58-60 tok/s
QUANT_GPU + Q4K_QUAD:        29-30 tok/s
```

The opt-in path still produces the expected continuations:

```text
Hello! How can I help you today?
```

and the Chinese poem sanity prompt still emits a coherent `《秋夜》` style
poem. The failure is performance, not correctness.

## Profile

With `QW36_METAL_PERF=1 QW36_METAL_QUANT_GPU=1 QW36_METAL_Q4K_QUAD=1`:

```text
q4k_quad_3584x1024   avg 235 us
q4k_quad_1024x3584   avg 844 us
q4k_quad_2048x1024   avg 233 us
q4k_quad_1024x2048   avg 460 us
```

This is substantially slower than the old 256-thread Q4_K kernel and much
slower than the fp16 MPS path.

## Root Cause

MLX qmv_quad is fast because its affine quant format lets each quad lane run a
packed `qdot`: nibbles are unpacked from `uint32_t`, one scale/bias pair covers
the group, and the bias term can use a precomputed input sum.

GGUF Q4_K does not map cleanly onto that shortcut:

- each 256-element block has `d`, `dmin`, and 8 separate scale/min pairs;
- scale/min changes every 32 values;
- the direct port must decode Q4_K inside each lane for every row and value;
- using 32 threads per 64 rows creates long serial lane-local loops
  (`K / 4 * 8` decodes per thread) instead of the old kernel's wide
  256-thread row parallelism.

So the qmv_quad schedule removes barriers but loses too much intra-row
parallelism for GGUF direct-decode.

## Decision

`QW36_METAL_Q4K_QUAD=1` remains an opt-in experiment only. The default
`QW36_METAL_QUANT_GPU=1` path continues to use the old Q4_K kernel.

The next viable route is not this direct port. Better candidates:

- convert GGUF Q4_K weights at load time into an MLX-affine/qmv-friendly packed
  layout, then use a true packed qdot kernel;
- write a row-tiled Q4_K kernel that keeps more than 32 threads per row while
  reducing cross-SIMD barrier cost;
- skip GGUF-native decode for speed and keep fp16 materialization as the
  current fast path.
