# tests/

Golden vectors + a tiny harness. Each backend must produce identical fp32
output on these inputs.

Layout (to be filled in):

```
tests/
├── golden_rmsnorm.bin       # in: [hidden] x; out: [hidden] y
├── golden_matmul.bin        # in: x, W; out: y
├── golden_attention.bin     # in: x + KV; out: y
├── golden_swiglu.bin
└── harness.c                # runs each backend against the golden bins
```

Owner: Claude.
