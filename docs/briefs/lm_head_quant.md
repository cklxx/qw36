# Codex brief — lm_head quant path (target: +30-50% on triple-affine)

**Context**: We just landed Q4K_AFFINE32 + Q5K_AFFINE32 + Q6K_SCALE16
repack for layer weights (commits 766b7c0 / 90b1377 / b5e0ecb). Result:
85 → 170 tok/s short / 139 sustained. Beats fp16 MPS (119 / 103).

**The remaining lever**: `lm_head` is `output.weight` aliased to
`token_embd.weight` (Qwen3.5-0.8B has tied embeddings). It's Q6_K in
the GGUF but currently force-materialized to fp16 (`common/qw36.c`
lines 912-934) because the comment said "152K threadgroups, far slower
than MPS fp16 gemv".

That comment predates our `qw36_matmul_q6k_scale16_qmv_fast_f32` kernel
(metal/qw36_metal.metal:1952) which uses 16-row TGs. For 152K rows that's
9500 TGs — feasible on M-class GPU.

**Measured impact**: I added a temporary `QW36_HACK_SKIP_LM_HEAD=1` env
hack and ran 64-token sustained generation. WITH lm_head: 100 tok/s.
WITHOUT lm_head matmul: 154 tok/s. **lm_head costs ~3.4ms/token**.
Quantizing it (fp16 → Q6K_SCALE16) should halve its bandwidth, saving
~1.5-2ms = +30-50 tok/s. Reverting the hack now.

## Your task

Add opt-in env `QW36_METAL_QUANT_GPU_LM_HEAD=1`. When set:

1. Detect `w->lm_head == w->embed_tokens` (alias case for tied embeddings).
2. Allocate a *new* `qw36_lazy_w` for lm_head — distinct from embed_tokens.
   - Copy `data/ggml_type/rows/cols/n_extra` from the original. Keep
     `dtype = QW36_DTYPE_Q6_K` (or `Q5_K`) and `gpu_buf = NULL` initially.
   - Use `eng_own` for the new lazy_w struct so cleanup is automatic.
3. Apply the existing `lazy_repack_q6k_to_scale16` (or `_q5k_to_affine32`)
   on the new lazy_w — this mutates `data` / `dtype` in place to the
   affine layout and sets `gpu_buf = NULL`.
4. Re-point `w->lm_head` to the new lazy_w.
5. The first `qw36__matmul_lazy_dev` call will trigger upload via the
   existing path (qw36__gpu_cached_upload → metal_upload). The
   metal_matmul dispatch at metal/qw36_metal.m:1238 already routes
   `Q6K_SCALE16` to the right kernel — **no metal changes needed**.

**Don't touch**:
- `embed_tokens` — must remain fp16 GPU buffer for the embedding_lookup
  fast path (`qw36__embed_lookup_lazy_dev`).
- The alias path when env is **not** set — current behavior must be
  bit-identical (`./qw36_metal -m <gguf> -p Hello -n 64` produces
  identical "Hello! How can I help you today?" output).

## Verify

```bash
make -C metal
M=/Users/bytedance/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf

# Sanity: output preserved
QW36_METAL_QUANT_GPU=1 \
QW36_METAL_Q4K_AFFINE32=1 \
QW36_METAL_Q5K_AFFINE32=1 \
QW36_METAL_Q6K_SCALE16=1 \
QW36_METAL_QUANT_GPU_LM_HEAD=1 \
./qw36_metal -m "$M" -p "Hello" -n 64

# Expect: "Hello! How can I help you today?"

# Bench 3x
for i in 1 2 3; do
  QW36_METAL_QUANT_GPU=1 QW36_METAL_Q4K_AFFINE32=1 \
  QW36_METAL_Q5K_AFFINE32=1 QW36_METAL_Q6K_SCALE16=1 \
  QW36_METAL_QUANT_GPU_LM_HEAD=1 \
  ./qw36_metal -m "$M" -p "Write a long essay" -n 256 2>&1 | \
  grep "generated.*tok/s" | tail -1
done
```

Compare against baseline (without `QW36_METAL_QUANT_GPU_LM_HEAD=1`). Need
≥+20% to call this a win. If correctness fails or perf is flat, leave it
opt-in and document `docs/lm_head_quant_*.md` as the next failed attempt.

## Files to touch

- `common/qw36.c` — engine init (around lines 912-940 and the repack loop
  ~975-1050). Add the env check, the new lazy_w alloc, and the repack.

That's it. The metal side already has the kernel — your job is host wiring.

## Done criteria

- `make -C metal` clean (no new warnings)
- Output matches baseline character-for-character on "Hello" prompt
- 3× n=256 essay bench shows tok/s vs no-env baseline
- Commit with bench numbers in the message
