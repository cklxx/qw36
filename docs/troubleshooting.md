# Troubleshooting

Common failure modes and what to check. Once `--doctor` lands (task
3.1), each entry below will be linked from its corresponding FAIL line.

## Build issues

### `make all` succeeds but only `qw36_cpu` exists

Expected on a machine without Metal SDK / nvcc / hipcc. `make all` is
intentionally soft-fail: missing GPU toolchain → that backend is
skipped silently.

**Confirm with:** `ls -la qw36_*`. You should see `qw36_cpu` plus
whichever GPU binaries the host's toolchain could build.

**Force a specific backend:** `make -C metal` (etc.) errors loudly if
the toolchain isn't there.

### `xcrun: error: missing required argument` on macOS

You're missing the Command Line Tools. Run `xcode-select --install`,
then retry `make metal`.

### `nvcc: command not found` but you have CUDA installed

`nvcc` isn't on `PATH`. Either:

```bash
export PATH="/usr/local/cuda/bin:$PATH"   # adjust to your CUDA path
make cuda
```

or set `NVCC=/usr/local/cuda/bin/nvcc`. The Makefile honors the env.

### CUDA build wants `sm_<n>` for an older GPU

The default `arch=native` requires nvcc 11.5+. On older toolchains:

```bash
make -C cuda NVCCFLAGS="-O3 -std=c++17 -arch=sm_80"
```

The CI compile-check uses `sm_80` for the same reason — it has no GPU
to query.

## Runtime issues

### `qw36_metal: missing general.architecture`

The GGUF you passed isn't a Qwen3.5/3.6 family checkpoint. Re-check
the file with `./qw36_dump_tensor <model.gguf>`. qw36 only supports
`general.architecture = qwen35` (Qwen3.5 dense + DN hybrid) or
`qwen35moe` (Qwen3.6 MoE + DN hybrid).

### Output is gibberish

A short list, ordered by likelihood:

1. **You're running the wrong special-token mode.** Try
   `--no-special` (treats prompt as raw, no chat-template). If output
   improves, your prompt was being wrapped in a chat template that
   the model rejected.
2. **The model has untied embeddings and `--fast` quantized lm_head**
   doesn't yet support the architecture variant. Compare CPU vs Metal:
   ```bash
   ./qw36_cpu     -m <gguf> -p "Hello" -n 8 --no-special
   ./qw36_metal --fast -m <gguf> -p "Hello" -n 8 --no-special
   ```
   If CPU is coherent and Metal isn't, file an issue with the GGUF
   metadata (`./qw36_metal --info -m <gguf>`) and the divergent
   outputs.
3. **It's actually correct and the model's just bad.** Try a known-
   good prompt: `Hello` on Qwen3.5-0.8B-Q4_K_M should produce
   `Hello! How can I help you today?`. If that works but your prompt
   doesn't, the issue is your prompt or model choice, not qw36.

### `qw36_metal: metal: no MTLDevice`

You're on a host without Metal (Linux, Intel Mac running an old
macOS, headless macOS). qw36 falls back to CPU silently when run as
`./qw36_cpu`; the Metal binary itself errors.

### Out of memory at prefill on a large model

The `lazy_materialize_f16` path converts quantized weights to fp16,
which doubles to ~2x file size. For Qwen3.6-35B-A3B (22 GB GGUF) the
fp16 materialization would need ~44 GB — too big for most Mac
machines.

Use `--fast` (or `--profile=fast`) which keeps weights quantized on
GPU. `--profile=reference` is for precision testing only, not for
large models.

If even `--fast` OOMs, drop to CPU: `./qw36_cpu ...`.

### Tokens come out repeated (e.g. `bye bye bye bye`)

Sampler is greedy (`-t 0`) and the model genuinely wants that token.
Try a temperature: `-t 0.7 --top-p 0.9`. If the repetition persists
across sampler settings, the model's forward is producing very
peaked logits — file an issue with the model + prompt.

### Prefill is slow on first run, fast on second

First-run mmap pays disk reads for a 22 GB GGUF. Subsequent runs read
from the page cache. Expect 30-60s prefill on a cold cache for a 35B
model; that's not qw36 being slow.

## Performance issues

### tok/s much lower than the FINAL_STATUS.md ladder

Check, in order:

1. **Load.** `uptime | awk -F'load averages:' '{print $2}'`. If 1-min
   avg > 3, you're contending with WindowServer and other Metal
   clients. Quote the load in your bug report.
2. **Profile.** Did you pass `--fast`? Without it, you get the
   conservative reference path (~75 tok/s, not 200+).
3. **Wallclock vs PERF.** `QW36_METAL_PERF=1` disables the persistent
   compute encoder. PERF tok/s ≈ 1/4 of normal. Don't bench under
   PERF.
4. **GPU actually engaged?** Run `QW36_METAL_PERF=1 ... -n 2` and look
   at the `[metal-perf]` table. If the kernels you expect aren't
   there with non-zero `gpu_ms`, your build is falling back to CPU
   for some op. See `memory/feedback_check_gpu_first.md`.

### `--fast` gives gibberish on a model that worked before

You're hitting a `--fast`-path-only bug. Drop to `--profile=reference`
(or just no profile flag) to confirm:

```bash
./qw36_metal -m <gguf> -p "Hello" -n 8 --no-special   # default
./qw36_metal --fast -m <gguf> -p "Hello" -n 8 --no-special
```

If reference is coherent and `--fast` isn't, file the issue with
both outputs and the GGUF metadata.

### Long-context (n > 1024) is slow

Expected. qw36 attention is O(seq_len) per token; at n=2048 each
token spends ~30% of its time in attention. The KV cache transpose
(`b4bb6f6`) is opt-in via `QW36_METAL_KV_TRANSPOSED=1` and helps
~24% at n=2048 — at the cost of -10% at n=64. We don't default it
on (task 2.4 will add an auto heuristic).

Flash-attn single-pass (task 2.2 / #71 Z) is the next lever; not
landed yet.

## "But MLX is faster"

Yes, by ~16% on `Hello -n 64` (204 vs 244 tok/s for us vs MLX on
Qwen3.5-0.8B-Q4_K_M on the same machine), and 3.3× on long-context
essay (100 vs 320 tok/s at n=1024). MLX has a graph compiler we
don't, and they've spent more engineer-months on Apple Silicon
quant kernels.

What we offer that MLX doesn't:

- One C binary per backend, no Python.
- Works on CUDA / AMD HIP from the same vtable (modulo task 1.2's
  catch-up work).
- Auditable forward: every layer has a CPU reference and a precision
  smoke gate.

If raw tok/s is your only criterion on Apple Silicon, MLX is right.
If you need a small, portable engine you can read end-to-end in an
afternoon, qw36 is right.

## Filing a bug

Include:

1. `git rev-parse HEAD` (commit you built from)
2. `./qw36_metal --info -m <gguf> 2>&1 | head -20` (config the engine
   loaded)
3. The exact command line
4. The full stderr output up to the failure
5. `uptime` 1-min load average if it's a perf claim
6. Whether CPU and Metal agree on the output

The repo's `docs/qwen36_35b_a3b_status.md` is a worked example of
how a useful issue should read: symptom, what's known to work, what
was bisected, residual unknowns.
