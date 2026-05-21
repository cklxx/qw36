# Model support matrix

What we test (and what we don't). Updated 2026-05-21.

| level | meaning |
|-------|---------|
| 🟢 | tested in CI / smoke; perf number published |
| 🟡 | functional smoke; perf cycle pending |
| 🔵 | should work (same arch family) but not regularly tested |
| 🔴 | known broken / not yet supported |
| —  | not applicable (e.g. MoE on a dense model) |

## Qwen3.5 / Qwen3.6 family

| model              | quant     | CPU | Metal `--fast` | CUDA  | AMD HIP | notes |
|--------------------|-----------|-----|-----------------|-------|---------|-------|
| Qwen3.5-0.8B       | Q4_K_M    | 🟢   | 🟢 **204 / 176 / 92 tok/s**       | 🔵     | 🔵       | reference; CI smoke; perf target |
| Qwen3.5-0.8B       | Q5_K_M    | 🟢   | 🟢                | 🔵     | 🔵       | registered in `tools/download_model.sh` |
| Qwen3.5-0.8B       | Q8_0      | 🟢   | 🟢                | 🔵     | 🔵       | registered |
| Qwen3.5-1.7B       | Q4_K_M    | 🔵   | 🔵                | 🔵     | 🔵       | same arch as 0.8B; not regularly tested |
| Qwen3.5-4B         | Q4_K_M    | 🔵   | 🔵                | 🔵     | 🔵       | should work; not regularly tested |
| Qwen3.5-8B         | Q4_K_M    | 🔵   | 🔵                | 🔵     | 🔵       | should work; not regularly tested |
| Qwen3.5-14B        | Q4_K_M    | 🔵   | 🔵                | 🔵     | 🔵       | should work; not regularly tested |
| Qwen3.6-35B-A3B    | Q4_K_XL   | 🟡   | 🟡 functional   | —     | —       | MoE+DN aligned (`59ebce1`); perf cycle in progress |
| Qwen3.5-72B        | Q4_K_M    | 🔴   | 🔴                | —     | —       | likely OOM on 64 GB M-class; needs streaming weight path |

CPU / Metal 🔵 entries should reasonably work because they share the
Qwen3.5/3.6 architecture (vanilla GQA + Gated DeltaNet hybrid) — but
they are not in CI. If you try one and see degeneracies, file an
issue with `--info` + the first divergent step.

CUDA / AMD column is 🔵 across the board because those backends are
compile-checked but not runtime-checked on this host. Roadmap theme
1.2 catches them up.

## Other GGUF model families (not supported)

qw36 explicitly **does not** support:

- Llama 2 / 3 / 3.1 / 3.2 family (different attention layout, no DN)
- Mistral / Mixtral (different MoE layout)
- Phi-3 / Phi-3.5 (different normalization)
- Gemma 1 / 2 (different positional encoding)
- DeepSeek-V2 / V3 (Multi-head Latent Attention; out of scope)
- Anything with `general.architecture` other than `qwen35` or
  `qwen35moe`

The engine arch-detection in `qw36_engine_open` rejects unknown
`general.architecture` with a clear error rather than silently
producing garbage. If you want to add a family, see
`docs/architecture.md` § "Adding a new model architecture".

## What "tested" actually means per cell

- 🟢 for 0.8B-Q4_K_M means: CI runs `tests/precision_cpu_vs_metal.sh`
  at step 0 (bitwise CPU == Metal fp32), `tests/quant_fastest_smoke.sh`
  (`Hello!` → "How can I help you today?"), `tests/kernel_golden.sh`,
  and the perf-gate fires at ±10% from `tests/perf_baseline.json`.
- 🟡 for 35B-A3B means: `./qw36_cpu` and `./qw36_metal --fast`
  produce coherent text (not byte-equal — Metal has minor numerical
  drift on the longer model). Smoke is manual, not CI.
- 🔵 means: nobody has tested this combination, the architecture
  suggests it should work. Bug reports welcome.
- 🔴 means: known broken. Roadmap item to address.

## How to add a model to the registry

1. Drop the GGUF in `~/.cache/qw36/models/` (any path is fine — the
   tools just need an absolute path).
2. Add a row to `tools/download_model.sh`'s `REGISTRY` heredoc with
   the HF repo + file + size.
3. Smoke it locally:
   ```bash
   ./qw36_metal --doctor -m <path>
   ./qw36_metal --fast -m <path> -p Hello -n 16
   ```
4. If it produces coherent text, mark 🟡 in this matrix. If you also
   write a CI smoke + add it to `tests/perf_baseline.json`, mark 🟢.

## See also

- [`README.md`](../README.md) § "Supported models" — short version
  pointing here.
- [`docs/architecture.md`](architecture.md) — engine internals + how
  to add a new architecture.
- [`docs/troubleshooting.md`](troubleshooting.md) — symptom → check
  guide; first stop when a model misbehaves.
