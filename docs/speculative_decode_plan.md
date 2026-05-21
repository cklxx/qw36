# Speculative decoding plan

A design sketch for speculative decoding in qw36. Explicit non-goal
for the current 4-8 week roadmap window (Roadmap 2.6) — captured here
so the next contributor doesn't have to rediscover the design space.

## Goal

Generate N tokens per forward step of the target model instead of 1,
by using a small draft model to propose candidates and verifying them
in a single batched forward of the target. Standard win: 1.5-3× wallclock
on long generation, depending on draft acceptance rate.

## Why deferred for now

- **Multi-variable change.** Speculative decoding interacts with the
  KV cache (rollback on draft rejection), the sampler (verify under
  the target's logit distribution, not the draft's), and the prefill
  path (the draft has its own KV cache). Landing it during active
  kernel churn (flash-attn, KV Q8, bf16 KV) is exactly the kind of
  "N variables changed" debugging trap AGENTS.md §0 warns against.
- **Wrong order.** Spec decoding only helps once decode itself is
  efficient. We're still in the kernel-perf phase; landing spec
  decoding before flash-attn would make the flash-attn bench data
  ambiguous.
- **Draft model selection.** We don't ship a draft model with qw36.
  Picking one (Qwen3.5-0.5B? a 4-layer reduced 0.8B?) is a project
  decision that needs benching once the target perf is locked in.

When the target gets to >250 tok/s short / >150 tok/s sustained on
0.8B with all kernel levers pulled, spec decoding becomes the next
big lever.

## Sketch (when we get there)

### Algorithm

For each batched draft step:

1. **Propose:** run the draft model M times forward greedily,
   producing M candidate tokens `d_1..d_M`.
2. **Verify:** run ONE target forward over the concatenated prefix
   `[prefix, d_1..d_M]`. The target produces M+1 logit distributions
   `p_1..p_{M+1}` (one per prefix position + one for the next token).
3. **Accept:** walk `i = 1..M`, sampling `t_i` from `p_i`:
   - If `t_i == d_i`, accept and continue.
   - If `t_i != d_i`, accept up to `t_{i-1}`, reject `d_i..d_M`, and
     append `t_i` as the corrective token.
4. **Roll back:** the target's KV cache wrote M+1 positions. If we
   accepted K < M tokens, we need to drop the K+1..M tail. The
   target produced K+1 useful tokens (accepted K plus the
   corrective).

The acceptance rate depends entirely on the draft's quality. Even at
50% acceptance we get ~2× on long generation.

### KV cache rollback

The hot bit. Two options:

- **Speculative-write, conditional-commit:** target forwards write
  K/V to scratch positions seq_pos..seq_pos+M. On accept K tokens,
  set `seq_pos += K+1`. On reject all, set `seq_pos += 1` (the
  corrective). The scratch positions get overwritten by the next
  draft batch. This is the simplest model — KV cache acts like a
  ring buffer past seq_pos.
- **Snapshot/restore:** snapshot K/V before each draft, restore on
  reject. Trivially correct but adds memcpy overhead per draft.

We prefer option 1 because it's what existing implementations
(vLLM, exllamav2) do; the only invariant is "anything past seq_pos
is undefined."

### Draft KV cache

The draft has its own per-layer K/V cache, independent of the
target's. Same engine plumbing applies, just with the draft
config. Probably easiest: two `qw36_engine` instances sharing the
same tokenizer (and possibly the same MoE expert weights if the
draft is a layer-trimmed sibling).

### Sampler integration

Greedy is trivial (acceptance ⇔ argmax equality). For temperature >
0, we sample `t_i ~ p_i` and compare to `d_i`. The probability of
acceptance under temperature is:

```
P(accept) = min(1, p_target(d_i) / p_draft(d_i))
```

If we reject, we sample from the residual distribution
`max(0, p_target - p_draft) / sum(max(0, p_target - p_draft))`.
Standard speculative sampling formula; see DeepMind's spec decode
paper.

### Engine API shape

```c
typedef struct qw36_spec_decoder qw36_spec_decoder;

qw36_spec_decoder *qw36_spec_decoder_new(qw36_engine *target,
                                          qw36_engine *draft,
                                          uint32_t draft_lookahead);
void              qw36_spec_decoder_free(qw36_spec_decoder *sd);

/* One spec step. Returns number of accepted tokens (1..lookahead+1).
 * Caller appends accepted tokens and continues. */
int qw36_spec_step(qw36_spec_decoder *sd,
                   qw36_state *target_st,
                   qw36_state *draft_st,
                   uint32_t *out_tokens, uint32_t *out_n);
```

Lives behind a separate file (`common/qw36_spec.c`?) so the rest of
the engine stays single-stream.

## Existing references to lift from

- DeepMind, "Accelerating Large Language Model Decoding with
  Speculative Sampling" (Chen et al, 2023).
- vLLM's `spec_decode_worker` implementation — most production-grade
  open implementation; uses chunked prefill + rejection sampling.
- exllamav2's `speculative.py` — single-stream, similar shape to
  what we'd build.
- agent-infer doesn't have spec decode; we're not lifting from there.

## What we ship in this PR (when it lands)

- New `common/qw36_spec.{h,c}` with the API above.
- KV cache rollback semantics documented; existing prefill path
  unchanged, new spec path lives alongside.
- A draft-model loader path (probably just two GGUFs into two
  `qw36_engine`s; sharing weights is a future optimization).
- One bench cell in `tests/perf_baseline.json` for `spec_decode:128`
  to track acceptance + wallclock.
- A `--draft <model>` CLI flag that opts in.

Estimated effort with a code agent: 2-3 sessions. Not in the current
window because of the cross-cutting nature; revisit once flash-attn
+ Q8 KV are stable.

## Anti-patterns to avoid

- Don't start spec decoding until target decode is dispatch-count
  optimized. A target that pays 24 dispatches per token spec-decoded
  over 4 tokens still pays 24 × 4 = 96 dispatches per spec step.
- Don't roll your own rejection-sampling math from scratch — the
  DeepMind paper has it correct; the implementation traps are in
  the rollback path, not the math.
- Don't ship without a draft-quality bench. A spec decoder that hits
  20% acceptance is slower than no spec decoder; we'd ship a
  regression.
