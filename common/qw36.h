/* qw36.h — Qwen 3.6 inference engine, public API.
 *
 * Architecture summary (mirrors ../agent-infer/crates/qwen35-spec):
 *   - Pre-norm transformer with RMSNorm (eps configurable, default 1e-6).
 *   - Grouped-Query Attention. Q-norm and K-norm applied per head before
 *     RoPE. Partial rotary factor in (0, 1].
 *   - SwiGLU MLP. Dense layers and (optionally) MoE layers with Top-K
 *     routing + shared expert.
 *   - Tied or untied lm_head.
 *
 * The public types here are deliberately POD — no hidden allocations. The
 * engine owns the big tensor blob; backends borrow pointers into it.
 */

#ifndef QW36_H
#define QW36_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------- */
/* Config                                                                 */
/* --------------------------------------------------------------------- */

/* Storage dtype. Numeric values intentionally mirror GGML_TYPE_* so the
 * GGUF loader can pass ggml_type through unchanged when there's a 1:1
 * mapping. QW36_DTYPE_UNSUPPORTED is the sentinel for "we know about it
 * but can't load it yet". */
typedef enum {
    QW36_DTYPE_F32  = 0,
    QW36_DTYPE_F16  = 1,
    QW36_DTYPE_BF16 = 30,
    QW36_DTYPE_Q4_K = 12,
    QW36_DTYPE_Q5_K = 13,
    QW36_DTYPE_Q6_K = 14,
    QW36_DTYPE_Q8_0 = 8,
    QW36_DTYPE_Q3_K = 11,
    QW36_DTYPE_Q2_K = 10,
    /* Internal post-load layouts: GGUF K-quants repacked as per-32 affine
     * groups for Metal qmv kernels. Not GGML/GGUF on-disk dtypes. */
    QW36_DTYPE_Q4K_AFFINE32 = 100,
    QW36_DTYPE_Q5K_AFFINE32 = 101,
    /* Q6_K is symmetric and uses 16-element scale groups, so its internal
     * qmv layout stores fp16 scale[16] plus packed 6-bit values. */
    QW36_DTYPE_Q6K_SCALE16 = 102,
    QW36_DTYPE_UNSUPPORTED = 0xFF
} qw36_dtype;

/* Per-layer attention flavor. Linear-attention layers (Gated DeltaNet) are
 * present in some Qwen3.5/3.6 hybrids; treat as TODO for now. */
typedef enum {
    QW36_ATTN_FULL = 0,
    QW36_ATTN_LINEAR = 1
} qw36_attn_kind;

typedef struct {
    /* Shapes */
    uint32_t hidden_size;
    uint32_t intermediate_size;       /* dense MLP */
    uint32_t num_hidden_layers;
    uint32_t num_attention_heads;
    uint32_t num_key_value_heads;     /* GQA: kv_heads <= heads */
    uint32_t head_dim;                /* explicit; do not derive */
    uint32_t vocab_size;
    uint32_t max_position_embeddings;

    /* Norm */
    float    rms_norm_eps;

    /* RoPE */
    float    rope_theta;
    float    partial_rotary_factor;   /* fraction of head_dim that rotates */
    /* mRoPE (Qwen3.5/3.6): per-axis pair counts. For text decode only the
     * first section uses seq_pos; later sections use axis-2/3/4 positions
     * which are zero, so those pairs effectively stay unrotated. */
    uint32_t rope_sections[4];
    uint32_t rope_n_sections;        /* 0 ⇒ no mRoPE, use plain RoPE */

    /* MoE (zero if dense) */
    uint32_t moe_num_experts;
    uint32_t moe_experts_per_tok;
    uint32_t moe_intermediate_size;
    uint32_t moe_shared_expert_intermediate_size;
    uint32_t moe_decoder_sparse_step; /* every Nth layer is MoE; 1 = all */
    uint8_t  moe_norm_topk_prob;

    /* Gated DeltaNet — zero if model has no DeltaNet layers. */
    uint32_t dn_num_key_heads;
    uint32_t dn_num_value_heads;
    uint32_t dn_key_head_dim;
    uint32_t dn_value_head_dim;
    uint32_t dn_conv_kernel_size;

    /* Head */
    uint8_t  tie_word_embeddings;
    /* Qwen3.5/3.6 vanilla attention has a Q-gate: q_proj output is
     * n_heads * head_dim * 2, and attn_out is multiplied by sigmoid(gate)
     * before o_proj. */
    uint8_t  has_q_gate;

    /* Per-layer flavor (length = num_hidden_layers, owned by config). NULL
     * means "all FULL". */
    qw36_attn_kind *layer_types;
} qw36_config;

/* --------------------------------------------------------------------- */
/* Weights — host-side view. A backend may mirror these to device memory. */
/* --------------------------------------------------------------------- */

typedef struct {
    /* Vanilla full attention (NULL on Gated DeltaNet layers) */
    void *q_proj;       /* [hidden, n_heads * head_dim] */
    void *k_proj;       /* [hidden, n_kv * head_dim]    */
    void *v_proj;       /* [hidden, n_kv * head_dim]    */
    void *o_proj;       /* [n_heads * head_dim, hidden] */
    void *q_norm;       /* [head_dim]                    */
    void *k_norm;       /* [head_dim]                    */
    void *input_layernorm;       /* [hidden] */
    void *post_attn_layernorm;   /* [hidden] */

    /* Gated DeltaNet (NULL on vanilla layers). Tensor names in GGUF:
     *   attn_qkv.weight    — fused QKV  [hidden, q_dim + k_dim + v_dim]
     *   attn_gate.weight   — z gating   [hidden, v_dim]
     *   ssm_alpha.weight   — a-proj     [hidden, n_v_heads]
     *   ssm_beta.weight    — b-proj     [hidden, n_v_heads]
     *   ssm_conv1d.weight  — depthwise short conv  [k, q_dim+k_dim+v_dim]
     *   ssm_dt.bias        — dt bias    [n_v_heads]
     *   ssm_a              — log A      [n_v_heads]
     *   ssm_norm.weight    — per-val-head RMSNorm gain [val_dim]
     *   ssm_out.weight     — out proj   [v_dim, hidden]
     */
    void *dn_qkv;
    void *dn_gate;
    void *dn_alpha;
    void *dn_beta;
    void *dn_conv1d;     /* short, materialized fp32 (kernel*channels small) */
    void *dn_dt_bias;
    void *dn_a_log;
    void *dn_norm;
    void *dn_out;

    /* MLP — dense path */
    void *gate_proj;    /* [hidden, intermediate] */
    void *up_proj;      /* [hidden, intermediate] */
    void *down_proj;    /* [intermediate, hidden] */

    /* MLP — MoE path (NULL on dense layers) */
    void *moe_router;           /* [hidden, num_experts]         */
    void *moe_expert_gate;      /* [num_experts, hidden, inter]  */
    void *moe_expert_up;        /* [num_experts, hidden, inter]  */
    void *moe_expert_down;      /* [num_experts, inter, hidden]  */
    void *moe_shared_gate;
    void *moe_shared_up;
    void *moe_shared_down;

    qw36_dtype dtype;
} qw36_layer_weights;

typedef struct {
    void *embed_tokens;   /* [vocab, hidden] */
    void *final_norm;     /* [hidden] */
    void *lm_head;        /* [hidden, vocab] — may alias embed_tokens */
    qw36_layer_weights *layers;
    qw36_dtype dtype;
} qw36_weights;

/* --------------------------------------------------------------------- */
/* Forward state — per-sequence KV cache + scratch buffers.               */
/* --------------------------------------------------------------------- */

typedef struct {
    /* KV cache: [num_layers][seq_len, n_kv * head_dim] */
    void   **k_cache;
    void   **v_cache;
    uint32_t num_layers;    /* length of k_cache / v_cache arrays */
    uint32_t seq_capacity;
    uint32_t seq_pos;       /* how many tokens have been written */
    qw36_dtype kv_dtype;
    /* Dtype of the residual stream / attn-output device buffers (x_dev,
     * x_rms_dev, q_dev). When QW36_DTYPE_F16 the kernels store fp16
     * directly and the engine's host↔device sync helpers convert. */
    qw36_dtype dev_x_dtype;

    /* Gated DeltaNet per-layer state (NULL on vanilla layers):
     *   conv_state[L]:  short-window history for ssm_conv1d (depthwise).
     *                   Size = (kernel_size - 1) * (q_dim + k_dim + v_dim) floats.
     *   delta_state[L]: rank-1 state S of shape [n_v_heads, key_dim, val_dim]. */
    float **conv_state;
    float **delta_state;

    /* Scratch (host or device). Engine reuses across steps. */
    float *x;               /* [hidden] residual */
    float *x_rms;
    float *q;               /* [n_heads * head_dim] */
    /* Qwen3.5/3.6 Q-gate scratch. Only allocated when config.has_q_gate.
     * q_full holds the raw q_proj output (size = 2 * q_dim), and q_gate
     * holds the de-interleaved per-head gate (size = q_dim). */
    float *q_full;
    float *q_gate;
    float *k;
    float *v;
    float *attn_scores;     /* [n_heads, seq_capacity] */
    float *gate;            /* [intermediate] */
    float *up;
    float *logits;          /* [vocab] */

    /* Persistent backend buffers. These are opaque qw36_gpu_buf* values kept
     * as void* here so the public C header does not depend on qw36_gpu.h. */
    void *gpu_backend;
    void *gpu_ctx;
    void **k_cache_dev;
    void **v_cache_dev;
    void *x_dev;
    void *x_rms_dev;
    void *q_dev;
    void *k_dev;
    void *v_dev;
    void *attn_scores_dev;
    void *gate_dev;
    void *up_dev;
    void *logits_dev;
    void **conv_state_dev;
    void **delta_state_dev;
    void *dn_qkv_dev;
    void *dn_qkv_act_dev;
    void *dn_z_dev;
    void *dn_alpha_dev;
    void *dn_beta_dev;
    void *dn_gout_dev;
} qw36_state;

/* --------------------------------------------------------------------- */
/* Engine handle (opaque to callers).                                     */
/* --------------------------------------------------------------------- */

typedef struct qw36_engine qw36_engine;
struct qw36_gguf_file;     /* forward — see qw36_gguf.h */
struct qw36_gpu_backend;   /* forward — see qw36_gpu.h  */

/* Construction / destruction. */
qw36_engine *qw36_engine_open(const char *gguf_path,
                              struct qw36_gpu_backend *backend,
                              char *err, size_t err_cap);
void         qw36_engine_close(qw36_engine *eng);

const qw36_config  *qw36_engine_config(const qw36_engine *eng);
const qw36_weights *qw36_engine_weights(const qw36_engine *eng);

/* Borrowed pointer to the engine's GGUF file — used to construct a
 * tokenizer without reopening. Lifetime = engine. */
const struct qw36_gguf_file *qw36_engine_gguf(const qw36_engine *eng);

/* Allocate / free per-sequence state. */
qw36_state *qw36_state_new(const qw36_engine *eng, uint32_t seq_capacity);
void        qw36_state_free(qw36_state *st);

/* Single decode step. Reads token at position st->seq_pos, advances pos,
 * fills st->logits. Returns 0 on success, -1 on error. */
int qw36_forward(qw36_engine *eng, qw36_state *st, uint32_t token);

/* Prefill: feed multiple tokens at once (length tokens, starting at
 * st->seq_pos). On return logits hold the distribution at the LAST token. */
int qw36_prefill(qw36_engine *eng, qw36_state *st,
                 const uint32_t *tokens, size_t length);

/* --------------------------------------------------------------------- */
/* Sampling                                                               */
/* --------------------------------------------------------------------- */

typedef struct {
    float temperature;   /* <= 0 ⇒ argmax */
    float top_p;         /* 0 or 1 ⇒ disabled */
    int   top_k;         /* 0 ⇒ disabled */
    uint64_t rng_seed;
} qw36_sampler;

uint32_t qw36_sample(const float *logits, uint32_t vocab,
                     qw36_sampler *s);

/* --------------------------------------------------------------------- */
/* Error helpers                                                          */
/* --------------------------------------------------------------------- */

const char *qw36_version(void);

#ifdef __cplusplus
}
#endif

#endif /* QW36_H */
