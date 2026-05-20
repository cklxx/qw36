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

typedef enum {
    QW36_DTYPE_F32 = 0,
    QW36_DTYPE_F16 = 1,
    QW36_DTYPE_BF16 = 2,
    QW36_DTYPE_Q8_0 = 8,
    QW36_DTYPE_Q4_K = 12,
    QW36_DTYPE_Q2_K = 10
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

    /* MoE (zero if dense) */
    uint32_t moe_num_experts;
    uint32_t moe_experts_per_tok;
    uint32_t moe_intermediate_size;
    uint32_t moe_shared_expert_intermediate_size;
    uint32_t moe_decoder_sparse_step; /* every Nth layer is MoE; 1 = all */
    uint8_t  moe_norm_topk_prob;

    /* Head */
    uint8_t  tie_word_embeddings;

    /* Per-layer flavor (length = num_hidden_layers, owned by config). NULL
     * means "all FULL". */
    qw36_attn_kind *layer_types;
} qw36_config;

/* --------------------------------------------------------------------- */
/* Weights — host-side view. A backend may mirror these to device memory. */
/* --------------------------------------------------------------------- */

typedef struct {
    /* Attention */
    void *q_proj;       /* [hidden, n_heads * head_dim] */
    void *k_proj;       /* [hidden, n_kv * head_dim]    */
    void *v_proj;       /* [hidden, n_kv * head_dim]    */
    void *o_proj;       /* [n_heads * head_dim, hidden] */
    void *q_norm;       /* [head_dim]                    */
    void *k_norm;       /* [head_dim]                    */
    void *input_layernorm;       /* [hidden] */
    void *post_attn_layernorm;   /* [hidden] */

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
    uint32_t seq_capacity;
    uint32_t seq_pos;       /* how many tokens have been written */
    qw36_dtype kv_dtype;

    /* Scratch (host or device). Engine reuses across steps. */
    float *x;               /* [hidden] residual */
    float *x_rms;
    float *q;               /* [n_heads * head_dim] */
    float *k;
    float *v;
    float *attn_scores;     /* [n_heads, seq_capacity] */
    float *gate;            /* [intermediate] */
    float *up;
    float *logits;          /* [vocab] */
} qw36_state;

/* --------------------------------------------------------------------- */
/* Engine handle (opaque to callers).                                     */
/* --------------------------------------------------------------------- */

typedef struct qw36_engine qw36_engine;

struct qw36_gpu_backend; /* forward — see qw36_gpu.h */

/* Construction / destruction. */
qw36_engine *qw36_engine_open(const char *gguf_path,
                              struct qw36_gpu_backend *backend,
                              char *err, size_t err_cap);
void         qw36_engine_close(qw36_engine *eng);

const qw36_config  *qw36_engine_config(const qw36_engine *eng);
const qw36_weights *qw36_engine_weights(const qw36_engine *eng);

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
