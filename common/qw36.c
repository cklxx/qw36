/* qw36.c — CPU reference forward pass + sampling + engine lifecycle.
 *
 * Owner: Claude.
 *
 * This file is the *reference*. Every GPU backend must produce fp32
 * output that matches this within tolerance on the golden vectors in
 * tests/. Keep it readable over fast — readability is the whole point.
 */

#include "qw36.h"
#include "qw36_gpu.h"
#include "qw36_gguf.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QW36_VERSION_STR "qw36 0.0.1-scaffold"

const char *qw36_version(void) { return QW36_VERSION_STR; }

struct qw36_engine {
    qw36_config       cfg;
    qw36_weights      weights;
    qw36_gpu_backend *backend;     /* may be NULL → CPU-only */
    qw36_gpu_ctx     *ctx;
    qw36_gguf_file   *gguf;

    /* Per-layer f32 buffers we materialized from quantized weights. The
     * engine owns these. NULL entries mean "no conversion needed; use the
     * raw mmap pointer in weights.*". */
    float            **owned_f32;
    size_t             owned_f32_n;

    /* Cached arch prefix ("qwen3" / "qwen3moe" / ...) — used to fetch
     * config keys. */
    char               arch[32];
};

/* --------------------------------------------------------------------- */
/* dtype conversion helpers                                               */
/* --------------------------------------------------------------------- */

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign << 31; }
        else {
            /* denorm → normalize */
            int e = 1;
            while (!(mant & 0x400u)) { mant <<= 1; e--; }
            mant &= 0x3FFu;
            f = (sign << 31) | ((uint32_t)(e + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = (sign << 31) | (0xFFu << 23) | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    union { uint32_t u; float f; } u; u.u = f; return u.f;
}

static float bf16_to_f32(uint16_t b) {
    union { uint32_t u; float f; } u;
    u.u = (uint32_t)b << 16;
    return u.f;
}

/* Read element i of a tensor in storage dtype, return fp32. */
static float load_elem_f32(const void *data, qw36_dtype dt, size_t i) {
    switch (dt) {
        case QW36_DTYPE_F32:  return ((const float    *)data)[i];
        case QW36_DTYPE_F16:  return f16_to_f32(((const uint16_t *)data)[i]);
        case QW36_DTYPE_BF16: return bf16_to_f32(((const uint16_t *)data)[i]);
        default: return 0.0f; /* quantized — TODO */
    }
}

/* Materialize an entire tensor to a freshly malloc'd fp32 buffer. Caller
 * frees. Returns NULL on unsupported dtype / oom. */
static float *materialize_f32(const void *data, qw36_dtype dt, size_t n) {
    float *out = (float *)malloc(n * sizeof(float));
    if (!out) return NULL;
    switch (dt) {
        case QW36_DTYPE_F32:
            memcpy(out, data, n * sizeof(float));
            return out;
        case QW36_DTYPE_F16: {
            const uint16_t *p = (const uint16_t *)data;
            for (size_t i = 0; i < n; i++) out[i] = f16_to_f32(p[i]);
            return out;
        }
        case QW36_DTYPE_BF16: {
            const uint16_t *p = (const uint16_t *)data;
            for (size_t i = 0; i < n; i++) out[i] = bf16_to_f32(p[i]);
            return out;
        }
        default:
            free(out);
            return NULL;
    }
}

/* --------------------------------------------------------------------- */
/* CPU reference math — all fp32, scalar. The GPU backends must match.   */
/* --------------------------------------------------------------------- */

static void rmsnorm_f32(float *out, const float *x, const float *w,
                        size_t n, float eps)
{
    double ss = 0.0;
    for (size_t i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float scale = (float)(1.0 / sqrt(ss / (double)n + (double)eps));
    for (size_t i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}

/* y[r] = sum_c W[r,c] * x[c]. Row-major W, shape [rows, cols]. */
static void matmul_f32(float *y, const float *x, const float *w,
                       size_t rows, size_t cols)
{
    for (size_t r = 0; r < rows; r++) {
        double acc = 0.0;
        const float *wr = w + r * cols;
        for (size_t c = 0; c < cols; c++) acc += (double)wr[c] * x[c];
        y[r] = (float)acc;
    }
}

static inline float silu(float x) { return x / (1.0f + expf(-x)); }

static void swiglu_mlp_f32(float *y, const float *x,
                           const float *w_gate, const float *w_up,
                           const float *w_down,
                           float *scratch_gate, float *scratch_up,
                           size_t hidden, size_t inter)
{
    matmul_f32(scratch_gate, x, w_gate, inter, hidden);
    matmul_f32(scratch_up,   x, w_up,   inter, hidden);
    for (size_t i = 0; i < inter; i++)
        scratch_gate[i] = silu(scratch_gate[i]) * scratch_up[i];
    matmul_f32(y, scratch_gate, w_down, hidden, inter);
}

/* In-place RoPE on a single head. Rotates the first `rot_dim` components;
 * the tail is left untouched (partial rotary). Pair convention:
 * (x[2i], x[2i+1]) — the "interleaved" convention Qwen3 uses. */
static void rope_head(float *x, size_t pos, size_t rot_dim, float theta_base)
{
    size_t pairs = rot_dim / 2;
    for (size_t i = 0; i < pairs; i++) {
        float inv_freq = 1.0f /
            powf(theta_base, (2.0f * (float)i) / (float)rot_dim);
        float angle = (float)pos * inv_freq;
        float c = cosf(angle), s = sinf(angle);
        float x0 = x[2*i], x1 = x[2*i + 1];
        x[2*i]     = x0 * c - x1 * s;
        x[2*i + 1] = x0 * s + x1 * c;
    }
}

/* attention_f32 — single-token decode against a populated KV cache.
 *
 * Steps:
 *   1. q = x @ Wq^T, split into n_heads heads of size head_dim.
 *      k = x @ Wk^T, v = x @ Wv^T, split into n_kv heads.
 *   2. Per-head RMSNorm with q_norm / k_norm.
 *   3. RoPE on q and k at position seq_pos.
 *   4. Append k, v to k_cache[seq_pos], v_cache[seq_pos].
 *   5. For each query head h, dot it against k_cache rows 0..=seq_pos of
 *      its kv head (h mod n_kv), scaled by 1/sqrt(head_dim), softmax,
 *      then weighted sum with v_cache.
 *   6. Output concat is `y`, shape [hidden = n_heads * head_dim]. The
 *      caller multiplies by o_proj.
 *
 * `attn_scratch` must point to at least (n_heads * (seq_capacity + 1))
 * floats — workspace for the per-head score arrays.
 */
static void attention_f32(float *y,
                          const float *x,
                          const float *wq, const float *wk, const float *wv,
                          const float *q_norm, const float *k_norm,
                          float *k_cache, float *v_cache,
                          uint32_t hidden, uint32_t n_heads, uint32_t n_kv,
                          uint32_t head_dim, uint32_t seq_pos,
                          uint32_t seq_capacity,
                          float rope_theta, float partial_rotary_factor,
                          float rms_eps,
                          float *attn_scratch)
{
    const uint32_t kv_dim    = n_kv * head_dim;
    const uint32_t q_dim     = n_heads * head_dim;
    const uint32_t rot_dim   = (uint32_t)((float)head_dim * partial_rotary_factor);

    /* Project. We reuse y as scratch for q (sized hidden = q_dim). */
    float *q = y;                                 /* [q_dim]  */
    float *k_row = k_cache + (size_t)seq_pos * kv_dim; /* row in cache */
    float *v_row = v_cache + (size_t)seq_pos * kv_dim;

    matmul_f32(q,     x, wq, q_dim,  hidden);
    matmul_f32(k_row, x, wk, kv_dim, hidden);
    matmul_f32(v_row, x, wv, kv_dim, hidden);

    /* Q-norm / K-norm per head, then RoPE. */
    for (uint32_t h = 0; h < n_heads; h++) {
        float *qh = q + h * head_dim;
        rmsnorm_f32(qh, qh, q_norm, head_dim, rms_eps);
        rope_head(qh, seq_pos, rot_dim, rope_theta);
    }
    for (uint32_t h = 0; h < n_kv; h++) {
        float *kh = k_row + h * head_dim;
        rmsnorm_f32(kh, kh, k_norm, head_dim, rms_eps);
        rope_head(kh, seq_pos, rot_dim, rope_theta);
    }

    /* Attention. */
    const float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);
    for (uint32_t h = 0; h < n_heads; h++) {
        const uint32_t kvh = h % n_kv;
        const float *qh = q + h * head_dim;
        float *scores = attn_scratch + (size_t)h * (seq_capacity + 1);

        /* Dot product with each cached k. */
        double maxv = -INFINITY;
        for (uint32_t t = 0; t <= seq_pos; t++) {
            const float *kh = k_cache + (size_t)t * kv_dim + kvh * head_dim;
            double dot = 0.0;
            for (uint32_t d = 0; d < head_dim; d++)
                dot += (double)qh[d] * kh[d];
            scores[t] = (float)dot * inv_sqrt_d;
            if (scores[t] > maxv) maxv = scores[t];
        }
        /* Softmax. */
        double sum = 0.0;
        for (uint32_t t = 0; t <= seq_pos; t++) {
            scores[t] = (float)exp((double)scores[t] - maxv);
            sum += scores[t];
        }
        float inv_sum = (float)(1.0 / sum);
        for (uint32_t t = 0; t <= seq_pos; t++) scores[t] *= inv_sum;

        /* Weighted sum of v rows → into a fresh "head_out" we keep in
         * the scratch tail (per-head, since y aliases q above). After all
         * heads are computed we write back to y. */
    }

    /* Second pass: now that all heads have scores, materialize each
     * head's output into y in place. We need the head outputs first
     * (since y was holding q above). Use the tail of attn_scratch for
     * the n_heads * head_dim staging buffer. */
    float *staging = attn_scratch + (size_t)n_heads * (seq_capacity + 1);
    for (uint32_t h = 0; h < n_heads; h++) {
        const uint32_t kvh = h % n_kv;
        const float *scores = attn_scratch + (size_t)h * (seq_capacity + 1);
        float *out = staging + h * head_dim;
        for (uint32_t d = 0; d < head_dim; d++) {
            double acc = 0.0;
            for (uint32_t t = 0; t <= seq_pos; t++) {
                const float *vh = v_cache + (size_t)t * kv_dim + kvh * head_dim;
                acc += (double)scores[t] * vh[d];
            }
            out[d] = (float)acc;
        }
    }
    memcpy(y, staging, (size_t)q_dim * sizeof(float));
}

/* --------------------------------------------------------------------- */
/* GGUF → config + weights binding                                        */
/* --------------------------------------------------------------------- */

static int eng_get_u32(const qw36_engine *eng, const char *suffix, uint32_t *out) {
    char key[128];
    snprintf(key, sizeof(key), "%s.%s", eng->arch, suffix);
    return qw36_gguf_get_u32(eng->gguf, key, out);
}
static int eng_get_f32(const qw36_engine *eng, const char *suffix, float *out) {
    char key[128];
    snprintf(key, sizeof(key), "%s.%s", eng->arch, suffix);
    return qw36_gguf_get_f32(eng->gguf, key, out);
}

/* Bind a tensor by name → fp32-materialized pointer, recorded in
 * eng->owned_f32 so it gets freed on close. */
static void *bind_tensor_f32(qw36_engine *eng, const char *name) {
    qw36_gguf_tensor t;
    if (qw36_gguf_get_tensor(eng->gguf, name, &t)) return NULL;
    size_t numel = 1;
    for (uint32_t d = 0; d < t.n_dims; d++) numel *= (size_t)t.dims[d];
    float *p = materialize_f32(t.data, t.dtype, numel);
    if (!p) return NULL;
    /* track for free */
    eng->owned_f32 = (float **)realloc(eng->owned_f32,
        sizeof(float *) * (eng->owned_f32_n + 1));
    eng->owned_f32[eng->owned_f32_n++] = p;
    return p;
}

/* Same but optional: returns NULL if tensor is absent (no error). */
static void *bind_tensor_f32_opt(qw36_engine *eng, const char *name) {
    qw36_gguf_tensor t;
    if (qw36_gguf_get_tensor(eng->gguf, name, &t)) return NULL;
    return bind_tensor_f32(eng, name);
}

/* --------------------------------------------------------------------- */
/* Engine lifecycle                                                       */
/* --------------------------------------------------------------------- */

qw36_engine *qw36_engine_open(const char *gguf_path,
                              qw36_gpu_backend *backend,
                              char *err, size_t err_cap)
{
    qw36_engine *eng = (qw36_engine *)calloc(1, sizeof(*eng));
    if (!eng) {
        if (err && err_cap) snprintf(err, err_cap, "oom");
        return NULL;
    }
    eng->backend = backend;

    eng->gguf = qw36_gguf_open(gguf_path, err, err_cap);
    if (!eng->gguf) { free(eng); return NULL; }

    /* Detect architecture. We accept any qwen3-family string. */
    const char *arch = NULL;
    if (qw36_gguf_get_str(eng->gguf, "general.architecture", &arch) || !arch) {
        if (err && err_cap) snprintf(err, err_cap, "missing general.architecture");
        qw36_engine_close(eng); return NULL;
    }
    snprintf(eng->arch, sizeof(eng->arch), "%s", arch);

    qw36_config *c = &eng->cfg;
    if (eng_get_u32(eng, "embedding_length",      &c->hidden_size) ||
        eng_get_u32(eng, "feed_forward_length",   &c->intermediate_size) ||
        eng_get_u32(eng, "block_count",           &c->num_hidden_layers) ||
        eng_get_u32(eng, "attention.head_count",  &c->num_attention_heads) ||
        eng_get_u32(eng, "attention.head_count_kv", &c->num_key_value_heads) ||
        eng_get_u32(eng, "context_length",        &c->max_position_embeddings))
    {
        if (err && err_cap) snprintf(err, err_cap, "missing required %s.* config key",
                                     eng->arch);
        qw36_engine_close(eng); return NULL;
    }
    /* head_dim — Qwen3 stores attention.key_length; fall back to hidden/heads. */
    if (eng_get_u32(eng, "attention.key_length", &c->head_dim)) {
        c->head_dim = c->hidden_size / c->num_attention_heads;
    }
    /* vocab — read from the tokens array via the gguf module. */
    {
        qw36_gguf_tensor unused;
        (void)unused;
        /* Best-effort: many qwen models also set <arch>.vocab_size. */
        if (eng_get_u32(eng, "vocab_size", &c->vocab_size) != 0) {
            /* Fallback to the embedding table row count. */
            qw36_gguf_tensor t;
            if (qw36_gguf_get_tensor(eng->gguf, "token_embd.weight", &t) == 0
                && t.n_dims >= 2) {
                c->vocab_size = (uint32_t)t.dims[1];
            }
        }
    }
    if (eng_get_f32(eng, "attention.layer_norm_rms_epsilon", &c->rms_norm_eps))
        c->rms_norm_eps = 1e-6f;
    if (eng_get_f32(eng, "rope.freq_base", &c->rope_theta))
        c->rope_theta = 1000000.0f;
    c->partial_rotary_factor = 1.0f;
    {
        float prf;
        if (eng_get_f32(eng, "rope.partial_rotary_factor", &prf) == 0 && prf > 0)
            c->partial_rotary_factor = prf;
    }
    {
        uint32_t tie = 0;
        eng_get_u32(eng, "tie_word_embeddings", &tie);
        c->tie_word_embeddings = (uint8_t)tie;
    }

    /* MoE config — optional. Picks up qwen3moe.expert_count etc. */
    eng_get_u32(eng, "expert_count",              &c->moe_num_experts);
    eng_get_u32(eng, "expert_used_count",         &c->moe_experts_per_tok);
    eng_get_u32(eng, "expert_feed_forward_length",&c->moe_intermediate_size);
    if (!c->moe_decoder_sparse_step) c->moe_decoder_sparse_step = 1;

    /* Bind global tensors. */
    qw36_weights *w = &eng->weights;
    w->dtype = QW36_DTYPE_F32; /* CPU reference materializes to fp32 */
    w->embed_tokens = bind_tensor_f32(eng, "token_embd.weight");
    w->final_norm   = bind_tensor_f32(eng, "output_norm.weight");
    w->lm_head      = bind_tensor_f32_opt(eng, "output.weight");
    if (!w->lm_head && c->tie_word_embeddings) w->lm_head = w->embed_tokens;
    if (!w->embed_tokens || !w->final_norm) {
        if (err && err_cap) snprintf(err, err_cap, "missing embedding or output norm");
        qw36_engine_close(eng); return NULL;
    }

    /* Bind per-layer tensors. */
    w->layers = (qw36_layer_weights *)calloc(c->num_hidden_layers,
                                             sizeof(qw36_layer_weights));
    if (!w->layers) {
        if (err && err_cap) snprintf(err, err_cap, "oom (layers)");
        qw36_engine_close(eng); return NULL;
    }
    char name[128];
    for (uint32_t l = 0; l < c->num_hidden_layers; l++) {
        qw36_layer_weights *L = &w->layers[l];
        L->dtype = QW36_DTYPE_F32;
        #define BIND(field, fmt) do { \
            snprintf(name, sizeof(name), fmt, l);             \
            L->field = bind_tensor_f32_opt(eng, name);        \
        } while (0)

        BIND(input_layernorm,     "blk.%u.attn_norm.weight");
        BIND(q_proj,              "blk.%u.attn_q.weight");
        BIND(k_proj,              "blk.%u.attn_k.weight");
        BIND(v_proj,              "blk.%u.attn_v.weight");
        BIND(o_proj,              "blk.%u.attn_output.weight");
        BIND(q_norm,              "blk.%u.attn_q_norm.weight");
        BIND(k_norm,              "blk.%u.attn_k_norm.weight");
        BIND(post_attn_layernorm, "blk.%u.ffn_norm.weight");
        BIND(gate_proj,           "blk.%u.ffn_gate.weight");
        BIND(up_proj,             "blk.%u.ffn_up.weight");
        BIND(down_proj,           "blk.%u.ffn_down.weight");

        /* MoE-specific (Qwen3-MoE). Absent on dense models — that's fine. */
        BIND(moe_router,          "blk.%u.ffn_gate_inp.weight");
        BIND(moe_expert_gate,     "blk.%u.ffn_gate_exps.weight");
        BIND(moe_expert_up,       "blk.%u.ffn_up_exps.weight");
        BIND(moe_expert_down,     "blk.%u.ffn_down_exps.weight");
        BIND(moe_shared_gate,     "blk.%u.ffn_gate_shexp.weight");
        BIND(moe_shared_up,       "blk.%u.ffn_up_shexp.weight");
        BIND(moe_shared_down,     "blk.%u.ffn_down_shexp.weight");
        #undef BIND
    }

    /* TODO(Claude): if backend is non-NULL, upload weights and switch the
     * forward path to use the backend vtable. For v0 we always run CPU. */
    (void)backend;
    return eng;
}

void qw36_engine_close(qw36_engine *eng)
{
    if (!eng) return;
    if (eng->backend && eng->backend->destroy && eng->ctx)
        eng->backend->destroy(eng->ctx);
    if (eng->owned_f32) {
        for (size_t i = 0; i < eng->owned_f32_n; i++) free(eng->owned_f32[i]);
        free(eng->owned_f32);
    }
    free(eng->cfg.layer_types);
    free(eng->weights.layers);
    if (eng->gguf) qw36_gguf_close(eng->gguf);
    free(eng);
}

const qw36_config  *qw36_engine_config(const qw36_engine *eng)  { return &eng->cfg; }
const qw36_weights *qw36_engine_weights(const qw36_engine *eng) { return &eng->weights; }
const struct qw36_gguf_file *qw36_engine_gguf(const qw36_engine *eng) { return eng->gguf; }

/* --------------------------------------------------------------------- */
/* State                                                                  */
/* --------------------------------------------------------------------- */

qw36_state *qw36_state_new(const qw36_engine *eng, uint32_t seq_capacity)
{
    const qw36_config *c = &eng->cfg;
    qw36_state *st = (qw36_state *)calloc(1, sizeof(*st));
    if (!st) return NULL;
    st->seq_capacity = seq_capacity;
    st->seq_pos      = 0;
    st->kv_dtype     = QW36_DTYPE_F32;
    st->num_layers   = c->num_hidden_layers;

    const size_t kv_dim   = (size_t)c->num_key_value_heads * c->head_dim;
    const size_t q_dim    = (size_t)c->num_attention_heads * c->head_dim;
    const size_t hidden   = c->hidden_size;
    const size_t inter    = c->intermediate_size ? c->intermediate_size : 1;
    const size_t vocab    = c->vocab_size;
    const size_t L        = c->num_hidden_layers;

    st->k_cache = (void **)calloc(L, sizeof(void *));
    st->v_cache = (void **)calloc(L, sizeof(void *));
    if (!st->k_cache || !st->v_cache) goto fail;
    for (size_t l = 0; l < L; l++) {
        st->k_cache[l] = calloc((size_t)seq_capacity * kv_dim, sizeof(float));
        st->v_cache[l] = calloc((size_t)seq_capacity * kv_dim, sizeof(float));
        if (!st->k_cache[l] || !st->v_cache[l]) goto fail;
    }

    /* Scratch — big enough for attention staging (n_heads*(cap+1) +
     * n_heads*head_dim) and for MLP gate/up. */
    const size_t attn_scratch_n =
        (size_t)c->num_attention_heads * ((size_t)seq_capacity + 1)
      + (size_t)c->num_attention_heads * c->head_dim;
    st->x           = (float *)calloc(hidden,        sizeof(float));
    st->x_rms       = (float *)calloc(hidden,        sizeof(float));
    st->q           = (float *)calloc(q_dim,         sizeof(float));
    st->k           = (float *)calloc(kv_dim,        sizeof(float));
    st->v           = (float *)calloc(kv_dim,        sizeof(float));
    st->attn_scores = (float *)calloc(attn_scratch_n, sizeof(float));
    st->gate        = (float *)calloc(inter,         sizeof(float));
    st->up          = (float *)calloc(inter,         sizeof(float));
    st->logits      = (float *)calloc(vocab,         sizeof(float));
    if (!st->x || !st->x_rms || !st->q || !st->k || !st->v ||
        !st->attn_scores || !st->gate || !st->up || !st->logits) goto fail;

    return st;

fail:
    qw36_state_free(st);
    return NULL;
}

void qw36_state_free(qw36_state *st)
{
    if (!st) return;
    if (st->k_cache) {
        for (uint32_t l = 0; l < st->num_layers; l++) free(st->k_cache[l]);
        free(st->k_cache);
    }
    if (st->v_cache) {
        for (uint32_t l = 0; l < st->num_layers; l++) free(st->v_cache[l]);
        free(st->v_cache);
    }
    free(st->x); free(st->x_rms); free(st->q); free(st->k); free(st->v);
    free(st->attn_scores); free(st->gate); free(st->up); free(st->logits);
    free(st);
}

/* --------------------------------------------------------------------- */
/* Forward — CPU reference per-layer loop.                                */
/* --------------------------------------------------------------------- */

int qw36_forward(qw36_engine *eng, qw36_state *st, uint32_t token)
{
    if (!eng || !st) return -1;
    const qw36_config  *c = &eng->cfg;
    const qw36_weights *w = &eng->weights;
    if (token >= c->vocab_size) return -1;
    if (st->seq_pos >= st->seq_capacity) return -1;

    const size_t hidden = c->hidden_size;
    const size_t inter  = c->intermediate_size;

    /* Embed: copy row `token` from embed_tokens into st->x. */
    {
        const float *row = (const float *)w->embed_tokens + (size_t)token * hidden;
        memcpy(st->x, row, hidden * sizeof(float));
    }

    for (uint32_t l = 0; l < c->num_hidden_layers; l++) {
        const qw36_layer_weights *L = &w->layers[l];
        float *x = st->x;

        /* --- attention block --- */
        rmsnorm_f32(st->x_rms, x, (const float *)L->input_layernorm, hidden, c->rms_norm_eps);

        /* attn_in = x_rms ; attn_out (pre-o_proj) lives in st->q (reused) */
        attention_f32(st->q,
                      st->x_rms,
                      (const float *)L->q_proj,
                      (const float *)L->k_proj,
                      (const float *)L->v_proj,
                      (const float *)L->q_norm,
                      (const float *)L->k_norm,
                      (float *)st->k_cache[l],
                      (float *)st->v_cache[l],
                      c->hidden_size, c->num_attention_heads,
                      c->num_key_value_heads, c->head_dim,
                      st->seq_pos, st->seq_capacity,
                      c->rope_theta, c->partial_rotary_factor,
                      c->rms_norm_eps,
                      st->attn_scores);

        /* o_proj: hidden = (n_heads * head_dim) → hidden */
        const uint32_t q_dim = c->num_attention_heads * c->head_dim;
        matmul_f32(st->x_rms, st->q, (const float *)L->o_proj, hidden, q_dim);

        /* Residual */
        for (size_t i = 0; i < hidden; i++) x[i] += st->x_rms[i];

        /* --- mlp block --- */
        rmsnorm_f32(st->x_rms, x, (const float *)L->post_attn_layernorm,
                    hidden, c->rms_norm_eps);

        if (L->moe_router) {
            /* TODO(Claude): MoE forward. For now, fall back to dense if
             * dense weights are also present, else zero contribution. */
            if (L->gate_proj && L->up_proj && L->down_proj) {
                swiglu_mlp_f32(st->x_rms, st->x_rms,
                               (const float *)L->gate_proj,
                               (const float *)L->up_proj,
                               (const float *)L->down_proj,
                               st->gate, st->up, hidden, inter);
            } else {
                memset(st->x_rms, 0, hidden * sizeof(float));
            }
        } else {
            swiglu_mlp_f32(st->x_rms, st->x_rms,
                           (const float *)L->gate_proj,
                           (const float *)L->up_proj,
                           (const float *)L->down_proj,
                           st->gate, st->up, hidden, inter);
        }

        for (size_t i = 0; i < hidden; i++) x[i] += st->x_rms[i];
    }

    /* Final norm + lm_head. */
    rmsnorm_f32(st->x_rms, st->x, (const float *)w->final_norm,
                hidden, c->rms_norm_eps);
    matmul_f32(st->logits, st->x_rms, (const float *)w->lm_head,
               c->vocab_size, hidden);

    st->seq_pos++;
    return 0;
}

int qw36_prefill(qw36_engine *eng, qw36_state *st,
                 const uint32_t *tokens, size_t length)
{
    for (size_t i = 0; i < length; i++) {
        int rc = qw36_forward(eng, st, tokens[i]);
        if (rc) return rc;
    }
    return 0;
}

/* --------------------------------------------------------------------- */
/* Sampling                                                               */
/* --------------------------------------------------------------------- */

static uint64_t splitmix64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

uint32_t qw36_sample(const float *logits, uint32_t vocab, qw36_sampler *s)
{
    if (s->temperature <= 0.0f) {
        uint32_t best = 0;
        float bv = logits[0];
        for (uint32_t i = 1; i < vocab; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
        return best;
    }
    double maxv = logits[0];
    for (uint32_t i = 1; i < vocab; i++) if (logits[i] > maxv) maxv = logits[i];
    double sum = 0.0;
    double *p = (double *)malloc(sizeof(double) * vocab);
    if (!p) return 0;
    for (uint32_t i = 0; i < vocab; i++) {
        p[i] = exp(((double)logits[i] - maxv) / (double)s->temperature);
        sum += p[i];
    }
    double r = (double)(splitmix64(&s->rng_seed) >> 11) /
               (double)(1ULL << 53) * sum;
    double acc = 0.0;
    uint32_t pick = vocab - 1;
    for (uint32_t i = 0; i < vocab; i++) {
        acc += p[i];
        if (acc >= r) { pick = i; break; }
    }
    free(p);
    return pick;
}
