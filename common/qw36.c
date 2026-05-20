/* qw36.c — CPU reference forward pass + sampling + engine lifecycle.
 *
 * Owner: Claude.
 *
 * This file is the *reference*. Every GPU backend must produce bitwise-
 * identical fp32 output on the golden vectors in tests/. Keep it readable
 * over fast — readability is the whole point.
 */

#include "qw36.h"
#include "qw36_gpu.h"
#include "qw36_gguf.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QW36_VERSION_STR "qw36 0.0.1-scaffold"

const char *qw36_version(void) { return QW36_VERSION_STR; }

struct qw36_engine {
    qw36_config       cfg;
    qw36_weights      weights;
    qw36_gpu_backend *backend;
    qw36_gpu_ctx     *ctx;
    qw36_gguf_file   *gguf;   /* owns the mmap */
};

/* --------------------------------------------------------------------- */
/* Reference math — fp32, scalar. Backends use these as the oracle.       */
/* --------------------------------------------------------------------- */

static void rmsnorm_f32(float *out, const float *x, const float *w,
                        size_t n, float eps)
{
    double ss = 0.0;
    for (size_t i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float scale = (float)(1.0 / sqrt(ss / (double)n + (double)eps));
    for (size_t i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}

static void matmul_f32(float *y, const float *x, const float *w,
                       size_t rows, size_t cols)
{
    /* y[r] = sum_c x[c] * w[r, c]. Row-major W. */
    for (size_t r = 0; r < rows; r++) {
        double acc = 0.0;
        const float *wr = w + r * cols;
        for (size_t c = 0; c < cols; c++) acc += (double)wr[c] * x[c];
        y[r] = (float)acc;
    }
}

static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

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

static void rope_pair(float *x, size_t pair_idx, size_t pos,
                      size_t rot_dim, float theta_base)
{
    /* Apply rotation to (x[2i], x[2i+1]). */
    float inv_freq = 1.0f /
        powf(theta_base, (2.0f * (float)pair_idx) / (float)rot_dim);
    float angle = (float)pos * inv_freq;
    float c = cosf(angle), s = sinf(angle);
    float x0 = x[2 * pair_idx], x1 = x[2 * pair_idx + 1];
    x[2 * pair_idx]     = x0 * c - x1 * s;
    x[2 * pair_idx + 1] = x0 * s + x1 * c;
}

/* attention_f32 — single-token decode against a populated KV cache.
 * Inputs:
 *   x        [hidden]                — pre-attn-norm-ed input
 *   wq/wk/wv [n_heads/n_kv * head_dim, hidden]
 *   q_norm/k_norm [head_dim]
 *   k_cache/v_cache [seq_capacity, n_kv * head_dim] — appended in-place
 * Output:
 *   y        [hidden]                — attention output BEFORE o_proj
 *
 * TODO(Claude): implement. The pattern is:
 *   q = matmul(x, wq); k = matmul(x, wk); v = matmul(x, wv)
 *   for each head: q_h = rmsnorm(q_h, q_norm); k_h = rmsnorm(k_h, k_norm)
 *   rope(q, k, pos)
 *   append k,v to caches at row=seq_pos
 *   for each head: scores = q_h · k_cache[0..=seq_pos, kv_head_of(h)] / sqrt(d)
 *                  softmax(scores); y_h = sum scores * v_cache[...]
 *   concat heads → y
 */
static void attention_f32(float *y, const float *x,
                          const float *wq, const float *wk, const float *wv,
                          const float *q_norm, const float *k_norm,
                          float *k_cache, float *v_cache,
                          uint32_t hidden, uint32_t n_heads, uint32_t n_kv,
                          uint32_t head_dim, uint32_t seq_pos,
                          uint32_t seq_capacity,
                          float rope_theta, float partial_rotary_factor,
                          float rms_eps)
{
    (void)y; (void)x; (void)wq; (void)wk; (void)wv;
    (void)q_norm; (void)k_norm; (void)k_cache; (void)v_cache;
    (void)hidden; (void)n_heads; (void)n_kv; (void)head_dim;
    (void)seq_pos; (void)seq_capacity;
    (void)rope_theta; (void)partial_rotary_factor; (void)rms_eps;
    /* TODO(Claude): see comment above. */
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

    /* TODO(Claude): load GGUF, populate eng->cfg and eng->weights, upload
     * to backend if non-NULL. For now: leave zeroed and report not-impl. */
    (void)gguf_path;
    if (err && err_cap) snprintf(err, err_cap, "qw36_engine_open: TODO");
    free(eng);
    return NULL;
}

void qw36_engine_close(qw36_engine *eng)
{
    if (!eng) return;
    if (eng->backend && eng->backend->destroy && eng->ctx)
        eng->backend->destroy(eng->ctx);
    if (eng->gguf) qw36_gguf_close(eng->gguf);
    free(eng->cfg.layer_types);
    free(eng->weights.layers);
    free(eng);
}

const qw36_config  *qw36_engine_config(const qw36_engine *eng)  { return &eng->cfg; }
const qw36_weights *qw36_engine_weights(const qw36_engine *eng) { return &eng->weights; }

/* --------------------------------------------------------------------- */
/* State                                                                  */
/* --------------------------------------------------------------------- */

qw36_state *qw36_state_new(const qw36_engine *eng, uint32_t seq_capacity)
{
    (void)eng; (void)seq_capacity;
    /* TODO(Claude): allocate KV cache (num_layers entries) and scratch. */
    return NULL;
}

void qw36_state_free(qw36_state *st)
{
    if (!st) return;
    /* TODO(Claude): free k_cache[L], v_cache[L], scratch. */
    free(st);
}

/* --------------------------------------------------------------------- */
/* Forward — dispatches to backend if present, else CPU reference.        */
/* --------------------------------------------------------------------- */

int qw36_forward(qw36_engine *eng, qw36_state *st, uint32_t token)
{
    (void)eng; (void)st; (void)token;
    /* TODO(Claude): wire up the per-layer loop.
     *   for l in 0..L:
     *     h = rmsnorm(x, input_layernorm)
     *     h = attention(h, layer.wq/k/v, q_norm, k_norm, k_cache[l], v_cache[l])
     *     h = matmul(h, o_proj)
     *     x = x + h
     *     h = rmsnorm(x, post_attn_layernorm)
     *     h = mlp(h, gate, up, down)  // or moe if this layer is sparse
     *     x = x + h
     *   x = rmsnorm(x, final_norm)
     *   logits = matmul(x, lm_head)
     */
    return -1;
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
    /* TODO(Claude): full top_k / top_p / temperature path. For now, sample
     * proportional to softmax(logits / T). */
    double maxv = logits[0];
    for (uint32_t i = 1; i < vocab; i++) if (logits[i] > maxv) maxv = logits[i];
    double sum = 0.0;
    /* Reuse caller-supplied scratch is cleaner; for the stub, malloc. */
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
    for (uint32_t i = 0; i < vocab; i++) { acc += p[i]; if (acc >= r) { pick = i; break; } }
    free(p);
    return pick;
}

/* Silence unused-static warnings for helpers that the engine wiring will
 * pick up once the TODOs above are implemented. */
static void qw36_unused_(void) {
    (void)rmsnorm_f32; (void)matmul_f32; (void)silu; (void)swiglu_mlp_f32;
    (void)rope_pair; (void)attention_f32;
}
static void (*qw36_unused_ref_)(void) = qw36_unused_;
