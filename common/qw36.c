/* qw36.c — CPU reference forward pass + sampling + engine lifecycle.
 *
 * Owner: Claude.
 *
 * This file is the *reference*. Every GPU backend must produce fp32
 * output that matches this within tolerance on the golden vectors in
 * tests/. Keep it readable over fast — readability is the whole point.
 */

#include "qw36_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QW36_VERSION_STR "qw36 0.0.1-scaffold"

const char *qw36_version(void) { return QW36_VERSION_STR; }

void *qw36__eng_own(qw36_engine *eng, void *p) {
    if (!p) return NULL;
    if (eng->owned_n >= eng->owned_cap) {
        size_t nc = eng->owned_cap ? eng->owned_cap * 2 : 64;
        void **arr = (void **)realloc(eng->owned, nc * sizeof(void *));
        if (!arr) return NULL;
        eng->owned = arr;
        eng->owned_cap = nc;
    }
    eng->owned[eng->owned_n++] = p;
    return p;
}

/* dtype conversion and block dequantization live in qw36_dequant.c. */

/* Set/cleared by qw36_forward so qw36__matmul_lazy can locate the backend
 * without threading the engine pointer through every helper signature.
 * Single-threaded today; promote to __thread when batched/multi-stream. */
qw36_engine *qw36__active_engine = NULL;
int qw36__skip_logits_this_forward = 0;

qw36_gpu_buf *qw36__gpu_cached_upload(qw36_engine *eng, const void *host,
                                        size_t bytes, qw36_dtype dtype)
{
    if (!eng || !eng->backend || !eng->ctx || !eng->backend->upload ||
        !host || !bytes)
        return NULL;
    for (size_t i = 0; i < eng->gpu_cache_n; i++) {
        qw36_gpu_cache_entry *e = &eng->gpu_cache[i];
        if (e->host == host && e->bytes == bytes && e->dtype == dtype)
            return e->gpu_buf;
    }
    if (eng->gpu_cache_n >= eng->gpu_cache_cap) {
        size_t nc = eng->gpu_cache_cap ? eng->gpu_cache_cap * 2 : 128;
        qw36_gpu_cache_entry *arr =
            (qw36_gpu_cache_entry *)realloc(eng->gpu_cache,
                                            nc * sizeof(*arr));
        if (!arr) return NULL;
        eng->gpu_cache = arr;
        eng->gpu_cache_cap = nc;
    }
    qw36_gpu_buf *gb = eng->backend->upload(eng->ctx, host, bytes, dtype);
    if (!gb) return NULL;
    eng->gpu_cache[eng->gpu_cache_n++] = (qw36_gpu_cache_entry){
        host, bytes, dtype, gb
    };
    return gb;
}

void qw36__gpu_cache_free(qw36_engine *eng)
{
    if (!eng) return;
    if (eng->backend && eng->backend->free && eng->ctx) {
        for (size_t i = 0; i < eng->gpu_cache_n; i++)
            eng->backend->free(eng->ctx, eng->gpu_cache[i].gpu_buf);
    }
    free(eng->gpu_cache);
    eng->gpu_cache = NULL;
    eng->gpu_cache_n = eng->gpu_cache_cap = 0;
}

int qw36__state_backend(qw36_state *st, qw36_gpu_backend **be_out,
                         qw36_gpu_ctx **ctx_out)
{
    if (!st || !st->gpu_backend || !st->gpu_ctx) return 0;
    qw36_gpu_backend *be = (qw36_gpu_backend *)st->gpu_backend;
    qw36_gpu_ctx *ctx = (qw36_gpu_ctx *)st->gpu_ctx;
    if (!be->alloc || !be->free || !be->download || !be->copy_from_host)
        return 0;
    if (be_out) *be_out = be;
    if (ctx_out) *ctx_out = ctx;
    return 1;
}

int qw36__state_copy_from_host(qw36_state *st, void *dst_dev,
                                const void *src, size_t bytes)
{
    qw36_gpu_backend *be;
    qw36_gpu_ctx *ctx;
    if (!dst_dev || !src || !qw36__state_backend(st, &be, &ctx)) return -1;
    be->copy_from_host(ctx, (qw36_gpu_buf *)dst_dev, src, bytes);
    return 0;
}

int qw36__state_download_to_host(qw36_state *st, void *src_dev,
                                  void *dst, size_t bytes)
{
    qw36_gpu_backend *be;
    qw36_gpu_ctx *ctx;
    if (!src_dev || !dst || !qw36__state_backend(st, &be, &ctx)) return -1;
    be->download(ctx, (qw36_gpu_buf *)src_dev, dst, bytes);
    return 0;
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

/* Materialize a small tensor (norm-sized) to fp32. Used for layernorms,
 * biases, A_log — anything small enough that the dequant pass is cheap. */
static float *bind_tensor_f32(qw36_engine *eng, const char *name) {
    qw36_gguf_tensor t;
    if (qw36_gguf_get_tensor(eng->gguf, name, &t)) return NULL;
    size_t numel = 1;
    for (uint32_t d = 0; d < t.n_dims; d++) numel *= (size_t)t.dims[d];
    float *p = qw36__materialize_f32(t.data, t.dtype, numel);
    if (!p) return NULL;
    if (!qw36__eng_own(eng, p)) { free(p); return NULL; }
    return p;
}

static float *bind_tensor_f32_opt(qw36_engine *eng, const char *name) {
    qw36_gguf_tensor t;
    if (qw36_gguf_get_tensor(eng->gguf, name, &t)) return NULL;
    return bind_tensor_f32(eng, name);
}

/* Materialize a lazy_w's quantized weight to a fresh fp32 host buffer
 * and flip the descriptor to point at it. Memory is registered on the
 * engine for free-on-close. Returns 0 on success, -1 on OOM/unsupported. */
static int lazy_materialize_f32(qw36_engine *eng, qw36_lazy_w *lw) {
    if (!lw || lw->dtype == QW36_DTYPE_F32) return 0;
    size_t numel = (size_t)lw->cols * (lw->rows ? lw->rows : 1);
    if (lw->n_extra) numel *= (size_t)lw->n_extra;
    float *p = qw36__materialize_f32(lw->data, lw->dtype, numel);
    if (!p) return -1;
    if (!qw36__eng_own(eng, p)) { free(p); return -1; }
    lw->data  = p;
    lw->dtype = QW36_DTYPE_F32;
    return 0;
}

static int lazy_materialize_f16(qw36_engine *eng, qw36_lazy_w *lw) {
    if (!lw || lw->dtype == QW36_DTYPE_F16) return 0;
    size_t numel = (size_t)lw->cols * (lw->rows ? lw->rows : 1);
    if (lw->n_extra) numel *= (size_t)lw->n_extra;
    float *tmp = qw36__materialize_f32(lw->data, lw->dtype, numel);
    if (!tmp) return -1;
    uint16_t *p = (uint16_t *)malloc(numel * sizeof(uint16_t));
    if (!p) { free(tmp); return -1; }
    for (size_t i = 0; i < numel; i++) p[i] = qw36__f32_to_f16(tmp[i]);
    free(tmp);
    if (!qw36__eng_own(eng, p)) { free(p); return -1; }
    lw->data  = p;
    lw->dtype = QW36_DTYPE_F16;
    return 0;
}

static int lazy_affine32_repack_sanity(const qw36_lazy_w *orig,
                                       const qw36_lazy_w *repacked,
                                       float *max_abs_out)
{
    if (!orig || !repacked || !max_abs_out || !orig->rows || !orig->cols)
        return -1;
    float *a = (float *)malloc((size_t)orig->cols * sizeof(float));
    float *b = (float *)malloc((size_t)orig->cols * sizeof(float));
    if (!a || !b) {
        free(a); free(b);
        return -1;
    }
    const size_t total_rows =
        (size_t)orig->rows * (orig->n_extra ? (size_t)orig->n_extra : 1u);
    size_t probes[3] = {0, total_rows / 2, total_rows ? total_rows - 1 : 0};
    float max_abs = 0.0f;
    for (size_t pi = 0; pi < 3 && total_rows; pi++) {
        size_t row = probes[pi];
        if (qw36__dequant_row(orig, row, a) ||
            qw36__dequant_row(repacked, row, b)) {
            free(a); free(b);
            return -1;
        }
        for (size_t i = 0; i < (size_t)orig->cols; i++) {
            float d = fabsf(a[i] - b[i]);
            if (d > max_abs) max_abs = d;
        }
    }
    free(a); free(b);
    *max_abs_out = max_abs;
    return 0;
}

/* Runtime-only low-memory Metal path: convert GGUF Q4_K into an affine
 * per-32 layout that is compatible with MLX-style qmv kernels:
 *   [fp16 scale[8] | fp16 bias[8] | packed q4 groups[8][16]]
 * per 256-element block. */
static int lazy_repack_q4k_to_affine32(qw36_engine *eng, qw36_lazy_w *lw,
                                       float *max_abs_out)
{
    if (!lw || lw->dtype != QW36_DTYPE_Q4_K) return 0;
    if (!lw->cols || (lw->cols % 256u) != 0 || !lw->rows) return -1;
    size_t numel = (size_t)lw->cols * (size_t)lw->rows;
    if (lw->n_extra) {
        if (numel > SIZE_MAX / (size_t)lw->n_extra) return -1;
        numel *= (size_t)lw->n_extra;
    }
    size_t bytes = qw36__tensor_bytes(QW36_DTYPE_Q4K_AFFINE32, numel);
    if (!bytes) return -1;
    uint8_t *p = (uint8_t *)malloc(bytes);
    if (!p) return -1;
    if (qw36__repack_q4k_affine32(lw, p)) {
        free(p);
        return -1;
    }

    qw36_lazy_w repacked = *lw;
    repacked.data = p;
    repacked.dtype = QW36_DTYPE_Q4K_AFFINE32;
    repacked.gpu_buf = NULL;
    float max_abs = 0.0f;
    if (lazy_affine32_repack_sanity(lw, &repacked, &max_abs)) {
        free(p);
        return -1;
    }
    /* Original Q4_K computes scale/bias in fp32 from fp16 super-scales;
     * affine32 stores the derived scale/bias in fp16. The check is for
     * fp16-rounding equivalence, not bitwise fp32 identity. */
    if (max_abs > 0.25f) {
        free(p);
        return -1;
    }
    if (max_abs_out && max_abs > *max_abs_out) *max_abs_out = max_abs;
    if (!qw36__eng_own(eng, p)) { free(p); return -1; }
    lw->data = p;
    lw->dtype = QW36_DTYPE_Q4K_AFFINE32;
    lw->gpu_buf = NULL;
    return 0;
}

/* Runtime-only low-memory Metal path: convert GGUF Q5_K into a sibling
 * affine per-32 layout:
 *   [fp16 scale[8] | fp16 bias[8] | packed q5 groups[8][20]]
 * per 256-element block. */
static int lazy_repack_q5k_to_affine32(qw36_engine *eng, qw36_lazy_w *lw,
                                       float *max_abs_out)
{
    if (!lw || lw->dtype != QW36_DTYPE_Q5_K) return 0;
    if (!lw->cols || (lw->cols % 256u) != 0 || !lw->rows) return -1;
    size_t numel = (size_t)lw->cols * (size_t)lw->rows;
    if (lw->n_extra) {
        if (numel > SIZE_MAX / (size_t)lw->n_extra) return -1;
        numel *= (size_t)lw->n_extra;
    }
    size_t bytes = qw36__tensor_bytes(QW36_DTYPE_Q5K_AFFINE32, numel);
    if (!bytes) return -1;
    uint8_t *p = (uint8_t *)malloc(bytes);
    if (!p) return -1;
    if (qw36__repack_q5k_affine32(lw, p)) {
        free(p);
        return -1;
    }

    qw36_lazy_w repacked = *lw;
    repacked.data = p;
    repacked.dtype = QW36_DTYPE_Q5K_AFFINE32;
    repacked.gpu_buf = NULL;
    float max_abs = 0.0f;
    if (lazy_affine32_repack_sanity(lw, &repacked, &max_abs)) {
        free(p);
        return -1;
    }
    if (max_abs > 0.25f) {
        free(p);
        return -1;
    }
    if (max_abs_out && max_abs > *max_abs_out) *max_abs_out = max_abs;
    if (!qw36__eng_own(eng, p)) { free(p); return -1; }
    lw->data = p;
    lw->dtype = QW36_DTYPE_Q5K_AFFINE32;
    lw->gpu_buf = NULL;
    return 0;
}

/* Runtime-only low-memory Metal path for Q6_K. Q6_K is symmetric and has
 * 16-element scale groups, so repack to:
 *   [fp16 scale[16] | packed q6 groups[16][12]]
 * per 256-element block. */
static int lazy_repack_q6k_to_scale16(qw36_engine *eng, qw36_lazy_w *lw,
                                      float *max_abs_out)
{
    if (!lw || lw->dtype != QW36_DTYPE_Q6_K) return 0;
    if (!lw->cols || (lw->cols % 256u) != 0 || !lw->rows) return -1;
    size_t numel = (size_t)lw->cols * (size_t)lw->rows;
    if (lw->n_extra) {
        if (numel > SIZE_MAX / (size_t)lw->n_extra) return -1;
        numel *= (size_t)lw->n_extra;
    }
    size_t bytes = qw36__tensor_bytes(QW36_DTYPE_Q6K_SCALE16, numel);
    if (!bytes) return -1;
    uint8_t *p = (uint8_t *)malloc(bytes);
    if (!p) return -1;
    if (qw36__repack_q6k_scale16(lw, p)) {
        free(p);
        return -1;
    }

    qw36_lazy_w repacked = *lw;
    repacked.data = p;
    repacked.dtype = QW36_DTYPE_Q6K_SCALE16;
    repacked.gpu_buf = NULL;
    float max_abs = 0.0f;
    if (lazy_affine32_repack_sanity(lw, &repacked, &max_abs)) {
        free(p);
        return -1;
    }
    if (max_abs > 0.25f) {
        free(p);
        return -1;
    }
    if (max_abs_out && max_abs > *max_abs_out) *max_abs_out = max_abs;
    if (!qw36__eng_own(eng, p)) { free(p); return -1; }
    lw->data = p;
    lw->dtype = QW36_DTYPE_Q6K_SCALE16;
    lw->gpu_buf = NULL;
    return 0;
}

/* Metal dense-MLP fast path: concatenate gate_proj and up_proj into one
 * [2*inter, hidden] dense matrix after materialization and before upload.
 * L->up_proj == NULL is the internal signal that the backend should split
 * w_gate's output into gate/up halves. */
static int lazy_fuse_dense_gate_up(qw36_engine *eng, qw36_layer_weights *L) {
    if (!eng || !L || L->moe_router || !L->gate_proj || !L->up_proj ||
        !L->down_proj)
        return 0;
    qw36_lazy_w *gate = (qw36_lazy_w *)L->gate_proj;
    qw36_lazy_w *up   = (qw36_lazy_w *)L->up_proj;
    /* Allow dense (F32/F16) and the K-quant family. For quants we copy
     * raw block bytes — qw36__tensor_bytes knows the per-block layout, and
     * cols must be a multiple of 256 (Q*_K block size). The downstream
     * matmul kernels (qmv_fast / qmv_mlx) consume the fused row range
     * (2*rows × cols) the same way they consume any single-tensor row range. */
    int is_dense = (gate->dtype == QW36_DTYPE_F32 ||
                    gate->dtype == QW36_DTYPE_F16);
    int is_qk = (gate->dtype == QW36_DTYPE_Q4_K ||
                 gate->dtype == QW36_DTYPE_Q5_K ||
                 gate->dtype == QW36_DTYPE_Q6_K ||
                 gate->dtype == QW36_DTYPE_Q4K_AFFINE32 ||
                 gate->dtype == QW36_DTYPE_Q5K_AFFINE32 ||
                 gate->dtype == QW36_DTYPE_Q6K_SCALE16);
    if (gate->dtype != up->dtype || (!is_dense && !is_qk) ||
        gate->rows != up->rows || gate->cols != up->cols ||
        gate->n_extra || up->n_extra)
        return 0;
    if (is_qk && (gate->cols % 256u) != 0)
        return 0;
    if (gate->rows > SIZE_MAX || gate->cols > SIZE_MAX ||
        gate->rows > UINT64_MAX / 2)
        return -1;
    const size_t rows = (size_t)gate->rows;
    const size_t cols = (size_t)gate->cols;
    if (cols && rows > SIZE_MAX / cols) return -1;
    const size_t half_numel = rows * cols;
    const size_t half_bytes = qw36__tensor_bytes(gate->dtype, half_numel);
    if (!half_bytes || half_bytes > SIZE_MAX / 2) return -1;

    uint8_t *buf = (uint8_t *)malloc(2 * half_bytes);
    qw36_lazy_w *fused = (qw36_lazy_w *)calloc(1, sizeof(*fused));
    if (!buf || !fused) {
        free(buf);
        free(fused);
        return -1;
    }
    memcpy(buf, gate->data, half_bytes);
    memcpy(buf + half_bytes, up->data, half_bytes);
    *fused = *gate;
    fused->data = buf;
    fused->rows = gate->rows * 2;
    fused->gpu_buf = NULL;
    if (!qw36__eng_own(eng, buf)) {
        free(buf);
        free(fused);
        return -1;
    }
    if (!qw36__eng_own(eng, fused)) {
        free(fused);
        return -1;
    }
    L->gate_proj = fused;
    L->up_proj = NULL;
    return 0;
}

/* Metal vanilla-attention fast path: concatenate q_proj/k_proj/v_proj into one
 * dense [q_rows + k_rows + v_rows, hidden] fp16 matrix after materialization and
 * before upload. L->k_proj == NULL && L->v_proj == NULL is the internal signal
 * that L->q_proj contains the fused QKV matrix. Q-gated layers keep the Q half
 * layout intact: per-head [Q(head_dim) | gate(head_dim)] rows come first. */
static int lazy_fuse_vanilla_qkv(qw36_engine *eng, qw36_layer_weights *L,
                                 const qw36_config *c) {
    if (!eng || !L || !c || !L->q_proj || !L->k_proj || !L->v_proj)
        return 0;
    qw36_lazy_w *q = (qw36_lazy_w *)L->q_proj;
    qw36_lazy_w *k = (qw36_lazy_w *)L->k_proj;
    qw36_lazy_w *v = (qw36_lazy_w *)L->v_proj;
    if (q->dtype != QW36_DTYPE_F16 || k->dtype != QW36_DTYPE_F16 ||
        v->dtype != QW36_DTYPE_F16 ||
        q->cols != k->cols || q->cols != v->cols ||
        q->n_extra || k->n_extra || v->n_extra)
        return 0;

    const uint64_t q_dim =
        (uint64_t)c->num_attention_heads * c->head_dim;
    const uint64_t kv_dim =
        (uint64_t)c->num_key_value_heads * c->head_dim;
    const uint64_t q_rows = c->has_q_gate ? q_dim * 2u : q_dim;
    if (q->rows != q_rows || k->rows != kv_dim || v->rows != kv_dim)
        return 0;
    if (q_rows > UINT64_MAX - kv_dim ||
        q_rows + kv_dim > UINT64_MAX - kv_dim)
        return -1;

    const uint64_t rows64 = q_rows + kv_dim + kv_dim;
    const size_t cols = (size_t)q->cols;
    if (rows64 > SIZE_MAX) return -1;
    const size_t rows = (size_t)rows64;
    if (cols && rows > SIZE_MAX / cols) return -1;
    const size_t numel = rows * cols;
    if (numel > SIZE_MAX / sizeof(uint16_t)) return -1;
    const size_t q_bytes = (size_t)q_rows * cols * sizeof(uint16_t);
    const size_t kv_bytes = (size_t)kv_dim * cols * sizeof(uint16_t);

    uint8_t *buf = (uint8_t *)malloc(numel * sizeof(uint16_t));
    qw36_lazy_w *fused = (qw36_lazy_w *)calloc(1, sizeof(*fused));
    if (!buf || !fused) {
        free(buf);
        free(fused);
        return -1;
    }
    memcpy(buf, q->data, q_bytes);
    memcpy(buf + q_bytes, k->data, kv_bytes);
    memcpy(buf + q_bytes + kv_bytes, v->data, kv_bytes);
    *fused = *q;
    fused->data = buf;
    fused->rows = rows64;
    fused->gpu_buf = NULL;
    if (!qw36__eng_own(eng, buf)) {
        free(buf);
        free(fused);
        return -1;
    }
    if (!qw36__eng_own(eng, fused)) {
        free(fused);
        return -1;
    }
    L->q_proj = fused;
    L->k_proj = NULL;
    L->v_proj = NULL;
    return 0;
}

/* Metal DeltaNet fast path: concatenate the four projections that share the
 * same x_rms input into one fp16 [qkv | z | alpha | beta] matrix. The fused
 * signal is L->dn_gate == L->dn_alpha == L->dn_beta == NULL; downstream code
 * reads z/alpha/beta by element offsets from the dn_qkv matmul output. */
static int lazy_fuse_dn_qkvzab(qw36_engine *eng, qw36_layer_weights *L,
                               const qw36_config *c) {
    if (!eng || !L || !c || !L->dn_qkv || !L->dn_gate ||
        !L->dn_alpha || !L->dn_beta)
        return 0;
    qw36_lazy_w *qkv = (qw36_lazy_w *)L->dn_qkv;
    qw36_lazy_w *z   = (qw36_lazy_w *)L->dn_gate;
    qw36_lazy_w *a   = (qw36_lazy_w *)L->dn_alpha;
    qw36_lazy_w *b   = (qw36_lazy_w *)L->dn_beta;
    if (qkv->dtype != QW36_DTYPE_F16 || z->dtype != QW36_DTYPE_F16 ||
        a->dtype != QW36_DTYPE_F16 || b->dtype != QW36_DTYPE_F16 ||
        qkv->cols != z->cols || qkv->cols != a->cols || qkv->cols != b->cols ||
        qkv->n_extra || z->n_extra || a->n_extra || b->n_extra)
        return 0;

    const uint64_t qkv_rows =
        (uint64_t)c->dn_num_key_heads * c->dn_key_head_dim * 2u +
        (uint64_t)c->dn_num_value_heads * c->dn_value_head_dim;
    const uint64_t z_rows =
        (uint64_t)c->dn_num_value_heads * c->dn_value_head_dim;
    const uint64_t ab_rows = c->dn_num_value_heads;
    if (!qkv_rows || !z_rows || !ab_rows ||
        qkv->rows != qkv_rows || z->rows != z_rows ||
        a->rows != ab_rows || b->rows != ab_rows)
        return 0;
    if (qkv_rows > UINT64_MAX - z_rows ||
        qkv_rows + z_rows > UINT64_MAX - ab_rows ||
        qkv_rows + z_rows + ab_rows > UINT64_MAX - ab_rows)
        return -1;

    const uint64_t rows64 = qkv_rows + z_rows + ab_rows + ab_rows;
    const size_t cols = (size_t)qkv->cols;
    if (rows64 > SIZE_MAX) return -1;
    const size_t rows = (size_t)rows64;
    if (cols && rows > SIZE_MAX / cols) return -1;
    const size_t numel = rows * cols;
    if (numel > SIZE_MAX / sizeof(uint16_t)) return -1;
    const size_t qkv_bytes = (size_t)qkv_rows * cols * sizeof(uint16_t);
    const size_t z_bytes = (size_t)z_rows * cols * sizeof(uint16_t);
    const size_t ab_bytes = (size_t)ab_rows * cols * sizeof(uint16_t);

    uint8_t *buf = (uint8_t *)malloc(numel * sizeof(uint16_t));
    qw36_lazy_w *fused = (qw36_lazy_w *)calloc(1, sizeof(*fused));
    if (!buf || !fused) {
        free(buf);
        free(fused);
        return -1;
    }
    uint8_t *dst = buf;
    memcpy(dst, qkv->data, qkv_bytes); dst += qkv_bytes;
    memcpy(dst, z->data, z_bytes);     dst += z_bytes;
    memcpy(dst, a->data, ab_bytes);    dst += ab_bytes;
    memcpy(dst, b->data, ab_bytes);
    *fused = *qkv;
    fused->data = buf;
    fused->rows = rows64;
    fused->gpu_buf = NULL;
    if (!qw36__eng_own(eng, buf)) {
        free(buf);
        free(fused);
        return -1;
    }
    if (!qw36__eng_own(eng, fused)) {
        free(fused);
        return -1;
    }
    L->dn_qkv = fused;
    L->dn_gate = NULL;
    L->dn_alpha = NULL;
    L->dn_beta = NULL;
    return 0;
}

/* Lazy bind for big tensors: keep a pointer into mmap + shape + dtype,
 * to be dequantized block-by-block during matmul. */
static qw36_lazy_w *bind_tensor_lazy(qw36_engine *eng, const char *name) {
    qw36_gguf_tensor t;
    if (qw36_gguf_get_tensor(eng->gguf, name, &t)) return NULL;
    qw36_lazy_w *lw = (qw36_lazy_w *)calloc(1, sizeof(*lw));
    if (!lw) return NULL;
    lw->data      = t.data;
    lw->dtype     = t.dtype;
    lw->ggml_type = t.ggml_type;
    lw->cols      = t.n_dims >= 1 ? t.dims[0] : 0;
    lw->rows      = t.n_dims >= 2 ? t.dims[1] : 0;
    lw->n_extra   = t.n_dims >= 3 ? t.dims[2] : 0;
    if (!qw36__eng_own(eng, lw)) { free(lw); return NULL; }
    return lw;
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
    /* Required keys. MoE-only variants may not store feed_forward_length;
     * fall back to expert_feed_forward_length later. */
    int missing = 0;
    if (eng_get_u32(eng, "embedding_length",        &c->hidden_size))       missing |= 1;
    if (eng_get_u32(eng, "block_count",             &c->num_hidden_layers)) missing |= 2;
    if (eng_get_u32(eng, "attention.head_count",    &c->num_attention_heads)) missing |= 4;
    if (eng_get_u32(eng, "attention.head_count_kv", &c->num_key_value_heads)) missing |= 8;
    if (eng_get_u32(eng, "context_length",          &c->max_position_embeddings)) missing |= 16;
    /* Optional / fall-back: dense MLP size. MoE-only models omit this. */
    eng_get_u32(eng, "feed_forward_length", &c->intermediate_size);
    if (missing) {
        if (err && err_cap) snprintf(err, err_cap,
            "missing required %s.* config keys (mask=0x%x)", eng->arch, missing);
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
        uint32_t rotary_dim = 0;
        if (eng_get_f32(eng, "rope.partial_rotary_factor", &prf) == 0 && prf > 0) {
            c->partial_rotary_factor = prf;
        } else if (eng_get_u32(eng, "rope.dimension_count", &rotary_dim) == 0 &&
                   rotary_dim > 0 && c->head_dim > 0) {
            c->partial_rotary_factor = (float)rotary_dim / (float)c->head_dim;
        }
    }
    /* Qwen3.5/3.6 mRoPE per-axis pair counts. Stored as
     * <arch>.rope.dimension_sections — for the 0.8B model this is
     * [11, 11, 10, 0] giving 32 pairs over 4 axes. Pairs after the sum
     * (or in section 0 for text decode) use seq_pos; other sections use
     * axis-1/2/3 positions which are 0 in text mode and so leave their
     * pairs unrotated. */
    {
        char key[128];
        snprintf(key, sizeof(key), "%s.rope.dimension_sections", eng->arch);
        int n = qw36_gguf_get_u32_array(eng->gguf, key,
                                        c->rope_sections, 4);
        /* Agent-infer's MLX text-decode path calls fast::rope with
         * rotary_dim from GGUF/config (qwen35.rope.dimension_count = 64
         * for the 0.8B model), traditional=false. We use plain NEOX by
         * default; set QW36_USE_MROPE_SECTIONS=1 to re-enable per-axis
         * chopping for experiments with multimodal positions. */
        const char *mrope_env = getenv("QW36_USE_MROPE_SECTIONS");
        if (mrope_env && atoi(mrope_env) != 0) {
            c->rope_n_sections = (n > 0) ? (uint32_t)n : 0;
        } else {
            c->rope_n_sections = 0;
        }
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

    /* Gated DeltaNet config. Prefer the explicit GGUF SSM metadata:
     *   group_count = key heads
     *   state_size  = key/value head dim
     *   inner_size  = value_heads * value_head_dim
     * Tensor-shape fallback keeps older experimental files loadable. */
    {
        qw36_gguf_tensor t;
        uint32_t inner_size = 0;
        eng_get_u32(eng, "ssm.group_count", &c->dn_num_key_heads);
        eng_get_u32(eng, "ssm.state_size",  &c->dn_key_head_dim);
        eng_get_u32(eng, "ssm.inner_size",  &inner_size);
        if (eng_get_u32(eng, "ssm.conv_kernel", &c->dn_conv_kernel_size)) {
            if (qw36_gguf_get_tensor(eng->gguf, "blk.0.ssm_conv1d.weight", &t) == 0) {
                /* GGUF stores dims innermost-first. ssm_conv1d shape is
                 * [k, channels] in numpy -> dims[0]=k, dims[1]=channels. */
                c->dn_conv_kernel_size = (uint32_t)t.dims[0];
            }
        }
        if (inner_size && c->dn_key_head_dim) {
            c->dn_value_head_dim = c->dn_key_head_dim;
            c->dn_num_value_heads = inner_size / c->dn_value_head_dim;
        }
        if (!c->dn_num_value_heads &&
            qw36_gguf_get_tensor(eng->gguf, "blk.0.ssm_a", &t) == 0)
            c->dn_num_value_heads = (uint32_t)t.dims[0];
        if (!c->dn_value_head_dim &&
            qw36_gguf_get_tensor(eng->gguf, "blk.0.ssm_norm.weight", &t) == 0)
            c->dn_value_head_dim = (uint32_t)t.dims[0];
        if (!c->dn_key_head_dim) c->dn_key_head_dim = c->dn_value_head_dim;
        if (!c->dn_num_key_heads &&
            qw36_gguf_get_tensor(eng->gguf, "blk.0.attn_qkv.weight", &t) == 0 &&
            t.n_dims >= 2 && c->dn_key_head_dim &&
            c->dn_num_value_heads && c->dn_value_head_dim) {
            uint64_t v_dim = (uint64_t)c->dn_num_value_heads * c->dn_value_head_dim;
            if (t.dims[1] > v_dim) {
                uint64_t qk_dim = (t.dims[1] - v_dim) / 2;
                if (qk_dim && qk_dim % c->dn_key_head_dim == 0)
                    c->dn_num_key_heads = (uint32_t)(qk_dim / c->dn_key_head_dim);
            }
        }
        if (qw36_gguf_get_tensor(eng->gguf, "blk.0.ssm_conv1d.weight", &t) == 0) {
            uint64_t qkv_dim = (uint64_t)c->dn_num_key_heads * c->dn_key_head_dim * 2
                             + (uint64_t)c->dn_num_value_heads * c->dn_value_head_dim;
            if (qkv_dim && t.n_dims >= 2 && t.dims[1] != qkv_dim) {
                fprintf(stderr,
                    "qw36: warning: DeltaNet qkv channels mismatch: "
                    "config=%llu tensor=%llu\n",
                    (unsigned long long)qkv_dim,
                    (unsigned long long)t.dims[1]);
            }
        }
    }

    /* Qwen3.5 vanilla attention has a *Q-gate*: q_proj output is
     *   n_heads * head_dim * 2  =  Q concat with a per-head gate.
     * (See agent-infer mlx_qwen35_model.cpp:789 has_qk_gate branch.)
     * That means the GGUF tensor `blk.X.attn_q.weight` has dim_1 =
     * n_heads * head_dim * 2, not n_heads * head_dim. The previous
     * 'override n_heads to dim_1/head_dim' fix mis-counted heads by 2×
     * for Qwen3.5 — keep the metadata head_count and detect the gate
     * separately. */
    c->has_q_gate = 0;
    {
        char nm[128];
        qw36_gguf_tensor t;
        for (uint32_t l = 0; l < c->num_hidden_layers; l++) {
            snprintf(nm, sizeof(nm), "blk.%u.attn_q.weight", l);
            if (qw36_gguf_get_tensor(eng->gguf, nm, &t) == 0) {
                uint64_t out_dim = t.dims[1];
                if (out_dim == (uint64_t)c->num_attention_heads * c->head_dim * 2) {
                    c->has_q_gate = 1;
                    fprintf(stderr,
                        "qw36: detected Q-gate (q_proj out = %llu = "
                        "n_heads %u * head_dim %u * 2)\n",
                        (unsigned long long)out_dim,
                        c->num_attention_heads, c->head_dim);
                } else if (out_dim == (uint64_t)c->num_attention_heads * c->head_dim) {
                    /* Standard Qwen3 vanilla: no gate. */
                } else {
                    /* Metadata mismatch — try to recover. */
                    uint32_t real_q = (uint32_t)(out_dim / c->head_dim);
                    if (real_q && real_q != c->num_attention_heads) {
                        fprintf(stderr,
                            "qw36: overriding num_attention_heads %u -> %u "
                            "(from blk.%u.attn_q shape; no gate assumed)\n",
                            c->num_attention_heads, real_q, l);
                        c->num_attention_heads = real_q;
                    }
                }
                /* K projection (no gate, just GQA factor). */
                snprintf(nm, sizeof(nm), "blk.%u.attn_k.weight", l);
                if (qw36_gguf_get_tensor(eng->gguf, nm, &t) == 0) {
                    uint32_t real_kv = (uint32_t)(t.dims[1] / c->head_dim);
                    if (real_kv && real_kv != c->num_key_value_heads) {
                        fprintf(stderr,
                            "qw36: overriding num_key_value_heads %u -> %u\n",
                            c->num_key_value_heads, real_kv);
                        c->num_key_value_heads = real_kv;
                    }
                }
                break;
            }
        }
    }

    /* Bind global tensors. Small (norm) → fp32; big (embed/lm_head) → lazy. */
    qw36_weights *w = &eng->weights;
    w->dtype = QW36_DTYPE_F32;
    w->embed_tokens = bind_tensor_lazy(eng, "token_embd.weight");
    w->final_norm   = bind_tensor_f32(eng, "output_norm.weight");
    w->lm_head      = bind_tensor_lazy(eng, "output.weight");
    /* Many Qwen3 / Qwen3.5 / Qwen3.6 checkpoints omit output.weight and
     * tie it to the input embedding without setting tie_word_embeddings
     * explicitly. Fall back unconditionally — never leave lm_head NULL. */
    if (!w->lm_head) {
        w->lm_head = w->embed_tokens;
        c->tie_word_embeddings = 1;
    }
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
        #define BIND_NORM(field, fmt) do { \
            snprintf(name, sizeof(name), fmt, l);             \
            L->field = bind_tensor_f32_opt(eng, name);        \
        } while (0)
        #define BIND_W(field, fmt) do { \
            snprintf(name, sizeof(name), fmt, l);             \
            L->field = bind_tensor_lazy(eng, name);           \
        } while (0)

        /* Norms are small — keep them fp32. */
        BIND_NORM(input_layernorm,     "blk.%u.attn_norm.weight");
        BIND_NORM(q_norm,              "blk.%u.attn_q_norm.weight");
        BIND_NORM(k_norm,              "blk.%u.attn_k_norm.weight");
        /* Some Qwen3.5/3.6 checkpoints name the post-attention norm
         * "post_attention_norm" (with the underscore expanded) instead of
         * "ffn_norm". Try both. */
        BIND_NORM(post_attn_layernorm, "blk.%u.ffn_norm.weight");
        if (!L->post_attn_layernorm)
            BIND_NORM(post_attn_layernorm, "blk.%u.post_attention_norm.weight");

        /* Vanilla attention projections — lazy (big). */
        BIND_W(q_proj,    "blk.%u.attn_q.weight");
        BIND_W(k_proj,    "blk.%u.attn_k.weight");
        BIND_W(v_proj,    "blk.%u.attn_v.weight");
        BIND_W(o_proj,    "blk.%u.attn_output.weight");

        /* Gated DeltaNet projections. attn_qkv / attn_gate are big (lazy);
         * the rest are small enough to materialize. conv1d is depthwise
         * (kernel × channels = a few thousand floats), keep as fp32. */
        BIND_W(dn_qkv,           "blk.%u.attn_qkv.weight");
        BIND_W(dn_gate,          "blk.%u.attn_gate.weight");
        BIND_W(dn_alpha,         "blk.%u.ssm_alpha.weight");
        BIND_W(dn_beta,          "blk.%u.ssm_beta.weight");
        BIND_NORM(dn_conv1d,     "blk.%u.ssm_conv1d.weight");
        BIND_NORM(dn_dt_bias,    "blk.%u.ssm_dt.bias");
        BIND_NORM(dn_a_log,      "blk.%u.ssm_a");
        if (L->dn_a_log && c->dn_num_value_heads) {
            float *a = (float *)L->dn_a_log;
            for (uint32_t i = 0; i < c->dn_num_value_heads; i++) {
                float v = fabsf(a[i]);
                if (v < 1.0e-10f) v = 1.0e-10f;
                a[i] = logf(v);
            }
        }
        BIND_NORM(dn_norm,       "blk.%u.ssm_norm.weight");
        BIND_W(dn_out,           "blk.%u.ssm_out.weight");

        /* Dense MLP — lazy. */
        BIND_W(gate_proj, "blk.%u.ffn_gate.weight");
        BIND_W(up_proj,   "blk.%u.ffn_up.weight");
        BIND_W(down_proj, "blk.%u.ffn_down.weight");

        /* MoE — lazy. The expert stacks are 3D [n_experts, hidden, inter]
         * and qw36_lazy_w records that via rows/cols/n_extra. */
        BIND_W(moe_router,        "blk.%u.ffn_gate_inp.weight");
        BIND_W(moe_expert_gate,   "blk.%u.ffn_gate_exps.weight");
        BIND_W(moe_expert_up,     "blk.%u.ffn_up_exps.weight");
        BIND_W(moe_expert_down,   "blk.%u.ffn_down_exps.weight");
        BIND_W(moe_shared_gate,   "blk.%u.ffn_gate_shexp.weight");
        BIND_W(moe_shared_up,     "blk.%u.ffn_up_shexp.weight");
        BIND_W(moe_shared_down,   "blk.%u.ffn_down_shexp.weight");
        #undef BIND_NORM
        #undef BIND_W
    }

    /* When a GPU backend is provided, eagerly materialize every lazy_w
     * tensor to fp32 in host memory and (on Apple Silicon / unified memory
     * backends) ask the backend to wrap that buffer as a device view.
     *
     * Memory budget: a 0.8B model materializes to ~3 GB fp32, fits easily.
     * A 35B-A3B materializes to ~140 GB — caller must use the CPU build
     * for that until streaming dequant is added. */
    if (backend) {
        eng->ctx = backend->init(err, err_cap);
        if (!eng->ctx) { qw36_engine_close(eng); return NULL; }
        /* Opt-in fp16 weight materialization on Metal — nearly doubles
         * decode throughput (55 → 81 tok/s) but introduces ~1e-3 drift
         * that can flip the argmax later in a greedy run. Default off so
         * tests/precision_cpu_vs_metal.sh stays bit-equal; set
         * QW36_METAL_FP16_WEIGHTS=1 for the perf path. */
        const char *fp16_env = getenv("QW36_METAL_FP16_WEIGHTS");
        const int fp16_lazy_weights =
            backend->name && strcmp(backend->name, "metal") == 0 &&
            fp16_env && atoi(fp16_env) != 0;
        /* QW36_METAL_QUANT_GPU=1 keeps Q4_K/Q5_K/Q6_K/Q8_0 weights in their
         * packed layout on the GPU and dispatches a native dequant+gemv
         * kernel per matmul. Skips the fp16 materialise (saves RAM) and the
         * f32↔f16 round-trip per call (lower per-op overhead, higher
         * effective bandwidth utilisation). Overrides fp16_lazy_weights. */
        const char *qgpu_env = getenv("QW36_METAL_QUANT_GPU");
        const int quant_gpu_weights =
            backend->name && strcmp(backend->name, "metal") == 0 &&
            qgpu_env && atoi(qgpu_env) != 0;
        const char *qk_repack_env = getenv("QW36_METAL_QK_REPACK");
        const char *qk_affine32_env = getenv("QW36_METAL_QK_AFFINE32");
        const int qk_repack_weights =
            quant_gpu_weights &&
            ((qk_repack_env && atoi(qk_repack_env) != 0) ||
             (qk_affine32_env && atoi(qk_affine32_env) != 0));
        /* QW36_METAL_FAST=1 turns on the full affine-repack + lm_head quant
         * path under quant-GPU mode. Individual flags still win when set
         * explicitly to 0 so users can isolate one component for debugging. */
        const char *fast_env = getenv("QW36_METAL_FAST");
        const int fast_path =
            quant_gpu_weights && fast_env && atoi(fast_env) != 0;
        const char *q4k_affine32_env = getenv("QW36_METAL_Q4K_AFFINE32");
        const int q4k_affine32_weights =
            qk_repack_weights ||
            (quant_gpu_weights &&
             (q4k_affine32_env ? atoi(q4k_affine32_env) != 0 : fast_path));
        const char *q5k_affine32_env = getenv("QW36_METAL_Q5K_AFFINE32");
        const int q5k_affine32_weights =
            qk_repack_weights ||
            (quant_gpu_weights &&
             (q5k_affine32_env ? atoi(q5k_affine32_env) != 0 : fast_path));
        const char *q6k_scale16_env = getenv("QW36_METAL_Q6K_SCALE16");
        const int q6k_scale16_weights =
            quant_gpu_weights &&
            (q6k_scale16_env ? atoi(q6k_scale16_env) != 0 : fast_path);
        const char *quant_lm_head_env =
            getenv("QW36_METAL_QUANT_GPU_LM_HEAD");
        const int quant_gpu_lm_head =
            quant_gpu_weights &&
            (quant_lm_head_env ? atoi(quant_lm_head_env) != 0 : fast_path);
        const int fuse_dense_gate_up =
            backend->name && strcmp(backend->name, "metal") == 0;
        const char *fuse_qkv_env = getenv("QW36_METAL_FUSE_QKV");
        const int fuse_vanilla_qkv =
            backend->name && strcmp(backend->name, "metal") == 0 &&
            fp16_lazy_weights && !quant_gpu_weights &&
            (!fuse_qkv_env || atoi(fuse_qkv_env) != 0);
        const char *fuse_dn_env = getenv("QW36_METAL_FUSE_DN_QKVZAB");
        const int fuse_dn_qkvzab =
            backend->name && strcmp(backend->name, "metal") == 0 &&
            fp16_lazy_weights && !quant_gpu_weights &&
            (!fuse_dn_env || atoi(fuse_dn_env) != 0);

        if (quant_gpu_lm_head && w->lm_head == w->embed_tokens) {
            const qw36_lazy_w *src = (const qw36_lazy_w *)w->embed_tokens;
            if (src && (src->dtype == QW36_DTYPE_Q5_K ||
                        src->dtype == QW36_DTYPE_Q6_K)) {
                qw36_lazy_w *lm_head =
                    (qw36_lazy_w *)calloc(1, sizeof(*lm_head));
                if (!lm_head) {
                    if (err && err_cap) snprintf(err, err_cap,
                        "%s: tied lm_head lazy clone failed", backend->name);
                    qw36_engine_close(eng); return NULL;
                }
                *lm_head = *src;
                lm_head->gpu_buf = NULL;
                if (!qw36__eng_own(eng, lm_head)) {
                    free(lm_head);
                    if (err && err_cap) snprintf(err, err_cap,
                        "%s: tied lm_head lazy clone ownership failed",
                        backend->name);
                    qw36_engine_close(eng); return NULL;
                }
                w->lm_head = lm_head;
            }
        }

        qw36_lazy_w *lws[] = {
            (qw36_lazy_w *)w->embed_tokens,
            (qw36_lazy_w *)w->lm_head,
            NULL
        };
        for (size_t i = 0; lws[i]; i++) {
            if (quant_gpu_lm_head && lws[i] == (qw36_lazy_w *)w->lm_head &&
                w->lm_head != w->embed_tokens &&
                (lws[i]->dtype == QW36_DTYPE_Q5_K ||
                 lws[i]->dtype == QW36_DTYPE_Q6_K))
                continue;
            /* Embed stays materialised even in quant-GPU mode for the
             * embedding_lookup fast path. lm_head does too unless the
             * explicit QW36_METAL_QUANT_GPU_LM_HEAD experiment split a tied
             * embedding descriptor above. */
            const int want_f16 = fp16_lazy_weights || quant_gpu_weights;
            int mrc = want_f16
                ? lazy_materialize_f16(eng, lws[i])
                : lazy_materialize_f32(eng, lws[i]);
            if (mrc) {
                if (err && err_cap) snprintf(err, err_cap,
                    "%s: backend materialize failed (unsupported dtype %d)",
                    backend->name, lws[i]->ggml_type);
                qw36_engine_close(eng); return NULL;
            }
        }
        /* lm_head usually aliases embed_tokens — avoid double-binding. */
        if (w->lm_head == w->embed_tokens) ((qw36_lazy_w *)w->lm_head)->gpu_buf =
            ((qw36_lazy_w *)w->embed_tokens)->gpu_buf;

        #define MAT_AS(field, want_f16_) do { \
            qw36_lazy_w *lw = (qw36_lazy_w *)L->field; \
            int skip_mat = lw && quant_gpu_weights && \
                qw36__dtype_is_native_gpu_quant(lw->dtype); \
            int mrc = (lw && !skip_mat) ? ((want_f16_) \
                ? lazy_materialize_f16(eng, lw) \
                : lazy_materialize_f32(eng, lw)) : 0; \
            if (mrc) { \
                if (err && err_cap) snprintf(err, err_cap, \
                    "%s: backend materialize failed at blk.%u." #field, \
                    backend->name, l); \
                qw36_engine_close(eng); return NULL; \
            } \
        } while (0)
        for (uint32_t l = 0; l < c->num_hidden_layers; l++) {
            qw36_layer_weights *L = &eng->weights.layers[l];
            MAT_AS(q_proj, fp16_lazy_weights);
            MAT_AS(k_proj, fp16_lazy_weights);
            MAT_AS(v_proj, fp16_lazy_weights);
            MAT_AS(o_proj, fp16_lazy_weights);
            MAT_AS(dn_qkv, fp16_lazy_weights);
            MAT_AS(dn_gate, fp16_lazy_weights);
            MAT_AS(dn_alpha, fuse_dn_qkvzab);
            MAT_AS(dn_beta, fuse_dn_qkvzab);
            MAT_AS(dn_out, fp16_lazy_weights);
            MAT_AS(gate_proj, fp16_lazy_weights);
            MAT_AS(up_proj, fp16_lazy_weights);
            MAT_AS(down_proj, fp16_lazy_weights);
            MAT_AS(moe_router, 0);
            MAT_AS(moe_expert_gate, fp16_lazy_weights);
            MAT_AS(moe_expert_up, fp16_lazy_weights);
            MAT_AS(moe_expert_down, fp16_lazy_weights);
            MAT_AS(moe_shared_gate, fp16_lazy_weights);
            MAT_AS(moe_shared_up, fp16_lazy_weights);
            MAT_AS(moe_shared_down, fp16_lazy_weights);
        }
        #undef MAT_AS

        if (q4k_affine32_weights || q5k_affine32_weights ||
            q6k_scale16_weights || quant_gpu_lm_head) {
            size_t repacked_q4_n = 0;
            size_t repacked_q5_n = 0;
            size_t repacked_q6_n = 0;
            float repack_max_abs = 0.0f;
            #define REPACK_Q4K(field) do { \
                qw36_lazy_w *lw = (qw36_lazy_w *)L->field; \
                if (q4k_affine32_weights && lw && lw->dtype == QW36_DTYPE_Q4_K) { \
                    if (lazy_repack_q4k_to_affine32(eng, lw, &repack_max_abs)) { \
                        if (err && err_cap) snprintf(err, err_cap, \
                            "%s: Q4_K affine32 repack failed at blk.%u." #field, \
                            backend->name, l); \
                        qw36_engine_close(eng); return NULL; \
                    } \
                    repacked_q4_n++; \
                } \
            } while (0)
            #define REPACK_Q5K(field) do { \
                qw36_lazy_w *lw = (qw36_lazy_w *)L->field; \
                if (q5k_affine32_weights && lw && lw->dtype == QW36_DTYPE_Q5_K) { \
                    if (lazy_repack_q5k_to_affine32(eng, lw, &repack_max_abs)) { \
                        if (err && err_cap) snprintf(err, err_cap, \
                            "%s: Q5_K affine32 repack failed at blk.%u." #field, \
                            backend->name, l); \
                        qw36_engine_close(eng); return NULL; \
                    } \
                    repacked_q5_n++; \
                } \
            } while (0)
            #define REPACK_Q6K(field) do { \
                qw36_lazy_w *lw = (qw36_lazy_w *)L->field; \
                if (q6k_scale16_weights && lw && lw->dtype == QW36_DTYPE_Q6_K) { \
                    if (lazy_repack_q6k_to_scale16(eng, lw, &repack_max_abs)) { \
                        if (err && err_cap) snprintf(err, err_cap, \
                            "%s: Q6_K scale16 repack failed at blk.%u." #field, \
                            backend->name, l); \
                        qw36_engine_close(eng); return NULL; \
                    } \
                    repacked_q6_n++; \
                } \
            } while (0)
            if (quant_gpu_lm_head && w->lm_head != w->embed_tokens) {
                qw36_lazy_w *lw = (qw36_lazy_w *)w->lm_head;
                if (lw && lw->dtype == QW36_DTYPE_Q5_K) {
                    if (lazy_repack_q5k_to_affine32(eng, lw,
                                                    &repack_max_abs)) {
                        if (err && err_cap) snprintf(err, err_cap,
                            "%s: Q5_K affine32 repack failed at lm_head",
                            backend->name);
                        qw36_engine_close(eng); return NULL;
                    }
                    repacked_q5_n++;
                } else if (lw && lw->dtype == QW36_DTYPE_Q6_K) {
                    if (lazy_repack_q6k_to_scale16(eng, lw,
                                                   &repack_max_abs)) {
                        if (err && err_cap) snprintf(err, err_cap,
                            "%s: Q6_K scale16 repack failed at lm_head",
                            backend->name);
                        qw36_engine_close(eng); return NULL;
                    }
                    repacked_q6_n++;
                }
            }
            for (uint32_t l = 0; l < c->num_hidden_layers; l++) {
                qw36_layer_weights *L = &eng->weights.layers[l];
                REPACK_Q4K(q_proj); REPACK_Q4K(k_proj); REPACK_Q4K(v_proj);
                REPACK_Q4K(o_proj);
                REPACK_Q4K(dn_qkv); REPACK_Q4K(dn_gate);
                REPACK_Q4K(dn_alpha); REPACK_Q4K(dn_beta);
                REPACK_Q4K(dn_out);
                REPACK_Q4K(gate_proj); REPACK_Q4K(up_proj);
                REPACK_Q4K(down_proj);
                REPACK_Q4K(moe_router);
                REPACK_Q4K(moe_expert_gate); REPACK_Q4K(moe_expert_up);
                REPACK_Q4K(moe_expert_down);
                REPACK_Q4K(moe_shared_gate); REPACK_Q4K(moe_shared_up);
                REPACK_Q4K(moe_shared_down);
                REPACK_Q5K(q_proj); REPACK_Q5K(k_proj); REPACK_Q5K(v_proj);
                REPACK_Q5K(o_proj);
                REPACK_Q5K(dn_qkv); REPACK_Q5K(dn_gate);
                REPACK_Q5K(dn_alpha); REPACK_Q5K(dn_beta);
                REPACK_Q5K(dn_out);
                REPACK_Q5K(gate_proj); REPACK_Q5K(up_proj);
                REPACK_Q5K(down_proj);
                REPACK_Q5K(moe_router);
                REPACK_Q5K(moe_expert_gate); REPACK_Q5K(moe_expert_up);
                REPACK_Q5K(moe_expert_down);
                REPACK_Q5K(moe_shared_gate); REPACK_Q5K(moe_shared_up);
                REPACK_Q5K(moe_shared_down);
                REPACK_Q6K(q_proj); REPACK_Q6K(k_proj); REPACK_Q6K(v_proj);
                REPACK_Q6K(o_proj);
                REPACK_Q6K(dn_qkv); REPACK_Q6K(dn_gate);
                REPACK_Q6K(dn_alpha); REPACK_Q6K(dn_beta);
                REPACK_Q6K(dn_out);
                REPACK_Q6K(gate_proj); REPACK_Q6K(up_proj);
                REPACK_Q6K(down_proj);
                REPACK_Q6K(moe_router);
                REPACK_Q6K(moe_expert_gate); REPACK_Q6K(moe_expert_up);
                REPACK_Q6K(moe_expert_down);
                REPACK_Q6K(moe_shared_gate); REPACK_Q6K(moe_shared_up);
                REPACK_Q6K(moe_shared_down);
            }
            #undef REPACK_Q6K
            #undef REPACK_Q5K
            #undef REPACK_Q4K
            if (repacked_q4_n || repacked_q5_n || repacked_q6_n) {
                fprintf(stderr,
                    "qw36: qk repacked Q4_K=%zu Q5_K=%zu Q6_K=%zu tensors "
                    "(sanity max_abs %.6g)\n",
                    repacked_q4_n, repacked_q5_n, repacked_q6_n,
                    repack_max_abs);
            }
        }

        if (fuse_dense_gate_up) {
            for (uint32_t l = 0; l < c->num_hidden_layers; l++) {
                qw36_layer_weights *L = &eng->weights.layers[l];
                if (lazy_fuse_dense_gate_up(eng, L)) {
                    if (err && err_cap) snprintf(err, err_cap,
                        "%s: backend gate_up fuse failed at blk.%u",
                        backend->name, l);
                    qw36_engine_close(eng); return NULL;
                }
            }
        }
        if (fuse_vanilla_qkv) {
            for (uint32_t l = 0; l < c->num_hidden_layers; l++) {
                qw36_layer_weights *L = &eng->weights.layers[l];
                if (lazy_fuse_vanilla_qkv(eng, L, c)) {
                    if (err && err_cap) snprintf(err, err_cap,
                        "%s: backend qkv fuse failed at blk.%u",
                        backend->name, l);
                    qw36_engine_close(eng); return NULL;
                }
            }
        }
        if (fuse_dn_qkvzab) {
            for (uint32_t l = 0; l < c->num_hidden_layers; l++) {
                qw36_layer_weights *L = &eng->weights.layers[l];
                if (lazy_fuse_dn_qkvzab(eng, L, c)) {
                    if (err && err_cap) snprintf(err, err_cap,
                        "%s: backend DN qkvzab fuse failed at blk.%u",
                        backend->name, l);
                    qw36_engine_close(eng); return NULL;
                }
            }
        }

        /* Now wrap every materialized buffer as a device-visible view.
         * For Apple Metal this is zero-copy via MTLBuffer NoCopy; for
         * CUDA/HIP this does an actual device copy. */
        #define UPLOAD(field) do { \
            qw36_lazy_w *lw = (qw36_lazy_w *)L->field; \
            if (lw && lw->data && !lw->gpu_buf) { \
                size_t numel = (size_t)lw->cols * (lw->rows ? lw->rows : 1); \
                if (lw->n_extra) numel *= (size_t)lw->n_extra; \
                size_t bytes = qw36__tensor_bytes(lw->dtype, numel); \
                if (!bytes) { qw36_engine_close(eng); return NULL; } \
                lw->gpu_buf = backend->upload(eng->ctx, lw->data, \
                    bytes, lw->dtype); \
            } \
        } while (0)
        for (uint32_t l = 0; l < c->num_hidden_layers; l++) {
            qw36_layer_weights *L = &eng->weights.layers[l];
            UPLOAD(q_proj); UPLOAD(k_proj); UPLOAD(v_proj); UPLOAD(o_proj);
            UPLOAD(dn_qkv); UPLOAD(dn_gate); UPLOAD(dn_alpha);
            UPLOAD(dn_beta); UPLOAD(dn_out);
            UPLOAD(gate_proj); UPLOAD(up_proj); UPLOAD(down_proj);
            UPLOAD(moe_router);
            UPLOAD(moe_expert_gate); UPLOAD(moe_expert_up); UPLOAD(moe_expert_down);
            UPLOAD(moe_shared_gate); UPLOAD(moe_shared_up); UPLOAD(moe_shared_down);
        }
        #undef UPLOAD
        /* Globals. */
        if (w->embed_tokens) {
            qw36_lazy_w *lw = (qw36_lazy_w *)w->embed_tokens;
            size_t numel = (size_t)lw->cols * lw->rows;
            size_t bytes = qw36__tensor_bytes(lw->dtype, numel);
            if (!bytes) { qw36_engine_close(eng); return NULL; }
            lw->gpu_buf = backend->upload(eng->ctx, lw->data,
                bytes, lw->dtype);
        }
        if (w->lm_head && w->lm_head != w->embed_tokens) {
            qw36_lazy_w *lw = (qw36_lazy_w *)w->lm_head;
            size_t numel = (size_t)lw->cols * lw->rows;
            size_t bytes = qw36__tensor_bytes(lw->dtype, numel);
            if (!bytes) { qw36_engine_close(eng); return NULL; }
            lw->gpu_buf = backend->upload(eng->ctx, lw->data,
                bytes, lw->dtype);
        } else if (w->lm_head == w->embed_tokens) {
            ((qw36_lazy_w *)w->lm_head)->gpu_buf =
                ((qw36_lazy_w *)w->embed_tokens)->gpu_buf;
        }
    }
    return eng;
}

void qw36_engine_close(qw36_engine *eng)
{
    if (!eng) return;
    qw36__gpu_cache_free(eng);
    if (eng->backend && eng->backend->destroy && eng->ctx)
        eng->backend->destroy(eng->ctx);
    if (eng->owned) {
        for (size_t i = 0; i < eng->owned_n; i++) free(eng->owned[i]);
        free(eng->owned);
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
    const size_t dn_qkv_dim = c->dn_num_value_heads
        ? (size_t)c->dn_num_key_heads * c->dn_key_head_dim * 2
        + (size_t)c->dn_num_value_heads * c->dn_value_head_dim
        : 0;
    const size_t dn_v_dim = c->dn_num_value_heads
        ? (size_t)c->dn_num_value_heads * c->dn_value_head_dim
        : 0;
    const size_t dn_proj_dim = c->dn_num_value_heads
        ? dn_qkv_dim + dn_v_dim + (size_t)c->dn_num_value_heads * 2
        : 0;
    const size_t dn_conv_window = (c->dn_num_value_heads &&
                                   c->dn_conv_kernel_size)
        ? (size_t)c->dn_conv_kernel_size - 1
        : 0;
    const size_t dn_s_elems = c->dn_num_value_heads
        ? (size_t)c->dn_num_value_heads * c->dn_key_head_dim
          * c->dn_value_head_dim
        : 0;

    st->k_cache = (void **)calloc(L, sizeof(void *));
    st->v_cache = (void **)calloc(L, sizeof(void *));
    if (!st->k_cache || !st->v_cache) goto fail;
    for (size_t l = 0; l < L; l++) {
        st->k_cache[l] = calloc((size_t)seq_capacity * kv_dim, sizeof(float));
        st->v_cache[l] = calloc((size_t)seq_capacity * kv_dim, sizeof(float));
        if (!st->k_cache[l] || !st->v_cache[l]) goto fail;
    }

    /* DeltaNet per-layer state. Sized to the conv kernel size and the
     * key/val dims read from config. Zero for layers without DeltaNet —
     * we allocate symmetrically across L for simplicity. */
    if (c->dn_num_value_heads) {
        st->conv_state  = (float **)calloc(L, sizeof(float *));
        st->delta_state = (float **)calloc(L, sizeof(float *));
        if (!st->conv_state || !st->delta_state) goto fail;
        for (size_t l = 0; l < L; l++) {
            st->conv_state[l] =
                (float *)calloc(dn_conv_window * dn_qkv_dim, sizeof(float));
            st->delta_state[l] = (float *)calloc(dn_s_elems, sizeof(float));
            if ((dn_conv_window && !st->conv_state[l]) || !st->delta_state[l])
                goto fail;
        }
    }

    /* Scratch — big enough for attention staging (n_heads*(cap+1) +
     * n_heads*head_dim) and for MLP gate/up. */
    const size_t attn_scratch_n =
        (size_t)c->num_attention_heads * ((size_t)seq_capacity + 1)
      + (size_t)c->num_attention_heads * c->head_dim;
    st->x           = (float *)calloc(hidden,        sizeof(float));
    st->x_rms       = (float *)calloc(hidden,        sizeof(float));
    st->q           = (float *)calloc(q_dim,         sizeof(float));
    if (c->has_q_gate) {
        st->q_full = (float *)calloc(q_dim * 2,      sizeof(float));
        st->q_gate = (float *)calloc(q_dim,          sizeof(float));
        if (!st->q_full || !st->q_gate) goto fail;
    }
    st->k           = (float *)calloc(kv_dim,        sizeof(float));
    st->v           = (float *)calloc(kv_dim,        sizeof(float));
    st->attn_scores = (float *)calloc(attn_scratch_n, sizeof(float));
    st->gate        = (float *)calloc(inter,         sizeof(float));
    st->up          = (float *)calloc(inter,         sizeof(float));
    st->logits      = (float *)calloc(vocab,         sizeof(float));
    /* row_scratch is used by qw36__matmul_lazy. Size = max(hidden, inter, vocab)
     * since matmul reads one full row of the weight at a time. */
    size_t rs_n = hidden;
    if (inter > rs_n) rs_n = inter;
    if (vocab > rs_n) rs_n = vocab;
    /* Note: q_dim/kv_dim are smaller than hidden in practice (n_heads *
     * head_dim ≤ hidden), so the existing bound suffices for QKV matmuls. */
    /* We stash the row scratch in attn_scores' tail — simpler than adding
     * a public field. Realloc attn_scores to fit. */
    free(st->attn_scores);
    st->attn_scores = (float *)calloc(attn_scratch_n + rs_n, sizeof(float));
    if (!st->x || !st->x_rms || !st->q || !st->k || !st->v ||
        !st->attn_scores || !st->gate || !st->up || !st->logits) goto fail;

    if (eng->backend && eng->ctx && eng->backend->alloc && eng->backend->free) {
        qw36_gpu_backend *be = eng->backend;
        qw36_gpu_ctx *ctx = eng->ctx;
        st->gpu_backend = (void *)be;
        st->gpu_ctx = (void *)ctx;
        st->k_cache_dev = (void **)calloc(L, sizeof(void *));
        st->v_cache_dev = (void **)calloc(L, sizeof(void *));
        if (!st->k_cache_dev || !st->v_cache_dev) goto fail;
        if (c->dn_num_value_heads) {
            st->conv_state_dev = (void **)calloc(L, sizeof(void *));
            st->delta_state_dev = (void **)calloc(L, sizeof(void *));
            if (!st->conv_state_dev || !st->delta_state_dev) goto fail;
        }

        const char *fp16_weights_env = getenv("QW36_METAL_FP16_WEIGHTS");
        const char *fp16_kv_env = getenv("QW36_METAL_FP16_KV");
        const char *quant_gpu_env = getenv("QW36_METAL_QUANT_GPU");
        const int gpu_weights_path =
            (fp16_weights_env && atoi(fp16_weights_env) != 0) ||
            (quant_gpu_env && atoi(quant_gpu_env) != 0);
        const int use_fp16_dev_kv =
            be->name && strcmp(be->name, "metal") == 0 &&
            gpu_weights_path &&
            (!fp16_kv_env || atoi(fp16_kv_env) != 0);
        const qw36_dtype dev_kv_dtype =
            use_fp16_dev_kv ? QW36_DTYPE_F16 : QW36_DTYPE_F32;
        const size_t dev_kv_elem_bytes =
            use_fp16_dev_kv ? sizeof(uint16_t) : sizeof(float);
        const size_t cache_bytes =
            (size_t)seq_capacity * kv_dim * dev_kv_elem_bytes;
        for (size_t l = 0; l < L; l++) {
            st->k_cache_dev[l] = be->alloc(ctx, cache_bytes, dev_kv_dtype);
            st->v_cache_dev[l] = be->alloc(ctx, cache_bytes, dev_kv_dtype);
            if (!st->k_cache_dev[l] || !st->v_cache_dev[l]) goto fail;
            if (c->dn_num_value_heads) {
                st->conv_state_dev[l] = be->alloc(ctx,
                    dn_conv_window * dn_qkv_dim * sizeof(float), QW36_DTYPE_F32);
                st->delta_state_dev[l] = be->alloc(ctx,
                    dn_s_elems * sizeof(float), QW36_DTYPE_F32);
                if (!st->conv_state_dev[l] || !st->delta_state_dev[l])
                    goto fail;
                if (be->copy_from_host) {
                    be->copy_from_host(ctx, (qw36_gpu_buf *)st->conv_state_dev[l],
                        st->conv_state[l],
                        dn_conv_window * dn_qkv_dim * sizeof(float));
                    be->copy_from_host(ctx, (qw36_gpu_buf *)st->delta_state_dev[l],
                        st->delta_state[l], dn_s_elems * sizeof(float));
                }
            }
        }

        /* x_dev (the residual stream) stays fp32. The x_rms/q_dev edge
         * buffers can be flipped to fp16 with QW36_METAL_FP16_EDGES=1 to
         * reproduce/bisect #46. This remains opt-in because it currently
         * changes step-0 logits when fed into MPS half GEMV. */
        const char *fp16_edges_env = getenv("QW36_METAL_FP16_EDGES");
        const char *fp16_xrms_env = getenv("QW36_METAL_FP16_XRMS");
        const char *fp16_q_env = getenv("QW36_METAL_FP16_Q");
        const int allow_fp16_edge =
            be->name && strcmp(be->name, "metal") == 0 &&
            fp16_weights_env && atoi(fp16_weights_env) != 0;
        const int use_fp16_edges =
            allow_fp16_edge && fp16_edges_env && atoi(fp16_edges_env) != 0;
        const int use_fp16_xrms =
            use_fp16_edges ||
            (allow_fp16_edge && fp16_xrms_env && atoi(fp16_xrms_env) != 0);
        const int use_fp16_q =
            use_fp16_edges ||
            (allow_fp16_edge && fp16_q_env && atoi(fp16_q_env) != 0);
        const qw36_dtype xrms_dtype =
            use_fp16_xrms ? QW36_DTYPE_F16 : QW36_DTYPE_F32;
        const qw36_dtype q_dev_dtype =
            use_fp16_q ? QW36_DTYPE_F16 : QW36_DTYPE_F32;
        const size_t xrms_elem_bytes =
            use_fp16_xrms ? sizeof(uint16_t) : sizeof(float);
        const size_t q_dev_elem_bytes =
            use_fp16_q ? sizeof(uint16_t) : sizeof(float);
        st->dev_x_dtype = QW36_DTYPE_F32;
        st->x_dev = be->alloc(ctx, hidden * sizeof(float), QW36_DTYPE_F32);
        st->x_rms_dev = be->alloc(ctx, hidden * xrms_elem_bytes, xrms_dtype);
        st->q_dev = be->alloc(ctx, q_dim * q_dev_elem_bytes, q_dev_dtype);
        st->k_dev = be->alloc(ctx, kv_dim * sizeof(float), QW36_DTYPE_F32);
        st->v_dev = be->alloc(ctx, kv_dim * sizeof(float), QW36_DTYPE_F32);
        st->attn_scores_dev = be->alloc(ctx,
            (attn_scratch_n + rs_n) * sizeof(float), QW36_DTYPE_F32);
        st->gate_dev = be->alloc(ctx, inter * sizeof(float), QW36_DTYPE_F32);
        st->up_dev = be->alloc(ctx, inter * sizeof(float), QW36_DTYPE_F32);
        st->logits_dev = be->alloc(ctx, vocab * sizeof(float), QW36_DTYPE_F32);
        if (c->dn_num_value_heads) {
            st->dn_qkv_dev = be->alloc(ctx, dn_proj_dim * sizeof(float),
                                       QW36_DTYPE_F32);
            st->dn_qkv_act_dev = be->alloc(ctx, dn_qkv_dim * sizeof(float),
                                           QW36_DTYPE_F32);
            st->dn_z_dev = be->alloc(ctx, dn_v_dim * sizeof(float),
                                     QW36_DTYPE_F32);
            st->dn_alpha_dev = be->alloc(ctx,
                (size_t)c->dn_num_value_heads * sizeof(float), QW36_DTYPE_F32);
            st->dn_beta_dev = be->alloc(ctx,
                (size_t)c->dn_num_value_heads * sizeof(float), QW36_DTYPE_F32);
            st->dn_gout_dev = be->alloc(ctx, dn_v_dim * sizeof(float),
                                        QW36_DTYPE_F32);
        }
        if (!st->x_dev || !st->x_rms_dev || !st->q_dev ||
            !st->k_dev || !st->v_dev || !st->attn_scores_dev ||
            !st->gate_dev || !st->up_dev || !st->logits_dev)
            goto fail;
        if (c->dn_num_value_heads &&
            (!st->dn_qkv_dev || !st->dn_qkv_act_dev || !st->dn_z_dev ||
             !st->dn_alpha_dev || !st->dn_beta_dev || !st->dn_gout_dev))
            goto fail;
    }

    return st;

fail:
    qw36_state_free(st);
    return NULL;
}

void qw36_state_free(qw36_state *st)
{
    if (!st) return;
    qw36_gpu_backend *be = NULL;
    qw36_gpu_ctx *ctx = NULL;
    (void)qw36__state_backend(st, &be, &ctx);
    if (be && ctx && be->free) {
        if (st->k_cache_dev) {
            for (uint32_t l = 0; l < st->num_layers; l++)
                be->free(ctx, (qw36_gpu_buf *)st->k_cache_dev[l]);
        }
        if (st->v_cache_dev) {
            for (uint32_t l = 0; l < st->num_layers; l++)
                be->free(ctx, (qw36_gpu_buf *)st->v_cache_dev[l]);
        }
        if (st->conv_state_dev) {
            for (uint32_t l = 0; l < st->num_layers; l++)
                be->free(ctx, (qw36_gpu_buf *)st->conv_state_dev[l]);
        }
        if (st->delta_state_dev) {
            for (uint32_t l = 0; l < st->num_layers; l++)
                be->free(ctx, (qw36_gpu_buf *)st->delta_state_dev[l]);
        }
        be->free(ctx, (qw36_gpu_buf *)st->x_dev);
        be->free(ctx, (qw36_gpu_buf *)st->x_rms_dev);
        be->free(ctx, (qw36_gpu_buf *)st->q_dev);
        be->free(ctx, (qw36_gpu_buf *)st->k_dev);
        be->free(ctx, (qw36_gpu_buf *)st->v_dev);
        be->free(ctx, (qw36_gpu_buf *)st->attn_scores_dev);
        be->free(ctx, (qw36_gpu_buf *)st->gate_dev);
        be->free(ctx, (qw36_gpu_buf *)st->up_dev);
        be->free(ctx, (qw36_gpu_buf *)st->logits_dev);
        be->free(ctx, (qw36_gpu_buf *)st->dn_qkv_dev);
        be->free(ctx, (qw36_gpu_buf *)st->dn_qkv_act_dev);
        be->free(ctx, (qw36_gpu_buf *)st->dn_z_dev);
        be->free(ctx, (qw36_gpu_buf *)st->dn_alpha_dev);
        be->free(ctx, (qw36_gpu_buf *)st->dn_beta_dev);
        be->free(ctx, (qw36_gpu_buf *)st->dn_gout_dev);
    }
    free(st->k_cache_dev);
    free(st->v_cache_dev);
    free(st->conv_state_dev);
    free(st->delta_state_dev);
    if (st->k_cache) {
        for (uint32_t l = 0; l < st->num_layers; l++) free(st->k_cache[l]);
        free(st->k_cache);
    }
    if (st->v_cache) {
        for (uint32_t l = 0; l < st->num_layers; l++) free(st->v_cache[l]);
        free(st->v_cache);
    }
    if (st->conv_state) {
        for (uint32_t l = 0; l < st->num_layers; l++) free(st->conv_state[l]);
        free(st->conv_state);
    }
    if (st->delta_state) {
        for (uint32_t l = 0; l < st->num_layers; l++) free(st->delta_state[l]);
        free(st->delta_state);
    }
    free(st->x); free(st->x_rms); free(st->q); free(st->q_full);
    free(st->q_gate); free(st->k); free(st->v);
    free(st->attn_scores); free(st->gate); free(st->up); free(st->logits);
    free(st);
}

/* --------------------------------------------------------------------- */
/* Forward — CPU reference per-layer loop.                                */
/* --------------------------------------------------------------------- */

int qw36_forward(qw36_engine *eng, qw36_state *st, uint32_t token)
{
    const int skip_logits = qw36__skip_logits_this_forward;
    qw36__skip_logits_this_forward = 0;
    if (!eng || !st) return -1;
    const qw36_config  *c = &eng->cfg;
    const qw36_weights *w = &eng->weights;
    if (token >= c->vocab_size) return -1;
    if (st->seq_pos >= st->seq_capacity) return -1;

    qw36_engine *prev_active = qw36__active_engine;
    qw36__active_engine = eng;
    int forward_batch_active = 0;
    if (eng->backend && eng->backend->begin_batch && eng->ctx) {
        eng->backend->begin_batch(eng->ctx);
        forward_batch_active = 1;
    }
#define QW36_FORWARD_END_BATCH() do { \
        if (forward_batch_active && eng->backend && eng->backend->end_batch && eng->ctx) { \
            eng->backend->end_batch(eng->ctx); \
            forward_batch_active = 0; \
        } \
    } while (0)
#define QW36_FORWARD_RETURN(code_) do { \
        QW36_FORWARD_END_BATCH(); \
        qw36__active_engine = prev_active; \
        return (code_); \
    } while (0)

    const size_t hidden = c->hidden_size;
    const size_t inter = c->intermediate_size;
    const size_t attn_scratch_n =
        (size_t)c->num_attention_heads * ((size_t)st->seq_capacity + 1)
      + (size_t)c->num_attention_heads * c->head_dim;
    float *row_scratch = st->attn_scores + attn_scratch_n;

    qw36_gpu_backend *state_be = NULL;
    qw36_gpu_ctx *state_ctx = NULL;
    int gpu_state = qw36__state_backend(st, &state_be, &state_ctx) &&
                    state_be == eng->backend && state_ctx == eng->ctx &&
                    st->x_dev && st->x_rms_dev && st->q_dev && st->logits_dev;
    int x_dev_valid = 0;
    int x_host_valid = 0;

    qw36_forward_ctx fc = {
        .eng = eng,
        .st = st,
        .cfg = c,
        .weights = w,
        .hidden = hidden,
        .inter = inter,
        .row_scratch = row_scratch,
        .gpu_state = gpu_state,
        .x_dev_valid = &x_dev_valid,
        .x_host_valid = &x_host_valid,
        .debug_layer = 0,
    };

    if (gpu_state &&
        qw36__embed_lookup_lazy_dev((qw36_gpu_buf *)st->x_dev,
                                    (const qw36_lazy_w *)w->embed_tokens,
                                    token) == 0) {
        x_dev_valid = 1;
    } else {
        if (qw36__embed_lookup_lazy((const qw36_lazy_w *)w->embed_tokens,
                                    token, st->x))
            QW36_FORWARD_RETURN(-3);
        x_host_valid = 1;
        if (gpu_state && qw36__ensure_x_dev(&fc)) QW36_FORWARD_RETURN(-8);
    }

    static int debug_layer = -1;
    if (debug_layer < 0) {
        const char *e = getenv("QW36_DEBUG_LAYER");
        debug_layer = e && atoi(e) ? 1 : 0;
    }
    fc.debug_layer = debug_layer;
    if (debug_layer) {
        if (qw36__ensure_x_host(&fc)) QW36_FORWARD_RETURN(-8);
        double ss = 0.0;
        for (size_t i = 0; i < hidden; i++) ss += (double)st->x[i] * st->x[i];
        fprintf(stderr, "[layer 0 in]  ||x||=%.4f x[0..3]=%.3f %.3f %.3f\n",
                sqrt(ss), st->x[0], st->x[1], st->x[2]);
    }

    static int bypass_layers = -1;
    static int max_layers = -1;
    if (bypass_layers < 0) {
        const char *e = getenv("QW36_BYPASS_LAYERS");
        bypass_layers = e && atoi(e) ? 1 : 0;
    }
    if (max_layers < 0) {
        const char *e = getenv("QW36_MAX_LAYERS");
        max_layers = e ? atoi(e) : (int)c->num_hidden_layers;
        if (max_layers < 0) max_layers = (int)c->num_hidden_layers;
    }

    for (uint32_t l = 0; l < c->num_hidden_layers; l++) {
        if (bypass_layers) break;
        if ((int)l >= max_layers) break;
        const qw36_layer_weights *L = &w->layers[l];

        /* attn submodule may set this to 1 if it folded residual_add
         * with the next post-attn rmsnorm into one dispatch. Reset
         * per layer. */
        fc.post_attn_rmsnorm_done = 0;

        int rc = 0;
        if (!L->q_proj && L->dn_qkv) {
            rc = qw36__attn_deltanet_forward(&fc, L, l);
        } else if (L->q_proj) {
            rc = qw36__attn_vanilla_forward(&fc, L, l);
        } else {
            QW36_FORWARD_RETURN(-2);
        }
        if (rc) QW36_FORWARD_RETURN(rc);

        if (debug_layer) {
            if (qw36__ensure_x_host(&fc)) QW36_FORWARD_RETURN(-8);
            double ss = 0.0;
            for (size_t i = 0; i < hidden; i++) ss += (double)st->x[i] * st->x[i];
            fprintf(stderr, "[L%u post-attn] ||x||=%.4f\n", l, sqrt(ss));
        }

        rc = qw36__mlp_forward(&fc, L, l);
        if (rc) QW36_FORWARD_RETURN(rc);

        if (debug_layer) {
            if (qw36__ensure_x_host(&fc)) QW36_FORWARD_RETURN(-8);
            double ss = 0.0;
            for (size_t i = 0; i < hidden; i++) ss += (double)st->x[i] * st->x[i];
            fprintf(stderr, "[L%u out]  ||x||=%.4f\n", l, sqrt(ss));
        }
    }

    if (skip_logits) {
        st->seq_pos++;
        QW36_FORWARD_RETURN(0);
    }

    if (gpu_state && eng->backend && eng->backend->rmsnorm && eng->backend->matmul) {
        if (qw36__ensure_x_dev(&fc)) QW36_FORWARD_RETURN(-8);
        int grc = 0;
        grc |= qw36__rmsnorm_dispatch_dev((qw36_gpu_buf *)st->x_rms_dev,
                                          (qw36_gpu_buf *)st->x_dev,
                                          (const float *)w->final_norm,
                                          hidden, c->rms_norm_eps);
        grc |= qw36__matmul_lazy_dev((qw36_gpu_buf *)st->logits_dev,
                                     (qw36_gpu_buf *)st->x_rms_dev,
                                     (const qw36_lazy_w *)w->lm_head);
        if (grc == 0) {
            QW36_FORWARD_END_BATCH();
            if (qw36__state_download_to_host(st, st->logits_dev, st->logits,
                                             (size_t)c->vocab_size * sizeof(float)))
                QW36_FORWARD_RETURN(-8);
            st->seq_pos++;
            QW36_FORWARD_RETURN(0);
        }
    }

    if (qw36__ensure_x_host(&fc)) QW36_FORWARD_RETURN(-8);
    qw36__rmsnorm_dispatch(st->x_rms, st->x, (const float *)w->final_norm,
                           hidden, c->rms_norm_eps);
    qw36__matmul_lazy(st->logits, st->x_rms,
                      (const qw36_lazy_w *)w->lm_head, row_scratch);

    st->seq_pos++;
    QW36_FORWARD_RETURN(0);
#undef QW36_FORWARD_END_BATCH
#undef QW36_FORWARD_RETURN
}

int qw36_prefill(qw36_engine *eng, qw36_state *st,
                 const uint32_t *tokens, size_t length)
{
    for (size_t i = 0; i < length; i++) {
        qw36__skip_logits_this_forward = (i + 1 < length);
        int rc = qw36_forward(eng, st, tokens[i]);
        qw36__skip_logits_this_forward = 0;
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
