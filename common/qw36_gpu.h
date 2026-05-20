/* qw36_gpu.h — backend vtable.
 *
 * Every backend (amd, metal, cuda) exports exactly one symbol:
 *
 *     struct qw36_gpu_backend *qw36_backend_create(void);
 *
 * The engine calls into the vtable for the hot path (matmul, attention,
 * sampling). CPU fallback lives in qw36.c and is the reference
 * implementation — keep it numerically equivalent to the GPU paths so we
 * can diff outputs in tests/.
 *
 * Lifetime: the backend owns its device context. `destroy` must release
 * everything. Tensors uploaded via `upload` are owned by the backend until
 * `destroy` runs.
 */

#ifndef QW36_GPU_H
#define QW36_GPU_H

#include <stddef.h>
#include <stdint.h>

#include "qw36.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque device handle (e.g. cudaStream_t wrapper, MTLCommandQueue wrapper).
 * Backends define the concrete type in their own .cu / .m / .cpp. */
typedef struct qw36_gpu_ctx qw36_gpu_ctx;

/* Device buffer handle. Implementation-defined. */
typedef struct qw36_gpu_buf qw36_gpu_buf;

typedef struct qw36_gpu_backend {
    const char *name;     /* "amd" | "metal" | "cuda" */

    /* Lifecycle */
    qw36_gpu_ctx *(*init)(char *err, size_t err_cap);
    void          (*destroy)(qw36_gpu_ctx *ctx);

    /* Tensor management */
    qw36_gpu_buf *(*upload)(qw36_gpu_ctx *ctx, const void *host,
                            size_t bytes, qw36_dtype dtype);
    void          (*download)(qw36_gpu_ctx *ctx, qw36_gpu_buf *buf,
                              void *host, size_t bytes);
    qw36_gpu_buf *(*alloc)(qw36_gpu_ctx *ctx, size_t bytes,
                           qw36_dtype dtype);
    void          (*free)(qw36_gpu_ctx *ctx, qw36_gpu_buf *buf);

    /* Math kernels — all blocking on the backend's default stream. Shapes
     * are passed explicitly so backends do not need to track them. */
    void (*rmsnorm)(qw36_gpu_ctx *ctx,
                    qw36_gpu_buf *out, qw36_gpu_buf *x,
                    qw36_gpu_buf *weight,
                    uint32_t hidden, float eps);

    /* y = x @ W^T,  W is [rows, cols], x is [batch, cols], y is [batch, rows] */
    void (*matmul)(qw36_gpu_ctx *ctx,
                   qw36_gpu_buf *y, qw36_gpu_buf *x, qw36_gpu_buf *w,
                   uint32_t batch, uint32_t rows, uint32_t cols);

    /* Fused QKV + rope + GQA attention against the KV cache up to seq_pos.
     * On entry: x is [hidden]. On exit: y is [hidden] (attn output, before
     * o_proj). q_norm/k_norm are applied internally. */
    void (*attention)(qw36_gpu_ctx *ctx,
                      qw36_gpu_buf *y,
                      qw36_gpu_buf *x,
                      qw36_gpu_buf *wq, qw36_gpu_buf *wk, qw36_gpu_buf *wv,
                      qw36_gpu_buf *q_norm, qw36_gpu_buf *k_norm,
                      qw36_gpu_buf *k_cache, qw36_gpu_buf *v_cache,
                      uint32_t hidden, uint32_t n_heads, uint32_t n_kv,
                      uint32_t head_dim, uint32_t seq_pos,
                      uint32_t seq_capacity,
                      float rope_theta, float partial_rotary_factor);

    /* SwiGLU MLP: y = down(silu(gate(x)) * up(x)) */
    void (*swiglu_mlp)(qw36_gpu_ctx *ctx,
                       qw36_gpu_buf *y, qw36_gpu_buf *x,
                       qw36_gpu_buf *w_gate, qw36_gpu_buf *w_up,
                       qw36_gpu_buf *w_down,
                       uint32_t hidden, uint32_t intermediate);

    /* MoE forward (optional — set to NULL on dense-only builds). */
    void (*moe_forward)(qw36_gpu_ctx *ctx,
                        qw36_gpu_buf *y, qw36_gpu_buf *x,
                        qw36_gpu_buf *router,
                        qw36_gpu_buf *expert_gate,
                        qw36_gpu_buf *expert_up,
                        qw36_gpu_buf *expert_down,
                        qw36_gpu_buf *shared_gate,
                        qw36_gpu_buf *shared_up,
                        qw36_gpu_buf *shared_down,
                        uint32_t hidden, uint32_t inter,
                        uint32_t num_experts, uint32_t experts_per_tok,
                        uint8_t  norm_topk);

    /* Residual add: x += y. */
    void (*residual_add)(qw36_gpu_ctx *ctx,
                         qw36_gpu_buf *x, qw36_gpu_buf *y,
                         uint32_t n);

    /* Embedding lookup: y = embed[token]. */
    void (*embedding_lookup)(qw36_gpu_ctx *ctx,
                             qw36_gpu_buf *y, qw36_gpu_buf *embed,
                             uint32_t token, uint32_t hidden);
} qw36_gpu_backend;

/* Each backend implements this. The CLI links against exactly one. */
qw36_gpu_backend *qw36_backend_create(void);

#ifdef __cplusplus
}
#endif

#endif /* QW36_GPU_H */
