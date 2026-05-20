/* qw36_cuda.cu — NVIDIA CUDA backend.
 *
 * Owner: codex. Implements qw36_gpu_backend on top of CUDA.
 *
 * Build: nvcc -O3 -std=c++17 -arch=native -I../common qw36_cuda.cu \
 *        ../common/qw36.c ../common/qw36_gguf.c \
 *        ../common/qw36_tokenizer.c ../common/qw36_cli.c -o qw36_cuda
 */

extern "C" {
#include "qw36.h"
#include "qw36_gpu.h"
}

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct qw36_gpu_ctx {
    int          device;
    cudaStream_t stream;
    /* TODO(codex): cuBLAS handle, scratch. */
};

struct qw36_gpu_buf {
    void      *dptr;
    size_t     bytes;
    qw36_dtype dtype;
};

/* --------------------------------------------------------------------- */
/* Lifecycle                                                              */
/* --------------------------------------------------------------------- */

static qw36_gpu_ctx *cuda_init(char *err, size_t err_cap)
{
    qw36_gpu_ctx *ctx = (qw36_gpu_ctx *)std::calloc(1, sizeof(*ctx));
    if (!ctx) { if (err && err_cap) std::snprintf(err, err_cap, "oom"); return nullptr; }
    /* TODO(codex):
     *   cudaGetDevice(&ctx->device);
     *   cudaStreamCreate(&ctx->stream);
     *   Print device name + capability.
     */
    if (err && err_cap) std::snprintf(err, err_cap, "cuda_init: TODO");
    std::free(ctx);
    return nullptr;
}

static void cuda_destroy(qw36_gpu_ctx *ctx)
{
    if (!ctx) return;
    /* TODO(codex): cudaStreamDestroy(ctx->stream); */
    std::free(ctx);
}

/* --------------------------------------------------------------------- */
/* Memory                                                                 */
/* --------------------------------------------------------------------- */

static qw36_gpu_buf *cuda_upload(qw36_gpu_ctx *ctx, const void *host,
                                 size_t bytes, qw36_dtype dtype)
{
    (void)ctx; (void)host; (void)bytes; (void)dtype;
    /* TODO(codex): cudaMallocAsync + cudaMemcpyAsync. */
    return nullptr;
}

static void cuda_download(qw36_gpu_ctx *ctx, qw36_gpu_buf *buf,
                          void *host, size_t bytes)
{ (void)ctx; (void)buf; (void)host; (void)bytes; /* TODO(codex) */ }

static qw36_gpu_buf *cuda_alloc(qw36_gpu_ctx *ctx, size_t bytes, qw36_dtype dtype)
{ (void)ctx; (void)bytes; (void)dtype; return nullptr; /* TODO(codex) */ }

static void cuda_free(qw36_gpu_ctx *ctx, qw36_gpu_buf *buf)
{ (void)ctx; (void)buf; /* TODO(codex) */ }

/* --------------------------------------------------------------------- */
/* Kernels                                                                */
/* --------------------------------------------------------------------- */

/* Example skeleton — codex will replace with real kernels.
 *
 * __global__ void qw36_rmsnorm_kernel(float* out, const float* x, const float* w,
 *                                     uint32_t hidden, float eps) { ... }
 */

static void cuda_rmsnorm(qw36_gpu_ctx *ctx, qw36_gpu_buf *out, qw36_gpu_buf *x,
                         qw36_gpu_buf *w, uint32_t hidden, float eps)
{ (void)ctx; (void)out; (void)x; (void)w; (void)hidden; (void)eps; /* TODO(codex) */ }

static void cuda_matmul(qw36_gpu_ctx *ctx, qw36_gpu_buf *y, qw36_gpu_buf *x,
                        qw36_gpu_buf *w, uint32_t batch, uint32_t rows, uint32_t cols)
{ (void)ctx; (void)y; (void)x; (void)w; (void)batch; (void)rows; (void)cols; /* TODO(codex) */ }

static void cuda_attention(qw36_gpu_ctx *ctx,
                           qw36_gpu_buf *y, qw36_gpu_buf *x,
                           qw36_gpu_buf *wq, qw36_gpu_buf *wk, qw36_gpu_buf *wv,
                           qw36_gpu_buf *q_norm, qw36_gpu_buf *k_norm,
                           qw36_gpu_buf *k_cache, qw36_gpu_buf *v_cache,
                           uint32_t hidden, uint32_t n_heads, uint32_t n_kv,
                           uint32_t head_dim, uint32_t seq_pos,
                           uint32_t seq_capacity,
                           float rope_theta, float partial_rotary_factor)
{
    (void)ctx; (void)y; (void)x; (void)wq; (void)wk; (void)wv;
    (void)q_norm; (void)k_norm; (void)k_cache; (void)v_cache;
    (void)hidden; (void)n_heads; (void)n_kv; (void)head_dim;
    (void)seq_pos; (void)seq_capacity;
    (void)rope_theta; (void)partial_rotary_factor;
    /* TODO(codex) */
}

static void cuda_swiglu(qw36_gpu_ctx *ctx, qw36_gpu_buf *y, qw36_gpu_buf *x,
                        qw36_gpu_buf *w_gate, qw36_gpu_buf *w_up, qw36_gpu_buf *w_down,
                        uint32_t hidden, uint32_t inter)
{ (void)ctx; (void)y; (void)x; (void)w_gate; (void)w_up; (void)w_down;
  (void)hidden; (void)inter; /* TODO(codex) */ }

static void cuda_residual_add(qw36_gpu_ctx *ctx, qw36_gpu_buf *x, qw36_gpu_buf *y, uint32_t n)
{ (void)ctx; (void)x; (void)y; (void)n; /* TODO(codex) */ }

static void cuda_embedding_lookup(qw36_gpu_ctx *ctx, qw36_gpu_buf *y, qw36_gpu_buf *embed,
                                  uint32_t token, uint32_t hidden)
{ (void)ctx; (void)y; (void)embed; (void)token; (void)hidden; /* TODO(codex) */ }

/* --------------------------------------------------------------------- */

static qw36_gpu_backend g_cuda_backend = {
    /* name              */ "cuda",
    /* init              */ cuda_init,
    /* destroy           */ cuda_destroy,
    /* upload            */ cuda_upload,
    /* download          */ cuda_download,
    /* alloc             */ cuda_alloc,
    /* free              */ cuda_free,
    /* rmsnorm           */ cuda_rmsnorm,
    /* matmul            */ cuda_matmul,
    /* attention         */ cuda_attention,
    /* swiglu_mlp        */ cuda_swiglu,
    /* moe_forward       */ nullptr, /* TODO(codex) */
    /* residual_add      */ cuda_residual_add,
    /* embedding_lookup  */ cuda_embedding_lookup,
};

extern "C" qw36_gpu_backend *qw36_backend_create(void) { return &g_cuda_backend; }
