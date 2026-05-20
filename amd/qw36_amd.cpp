/* qw36_amd.cpp — AMD HIP backend.
 *
 * Owner: codex. Implements qw36_gpu_backend on top of HIP / ROCm.
 *
 * Build: hipcc -O3 -std=c++17 -I../common qw36_amd.cpp ../common/qw36.c \
 *        ../common/qw36_gguf.c ../common/qw36_tokenizer.c ../common/qw36_cli.c \
 *        -o qw36_amd
 *
 * Implementation plan:
 *   1. Pick the default HIP device, create a stream.
 *   2. Wrap hipMalloc / hipMemcpy in qw36_gpu_buf.
 *   3. Implement kernels in __global__ functions:
 *        rmsnorm_kernel, matmul_kernel, embedding_kernel,
 *        residual_add_kernel, swiglu_kernels, attention_kernel.
 *      For v0 the matmul can be naive; correctness > speed.
 *   4. Expose `qw36_backend_create()` returning the static vtable.
 */

extern "C" {
#include "qw36.h"
#include "qw36_gpu.h"
}

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct qw36_gpu_ctx {
    int device;
    hipStream_t stream;
    /* TODO(codex): scratch buffers, cuBLAS-equivalent (rocBLAS) handle. */
};

struct qw36_gpu_buf {
    void   *dptr;
    size_t  bytes;
    qw36_dtype dtype;
};

/* --------------------------------------------------------------------- */
/* Lifecycle                                                              */
/* --------------------------------------------------------------------- */

static qw36_gpu_ctx *amd_init(char *err, size_t err_cap)
{
    qw36_gpu_ctx *ctx = (qw36_gpu_ctx *)std::calloc(1, sizeof(*ctx));
    if (!ctx) { if (err && err_cap) std::snprintf(err, err_cap, "oom"); return nullptr; }
    /* TODO(codex):
     *   hipGetDevice(&ctx->device);
     *   hipStreamCreate(&ctx->stream);
     *   Verify compute capability, log device name.
     */
    (void)ctx;
    if (err && err_cap) std::snprintf(err, err_cap, "amd_init: TODO");
    std::free(ctx);
    return nullptr;
}

static void amd_destroy(qw36_gpu_ctx *ctx)
{
    if (!ctx) return;
    /* TODO(codex): hipStreamDestroy(ctx->stream); free scratch. */
    std::free(ctx);
}

/* --------------------------------------------------------------------- */
/* Memory                                                                 */
/* --------------------------------------------------------------------- */

static qw36_gpu_buf *amd_upload(qw36_gpu_ctx *ctx, const void *host,
                                size_t bytes, qw36_dtype dtype)
{
    (void)ctx; (void)host; (void)bytes; (void)dtype;
    /* TODO(codex): hipMalloc + hipMemcpyAsync(H2D, stream). */
    return nullptr;
}

static void amd_download(qw36_gpu_ctx *ctx, qw36_gpu_buf *buf,
                         void *host, size_t bytes)
{
    (void)ctx; (void)buf; (void)host; (void)bytes;
    /* TODO(codex): hipMemcpyAsync(D2H, stream); hipStreamSynchronize. */
}

static qw36_gpu_buf *amd_alloc(qw36_gpu_ctx *ctx, size_t bytes, qw36_dtype dtype)
{ (void)ctx; (void)bytes; (void)dtype; return nullptr; /* TODO(codex) */ }

static void amd_free(qw36_gpu_ctx *ctx, qw36_gpu_buf *buf)
{ (void)ctx; (void)buf; /* TODO(codex): hipFree, free(buf). */ }

/* --------------------------------------------------------------------- */
/* Kernels — all are stubs. See DIVISION_OF_WORK.md for the spec.         */
/* --------------------------------------------------------------------- */

static void amd_rmsnorm(qw36_gpu_ctx *ctx, qw36_gpu_buf *out, qw36_gpu_buf *x,
                        qw36_gpu_buf *w, uint32_t hidden, float eps)
{ (void)ctx; (void)out; (void)x; (void)w; (void)hidden; (void)eps; /* TODO(codex) */ }

static void amd_matmul(qw36_gpu_ctx *ctx, qw36_gpu_buf *y, qw36_gpu_buf *x,
                       qw36_gpu_buf *w, uint32_t batch, uint32_t rows, uint32_t cols)
{ (void)ctx; (void)y; (void)x; (void)w; (void)batch; (void)rows; (void)cols; /* TODO(codex) */ }

static void amd_attention(qw36_gpu_ctx *ctx,
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

static void amd_swiglu(qw36_gpu_ctx *ctx, qw36_gpu_buf *y, qw36_gpu_buf *x,
                       qw36_gpu_buf *w_gate, qw36_gpu_buf *w_up, qw36_gpu_buf *w_down,
                       uint32_t hidden, uint32_t inter)
{ (void)ctx; (void)y; (void)x; (void)w_gate; (void)w_up; (void)w_down;
  (void)hidden; (void)inter; /* TODO(codex) */ }

static void amd_residual_add(qw36_gpu_ctx *ctx, qw36_gpu_buf *x, qw36_gpu_buf *y, uint32_t n)
{ (void)ctx; (void)x; (void)y; (void)n; /* TODO(codex) */ }

static void amd_embedding_lookup(qw36_gpu_ctx *ctx, qw36_gpu_buf *y, qw36_gpu_buf *embed,
                                 uint32_t token, uint32_t hidden)
{ (void)ctx; (void)y; (void)embed; (void)token; (void)hidden; /* TODO(codex) */ }

/* --------------------------------------------------------------------- */
/* Vtable                                                                 */
/* --------------------------------------------------------------------- */

static qw36_gpu_backend g_amd_backend = {
    /* name              */ "amd",
    /* init              */ amd_init,
    /* destroy           */ amd_destroy,
    /* upload            */ amd_upload,
    /* download          */ amd_download,
    /* alloc             */ amd_alloc,
    /* free              */ amd_free,
    /* rmsnorm           */ amd_rmsnorm,
    /* matmul            */ amd_matmul,
    /* attention         */ amd_attention,
    /* swiglu_mlp        */ amd_swiglu,
    /* moe_forward       */ nullptr, /* TODO(codex) */
    /* residual_add      */ amd_residual_add,
    /* embedding_lookup  */ amd_embedding_lookup,
};

extern "C" qw36_gpu_backend *qw36_backend_create(void) { return &g_amd_backend; }
