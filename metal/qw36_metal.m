/* qw36_metal.m — Apple Metal backend (Objective-C host code).
 *
 * Owner: codex. Implements qw36_gpu_backend using MTLDevice +
 * MTLComputeCommandEncoder. Shader source lives in qw36_metal.metal,
 * compiled to default.metallib at build time and loaded as a string here.
 *
 * Build: see metal/Makefile. We compile the .metal file to .metallib with
 * xcrun, then link the .m + the common .c files with clang.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "qw36.h"
#include "qw36_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct qw36_gpu_ctx {
    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary>      library;
    /* TODO(codex): cached MTLComputePipelineState per kernel. */
};

struct qw36_gpu_buf {
    id<MTLBuffer> mtl;
    size_t        bytes;
    qw36_dtype    dtype;
};

/* --------------------------------------------------------------------- */
/* Lifecycle                                                              */
/* --------------------------------------------------------------------- */

static qw36_gpu_ctx *metal_init(char *err, size_t err_cap)
{
    qw36_gpu_ctx *ctx = (qw36_gpu_ctx *)calloc(1, sizeof(*ctx));
    if (!ctx) { if (err && err_cap) snprintf(err, err_cap, "oom"); return NULL; }
    /* TODO(codex):
     *   ctx->device = MTLCreateSystemDefaultDevice();
     *   ctx->queue  = [ctx->device newCommandQueue];
     *   ctx->library = [ctx->device newDefaultLibrary]; // loads default.metallib
     *   Cache compute pipelines for each kernel function.
     */
    if (err && err_cap) snprintf(err, err_cap, "metal_init: TODO");
    free(ctx);
    return NULL;
}

static void metal_destroy(qw36_gpu_ctx *ctx)
{
    if (!ctx) return;
    /* ARC will release device/queue/library when ctx goes out of scope, but
     * we still need to free our struct. */
    free(ctx);
}

/* --------------------------------------------------------------------- */
/* Memory                                                                 */
/* --------------------------------------------------------------------- */

static qw36_gpu_buf *metal_upload(qw36_gpu_ctx *ctx, const void *host,
                                  size_t bytes, qw36_dtype dtype)
{
    (void)ctx; (void)host; (void)bytes; (void)dtype;
    /* TODO(codex): [ctx->device newBufferWithBytes:length:options:Shared]. */
    return NULL;
}

static void metal_download(qw36_gpu_ctx *ctx, qw36_gpu_buf *buf,
                           void *host, size_t bytes)
{
    (void)ctx; (void)buf; (void)host; (void)bytes;
    /* TODO(codex): waitUntilCompleted on a commit, memcpy from contents. */
}

static qw36_gpu_buf *metal_alloc(qw36_gpu_ctx *ctx, size_t bytes, qw36_dtype dtype)
{ (void)ctx; (void)bytes; (void)dtype; return NULL; /* TODO(codex) */ }

static void metal_free(qw36_gpu_ctx *ctx, qw36_gpu_buf *buf)
{ (void)ctx; (void)buf; /* TODO(codex) */ }

/* --------------------------------------------------------------------- */
/* Kernels — TODO(codex). Encoders dispatch shaders from qw36_metal.metal. */
/* --------------------------------------------------------------------- */

static void metal_rmsnorm(qw36_gpu_ctx *ctx, qw36_gpu_buf *out, qw36_gpu_buf *x,
                          qw36_gpu_buf *w, uint32_t hidden, float eps)
{ (void)ctx; (void)out; (void)x; (void)w; (void)hidden; (void)eps; /* TODO(codex) */ }

static void metal_matmul(qw36_gpu_ctx *ctx, qw36_gpu_buf *y, qw36_gpu_buf *x,
                         qw36_gpu_buf *w, uint32_t batch, uint32_t rows, uint32_t cols)
{ (void)ctx; (void)y; (void)x; (void)w; (void)batch; (void)rows; (void)cols; /* TODO(codex) */ }

static void metal_attention(qw36_gpu_ctx *ctx,
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

static void metal_swiglu(qw36_gpu_ctx *ctx, qw36_gpu_buf *y, qw36_gpu_buf *x,
                         qw36_gpu_buf *w_gate, qw36_gpu_buf *w_up, qw36_gpu_buf *w_down,
                         uint32_t hidden, uint32_t inter)
{ (void)ctx; (void)y; (void)x; (void)w_gate; (void)w_up; (void)w_down;
  (void)hidden; (void)inter; /* TODO(codex) */ }

static void metal_residual_add(qw36_gpu_ctx *ctx, qw36_gpu_buf *x, qw36_gpu_buf *y, uint32_t n)
{ (void)ctx; (void)x; (void)y; (void)n; /* TODO(codex) */ }

static void metal_embedding_lookup(qw36_gpu_ctx *ctx, qw36_gpu_buf *y, qw36_gpu_buf *embed,
                                   uint32_t token, uint32_t hidden)
{ (void)ctx; (void)y; (void)embed; (void)token; (void)hidden; /* TODO(codex) */ }

/* --------------------------------------------------------------------- */

static qw36_gpu_backend g_metal_backend = {
    .name              = "metal",
    .init              = metal_init,
    .destroy           = metal_destroy,
    .upload            = metal_upload,
    .download          = metal_download,
    .alloc             = metal_alloc,
    .free              = metal_free,
    .rmsnorm           = metal_rmsnorm,
    .matmul            = metal_matmul,
    .attention         = metal_attention,
    .swiglu_mlp        = metal_swiglu,
    .moe_forward       = NULL, /* TODO(codex) */
    .residual_add      = metal_residual_add,
    .embedding_lookup  = metal_embedding_lookup,
};

qw36_gpu_backend *qw36_backend_create(void) { return &g_metal_backend; }
