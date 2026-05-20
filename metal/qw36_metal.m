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
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "qw36.h"
#include "qw36_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct qw36_gpu_ctx {
    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary>      library;
    id<MTLComputePipelineState> rmsnorm;
    id<MTLComputePipelineState> matmul;
    id<MTLComputePipelineState> silu_mul;
    id<MTLComputePipelineState> residual_add;
    id<MTLComputePipelineState> embedding_lookup;
    id<MTLComputePipelineState> head_norm_rope;
    id<MTLComputePipelineState> kv_append;
    id<MTLComputePipelineState> attn_scores;
    id<MTLComputePipelineState> attn_softmax;
    id<MTLComputePipelineState> attn_combine;
};

struct qw36_gpu_buf {
    id<MTLBuffer> mtl;
    void         *host_copy;
    size_t        bytes;
    qw36_dtype    dtype;
};

/* --------------------------------------------------------------------- */
/* Host-side dequantization for simple bring-up of lazy GGUF weights.     */
/* --------------------------------------------------------------------- */

#define QW36_QK_K   256
#define QW36_QK8_0  32

static int metal_dtype_is_host_dequant(qw36_dtype dtype)
{
    return dtype == QW36_DTYPE_Q4_K ||
           dtype == QW36_DTYPE_Q6_K ||
           dtype == QW36_DTYPE_Q8_0;
}

static float metal_f16_to_f32(uint16_t h)
{
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
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
    union { uint32_t u; float f; } u;
    u.u = f;
    return u.f;
}

static void metal_dq_q8_0(const uint8_t *blocks, float *out, size_t n)
{
    const size_t nb = n / QW36_QK8_0;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t *b = blocks + i * 34;
        uint16_t dh;
        memcpy(&dh, b, sizeof(dh));
        const float d = metal_f16_to_f32(dh);
        const int8_t *qs = (const int8_t *)(b + 2);
        for (int j = 0; j < QW36_QK8_0; j++) *out++ = d * (float)qs[j];
    }
}

static void metal_q4_K_get_scale_min(int j, const uint8_t *q,
                                     uint8_t *d, uint8_t *m)
{
    if (j < 4) {
        *d = q[j]     & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

static void metal_dq_q4_K(const uint8_t *blocks, float *out, size_t n)
{
    const size_t nb = n / QW36_QK_K;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t *b = blocks + i * 144;
        uint16_t dh, dmh;
        memcpy(&dh,  b,     sizeof(dh));
        memcpy(&dmh, b + 2, sizeof(dmh));
        const float d = metal_f16_to_f32(dh);
        const float dmn = metal_f16_to_f32(dmh);
        const uint8_t *scales = b + 4;
        const uint8_t *qs = b + 16;
        int is = 0;
        for (int j = 0; j < QW36_QK_K; j += 64) {
            uint8_t sc, m;
            metal_q4_K_get_scale_min(is + 0, scales, &sc, &m);
            const float d1 = d * (float)sc, m1 = dmn * (float)m;
            metal_q4_K_get_scale_min(is + 1, scales, &sc, &m);
            const float d2 = d * (float)sc, m2 = dmn * (float)m;
            for (int l = 0; l < 32; l++) *out++ = d1 * (float)(qs[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++) *out++ = d2 * (float)(qs[l] >> 4) - m2;
            qs += 32;
            is += 2;
        }
    }
}

static void metal_dq_q6_K(const uint8_t *blocks, float *out, size_t n)
{
    const size_t nb = n / QW36_QK_K;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t *b = blocks + i * 210;
        const uint8_t *ql = b;
        const uint8_t *qh = b + 128;
        const int8_t *sc = (const int8_t *)(b + 128 + 64);
        uint16_t dh;
        memcpy(&dh, b + 128 + 64 + 16, sizeof(dh));
        const float d = metal_f16_to_f32(dh);
        for (int n_off = 0; n_off < QW36_QK_K; n_off += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                out[l +  0] = d * (float)sc[is + 0] * (float)q1;
                out[l + 32] = d * (float)sc[is + 2] * (float)q2;
                out[l + 64] = d * (float)sc[is + 4] * (float)q3;
                out[l + 96] = d * (float)sc[is + 6] * (float)q4;
            }
            out += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

static int metal_quant_geom(qw36_dtype dtype, size_t *qk, size_t *bytes_per_block)
{
    switch (dtype) {
        case QW36_DTYPE_Q4_K: *qk = 256; *bytes_per_block = 144; return 0;
        case QW36_DTYPE_Q6_K: *qk = 256; *bytes_per_block = 210; return 0;
        case QW36_DTYPE_Q8_0: *qk = 32;  *bytes_per_block = 34;  return 0;
        default: *qk = 0; *bytes_per_block = 0; return -1;
    }
}

static size_t metal_dtype_bytes(qw36_dtype dtype)
{
    switch (dtype) {
        case QW36_DTYPE_F32:  return 4;
        case QW36_DTYPE_F16:  return 2;
        case QW36_DTYPE_BF16: return 2;
        default: return 0;
    }
}

static uint32_t metal_matrix_rows_from_bytes(const qw36_gpu_buf *buf, uint32_t cols)
{
    if (!buf || cols == 0) return 0;

    size_t row_bytes = 0;
    if (metal_dtype_is_host_dequant(buf->dtype)) {
        size_t qk, bpb;
        if (metal_quant_geom(buf->dtype, &qk, &bpb) || cols % qk != 0)
            return 0;
        row_bytes = ((size_t)cols / qk) * bpb;
    } else {
        const size_t elem_bytes = metal_dtype_bytes(buf->dtype);
        if (elem_bytes == 0) return 0;
        row_bytes = (size_t)cols * elem_bytes;
    }

    if (row_bytes == 0 || buf->bytes % row_bytes != 0) return 0;
    const size_t rows = buf->bytes / row_bytes;
    return rows <= UINT32_MAX ? (uint32_t)rows : 0;
}

static int metal_dequant_row(const qw36_gpu_buf *buf, size_t row_idx,
                             size_t cols, float *out)
{
    if (!buf || !buf->host_copy) return -1;
    size_t qk, bpb;
    if (metal_quant_geom(buf->dtype, &qk, &bpb) || cols % qk != 0) return -1;
    const size_t blocks_per_row = cols / qk;
    const uint8_t *row = (const uint8_t *)buf->host_copy
                       + row_idx * blocks_per_row * bpb;
    switch (buf->dtype) {
        case QW36_DTYPE_Q4_K: metal_dq_q4_K(row, out, cols); return 0;
        case QW36_DTYPE_Q6_K: metal_dq_q6_K(row, out, cols); return 0;
        case QW36_DTYPE_Q8_0: metal_dq_q8_0(row, out, cols); return 0;
        default: return -1;
    }
}

static float *metal_dequant_matrix(const qw36_gpu_buf *buf,
                                   uint32_t rows, uint32_t cols)
{
    float *out = (float *)malloc((size_t)rows * cols * sizeof(float));
    if (!out) return NULL;
    for (uint32_t r = 0; r < rows; r++) {
        if (metal_dequant_row(buf, r, cols, out + (size_t)r * cols)) {
            free(out);
            return NULL;
        }
    }
    return out;
}

/* --------------------------------------------------------------------- */
/* Lifecycle                                                              */
/* --------------------------------------------------------------------- */

static id<MTLLibrary> metal_load_library(id<MTLDevice> device,
                                         char *err, size_t err_cap)
{
    NSError *ns_err = nil;
    id<MTLLibrary> lib = [device newDefaultLibrary];
    if (lib) return lib;

    NSString *exe_dir = [[[[NSProcessInfo processInfo] arguments] objectAtIndex:0]
        stringByDeletingLastPathComponent];
    NSArray<NSString *> *paths = @[
        @"default.metallib",
        @"metal/default.metallib",
        [exe_dir stringByAppendingPathComponent:@"default.metallib"],
        [exe_dir stringByAppendingPathComponent:@"metal/default.metallib"]
    ];

    NSFileManager *fm = [NSFileManager defaultManager];
    for (NSString *path in paths) {
        NSString *candidate = path;
        if (![fm fileExistsAtPath:candidate]) continue;
        NSURL *url = [NSURL fileURLWithPath:candidate];
        ns_err = nil;
        lib = [device newLibraryWithURL:url error:&ns_err];
        if (lib) return lib;
    }

    if (err && err_cap)
        snprintf(err, err_cap, "metal: failed to load default.metallib%s%s",
                 ns_err ? ": " : "",
                 ns_err ? [[ns_err localizedDescription] UTF8String] : "");
    return nil;
}

static id<MTLComputePipelineState> metal_make_pipeline(qw36_gpu_ctx *ctx,
                                                       NSString *name,
                                                       char *err,
                                                       size_t err_cap)
{
    id<MTLFunction> fn = [ctx->library newFunctionWithName:name];
    if (!fn) {
        if (err && err_cap)
            snprintf(err, err_cap, "metal: missing shader %s", [name UTF8String]);
        return nil;
    }

    NSError *ns_err = nil;
    id<MTLComputePipelineState> pipe =
        [ctx->device newComputePipelineStateWithFunction:fn error:&ns_err];
    if (!pipe && err && err_cap) {
        snprintf(err, err_cap, "metal: pipeline %s: %s",
                 [name UTF8String],
                 ns_err ? [[ns_err localizedDescription] UTF8String] : "unknown error");
    }
    return pipe;
}

static qw36_gpu_ctx *metal_init(char *err, size_t err_cap)
{
    qw36_gpu_ctx *ctx = (qw36_gpu_ctx *)calloc(1, sizeof(*ctx));
    if (!ctx) { if (err && err_cap) snprintf(err, err_cap, "oom"); return NULL; }

    ctx->device = MTLCreateSystemDefaultDevice();
    if (!ctx->device) {
        if (err && err_cap) snprintf(err, err_cap, "metal: no MTLDevice");
        free(ctx);
        return NULL;
    }

    ctx->queue = [ctx->device newCommandQueue];
    if (!ctx->queue) {
        if (err && err_cap) snprintf(err, err_cap, "metal: newCommandQueue failed");
        ctx->device = nil;
        free(ctx);
        return NULL;
    }

    ctx->library = metal_load_library(ctx->device, err, err_cap);
    if (!ctx->library) {
        ctx->queue = nil;
        ctx->device = nil;
        free(ctx);
        return NULL;
    }

    ctx->rmsnorm          = metal_make_pipeline(ctx, @"qw36_rmsnorm_f32", err, err_cap);
    ctx->matmul           = metal_make_pipeline(ctx, @"qw36_matmul_f32", err, err_cap);
    ctx->silu_mul         = metal_make_pipeline(ctx, @"qw36_silu_mul_f32", err, err_cap);
    ctx->residual_add     = metal_make_pipeline(ctx, @"qw36_residual_add_f32", err, err_cap);
    ctx->embedding_lookup = metal_make_pipeline(ctx, @"qw36_embedding_lookup_f32", err, err_cap);
    ctx->head_norm_rope   = metal_make_pipeline(ctx, @"qw36_head_norm_rope_f32", err, err_cap);
    ctx->kv_append        = metal_make_pipeline(ctx, @"qw36_kv_append_f32", err, err_cap);
    ctx->attn_scores      = metal_make_pipeline(ctx, @"qw36_attn_scores_f32", err, err_cap);
    ctx->attn_softmax     = metal_make_pipeline(ctx, @"qw36_attn_softmax_f32", err, err_cap);
    ctx->attn_combine     = metal_make_pipeline(ctx, @"qw36_attn_combine_f32", err, err_cap);

    if (!ctx->rmsnorm || !ctx->matmul || !ctx->silu_mul || !ctx->residual_add ||
        !ctx->embedding_lookup || !ctx->head_norm_rope || !ctx->kv_append ||
        !ctx->attn_scores || !ctx->attn_softmax || !ctx->attn_combine) {
        ctx->attn_combine = nil;
        ctx->attn_softmax = nil;
        ctx->attn_scores = nil;
        ctx->kv_append = nil;
        ctx->head_norm_rope = nil;
        ctx->embedding_lookup = nil;
        ctx->residual_add = nil;
        ctx->silu_mul = nil;
        ctx->matmul = nil;
        ctx->rmsnorm = nil;
        ctx->library = nil;
        ctx->queue = nil;
        ctx->device = nil;
        free(ctx);
        return NULL;
    }

    return ctx;
}

static void metal_destroy(qw36_gpu_ctx *ctx)
{
    if (!ctx) return;
    ctx->attn_combine = nil;
    ctx->attn_softmax = nil;
    ctx->attn_scores = nil;
    ctx->kv_append = nil;
    ctx->head_norm_rope = nil;
    ctx->embedding_lookup = nil;
    ctx->residual_add = nil;
    ctx->silu_mul = nil;
    ctx->matmul = nil;
    ctx->rmsnorm = nil;
    ctx->library = nil;
    ctx->queue = nil;
    ctx->device = nil;
    free(ctx);
}

/* --------------------------------------------------------------------- */
/* Memory                                                                 */
/* --------------------------------------------------------------------- */

static qw36_gpu_buf *metal_upload(qw36_gpu_ctx *ctx, const void *host,
                                  size_t bytes, qw36_dtype dtype)
{
    if (!ctx || !host) return NULL;
    qw36_gpu_buf *buf = (qw36_gpu_buf *)calloc(1, sizeof(*buf));
    if (!buf) return NULL;
    buf->mtl = [ctx->device newBufferWithBytes:host
                                        length:(bytes ? bytes : 1)
                                       options:MTLResourceStorageModeShared];
    if (!buf->mtl) { free(buf); return NULL; }
    buf->bytes = bytes;
    buf->dtype = dtype;
    if (metal_dtype_is_host_dequant(dtype) && bytes) {
        buf->host_copy = malloc(bytes);
        if (!buf->host_copy) {
            buf->mtl = nil;
            free(buf);
            return NULL;
        }
        memcpy(buf->host_copy, host, bytes);
    }
    return buf;
}

static void metal_download(qw36_gpu_ctx *ctx, qw36_gpu_buf *buf,
                           void *host, size_t bytes)
{
    (void)ctx;
    if (!buf || !host) return;
    if (bytes > buf->bytes) bytes = buf->bytes;
    memcpy(host, [buf->mtl contents], bytes);
}

static qw36_gpu_buf *metal_alloc(qw36_gpu_ctx *ctx, size_t bytes, qw36_dtype dtype)
{
    if (!ctx) return NULL;
    qw36_gpu_buf *buf = (qw36_gpu_buf *)calloc(1, sizeof(*buf));
    if (!buf) return NULL;
    buf->mtl = [ctx->device newBufferWithLength:(bytes ? bytes : 1)
                                        options:MTLResourceStorageModeShared];
    if (!buf->mtl) { free(buf); return NULL; }
    buf->bytes = bytes;
    buf->dtype = dtype;
    return buf;
}

static void metal_free(qw36_gpu_ctx *ctx, qw36_gpu_buf *buf)
{
    (void)ctx;
    if (!buf) return;
    buf->mtl = nil;
    free(buf->host_copy);
    free(buf);
}

/* --------------------------------------------------------------------- */
/* Kernels. Encoders dispatch shaders from qw36_metal.metal. */
/* --------------------------------------------------------------------- */

static void metal_dispatch_1d(qw36_gpu_ctx *ctx,
                              id<MTLComputePipelineState> pipe,
                              NSUInteger n,
                              void (^bind)(id<MTLComputeCommandEncoder> enc))
{
    if (!ctx || !pipe || n == 0) return;
    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    bind(enc);
    NSUInteger tg = pipe.maxTotalThreadsPerThreadgroup;
    if (tg > 256) tg = 256;
    if (tg > n) tg = n;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
  threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

static void metal_rmsnorm(qw36_gpu_ctx *ctx, qw36_gpu_buf *out, qw36_gpu_buf *x,
                          qw36_gpu_buf *w, uint32_t hidden, float eps)
{
    if (!out || !x || !w) return;
    metal_dispatch_1d(ctx, ctx->rmsnorm, hidden, ^(id<MTLComputeCommandEncoder> enc) {
        uint32_t x_dtype = (uint32_t)x->dtype;
        uint32_t w_dtype = (uint32_t)w->dtype;
        [enc setBuffer:out->mtl offset:0 atIndex:0];
        [enc setBuffer:x->mtl   offset:0 atIndex:1];
        [enc setBuffer:w->mtl   offset:0 atIndex:2];
        [enc setBytes:&hidden length:sizeof(hidden) atIndex:3];
        [enc setBytes:&eps    length:sizeof(eps)    atIndex:4];
        [enc setBytes:&x_dtype length:sizeof(x_dtype) atIndex:5];
        [enc setBytes:&w_dtype length:sizeof(w_dtype) atIndex:6];
    });
}

static void metal_matmul(qw36_gpu_ctx *ctx, qw36_gpu_buf *y, qw36_gpu_buf *x,
                         qw36_gpu_buf *w, uint32_t batch, uint32_t rows, uint32_t cols)
{
    if (!y || !x || !w) return;
    if (metal_dtype_is_host_dequant(w->dtype)) {
        float *w_f32 = metal_dequant_matrix(w, rows, cols);
        if (!w_f32) return;
        qw36_gpu_buf *w_tmp = metal_upload(ctx, w_f32,
                                           (size_t)rows * cols * sizeof(float),
                                           QW36_DTYPE_F32);
        free(w_f32);
        if (!w_tmp) return;
        metal_matmul(ctx, y, x, w_tmp, batch, rows, cols);
        metal_free(ctx, w_tmp);
        return;
    }

    if (ctx && batch == 1 && y->dtype == QW36_DTYPE_F32 &&
        x->dtype == QW36_DTYPE_F32 && w->dtype == QW36_DTYPE_F32) {
        MPSMatrixDescriptor *w_desc =
            [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                  columns:cols
                                                 rowBytes:(NSUInteger)cols * sizeof(float)
                                                 dataType:MPSDataTypeFloat32];
        MPSVectorDescriptor *x_desc =
            [MPSVectorDescriptor vectorDescriptorWithLength:cols
                                                   dataType:MPSDataTypeFloat32];
        MPSVectorDescriptor *y_desc =
            [MPSVectorDescriptor vectorDescriptorWithLength:rows
                                                   dataType:MPSDataTypeFloat32];
        MPSMatrix *wm = [[MPSMatrix alloc] initWithBuffer:w->mtl descriptor:w_desc];
        MPSVector *xv = [[MPSVector alloc] initWithBuffer:x->mtl descriptor:x_desc];
        MPSVector *yv = [[MPSVector alloc] initWithBuffer:y->mtl descriptor:y_desc];
        MPSMatrixVectorMultiplication *gemv =
            [[MPSMatrixVectorMultiplication alloc] initWithDevice:ctx->device
                                                             rows:rows
                                                          columns:cols];
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        [gemv encodeToCommandBuffer:cb inputMatrix:wm inputVector:xv resultVector:yv];
        [cb commit];
        [cb waitUntilCompleted];
        return;
    }

    NSUInteger n = (NSUInteger)batch * (NSUInteger)rows;
    metal_dispatch_1d(ctx, ctx->matmul, n, ^(id<MTLComputeCommandEncoder> enc) {
        uint32_t x_dtype = (uint32_t)x->dtype;
        uint32_t w_dtype = (uint32_t)w->dtype;
        [enc setBuffer:y->mtl offset:0 atIndex:0];
        [enc setBuffer:x->mtl offset:0 atIndex:1];
        [enc setBuffer:w->mtl offset:0 atIndex:2];
        [enc setBytes:&rows  length:sizeof(rows)  atIndex:3];
        [enc setBytes:&cols  length:sizeof(cols)  atIndex:4];
        [enc setBytes:&batch length:sizeof(batch) atIndex:5];
        [enc setBytes:&x_dtype length:sizeof(x_dtype) atIndex:6];
        [enc setBytes:&w_dtype length:sizeof(w_dtype) atIndex:7];
    });
}

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
    if (!ctx || !y || !x || !wq || !wk || !wv || !q_norm || !k_norm ||
        !k_cache || !v_cache || n_heads == 0 || n_kv == 0 || head_dim == 0)
        return;

    const uint32_t q_rows = metal_matrix_rows_from_bytes(wq, hidden);
    if (q_rows && q_rows % head_dim == 0) {
        const uint32_t inferred_heads = q_rows / head_dim;
        if (inferred_heads) n_heads = inferred_heads;
    }

    uint32_t q_len = n_heads * head_dim;
    uint32_t kv_len = n_kv * head_dim;
    uint32_t positions = seq_pos + 1;
    if (y->bytes < (size_t)q_len * sizeof(float)) return;
    qw36_gpu_buf *q = metal_alloc(ctx, (size_t)q_len * sizeof(float), QW36_DTYPE_F32);
    qw36_gpu_buf *k = metal_alloc(ctx, (size_t)kv_len * sizeof(float), QW36_DTYPE_F32);
    qw36_gpu_buf *v = metal_alloc(ctx, (size_t)kv_len * sizeof(float), QW36_DTYPE_F32);
    qw36_gpu_buf *scores = metal_alloc(ctx, (size_t)n_heads * positions * sizeof(float),
                                       QW36_DTYPE_F32);
    if (!q || !k || !v || !scores) goto done;

    metal_matmul(ctx, q, x, wq, 1, q_len, hidden);
    metal_matmul(ctx, k, x, wk, 1, kv_len, hidden);
    metal_matmul(ctx, v, x, wv, 1, kv_len, hidden);

    float rms_eps = 1.0e-6f;
    metal_dispatch_1d(ctx, ctx->head_norm_rope, n_heads, ^(id<MTLComputeCommandEncoder> enc) {
        uint32_t w_dtype = (uint32_t)q_norm->dtype;
        [enc setBuffer:q->mtl      offset:0 atIndex:0];
        [enc setBuffer:q_norm->mtl offset:0 atIndex:1];
        [enc setBytes:&n_heads length:sizeof(n_heads) atIndex:2];
        [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:3];
        [enc setBytes:&seq_pos length:sizeof(seq_pos) atIndex:4];
        [enc setBytes:&rope_theta length:sizeof(rope_theta) atIndex:5];
        [enc setBytes:&partial_rotary_factor length:sizeof(partial_rotary_factor) atIndex:6];
        [enc setBytes:&rms_eps length:sizeof(rms_eps) atIndex:7];
        [enc setBytes:&w_dtype length:sizeof(w_dtype) atIndex:8];
    });
    metal_dispatch_1d(ctx, ctx->head_norm_rope, n_kv, ^(id<MTLComputeCommandEncoder> enc) {
        uint32_t w_dtype = (uint32_t)k_norm->dtype;
        [enc setBuffer:k->mtl      offset:0 atIndex:0];
        [enc setBuffer:k_norm->mtl offset:0 atIndex:1];
        [enc setBytes:&n_kv length:sizeof(n_kv) atIndex:2];
        [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:3];
        [enc setBytes:&seq_pos length:sizeof(seq_pos) atIndex:4];
        [enc setBytes:&rope_theta length:sizeof(rope_theta) atIndex:5];
        [enc setBytes:&partial_rotary_factor length:sizeof(partial_rotary_factor) atIndex:6];
        [enc setBytes:&rms_eps length:sizeof(rms_eps) atIndex:7];
        [enc setBytes:&w_dtype length:sizeof(w_dtype) atIndex:8];
    });

    metal_dispatch_1d(ctx, ctx->kv_append, kv_len, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:k_cache->mtl offset:0 atIndex:0];
        [enc setBuffer:v_cache->mtl offset:0 atIndex:1];
        [enc setBuffer:k->mtl       offset:0 atIndex:2];
        [enc setBuffer:v->mtl       offset:0 atIndex:3];
        [enc setBytes:&seq_pos length:sizeof(seq_pos) atIndex:4];
        [enc setBytes:&seq_capacity length:sizeof(seq_capacity) atIndex:5];
        [enc setBytes:&kv_len length:sizeof(kv_len) atIndex:6];
    });

    NSUInteger score_n = (NSUInteger)n_heads * positions;
    metal_dispatch_1d(ctx, ctx->attn_scores, score_n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:scores->mtl  offset:0 atIndex:0];
        [enc setBuffer:q->mtl       offset:0 atIndex:1];
        [enc setBuffer:k_cache->mtl offset:0 atIndex:2];
        [enc setBytes:&n_heads length:sizeof(n_heads) atIndex:3];
        [enc setBytes:&n_kv length:sizeof(n_kv) atIndex:4];
        [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:5];
        [enc setBytes:&seq_pos length:sizeof(seq_pos) atIndex:6];
        [enc setBytes:&seq_capacity length:sizeof(seq_capacity) atIndex:7];
    });
    metal_dispatch_1d(ctx, ctx->attn_softmax, n_heads, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:scores->mtl offset:0 atIndex:0];
        [enc setBytes:&n_heads length:sizeof(n_heads) atIndex:1];
        [enc setBytes:&seq_pos length:sizeof(seq_pos) atIndex:2];
    });
    metal_dispatch_1d(ctx, ctx->attn_combine, q_len, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:y->mtl       offset:0 atIndex:0];
        [enc setBuffer:scores->mtl  offset:0 atIndex:1];
        [enc setBuffer:v_cache->mtl offset:0 atIndex:2];
        [enc setBytes:&n_heads length:sizeof(n_heads) atIndex:3];
        [enc setBytes:&n_kv length:sizeof(n_kv) atIndex:4];
        [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:5];
        [enc setBytes:&seq_pos length:sizeof(seq_pos) atIndex:6];
        [enc setBytes:&seq_capacity length:sizeof(seq_capacity) atIndex:7];
    });

done:
    metal_free(ctx, scores);
    metal_free(ctx, v);
    metal_free(ctx, k);
    metal_free(ctx, q);
}

static void metal_swiglu(qw36_gpu_ctx *ctx, qw36_gpu_buf *y, qw36_gpu_buf *x,
                         qw36_gpu_buf *w_gate, qw36_gpu_buf *w_up, qw36_gpu_buf *w_down,
                         uint32_t hidden, uint32_t inter)
{
    if (!ctx || !y || !x || !w_gate || !w_up || !w_down) return;
    qw36_gpu_buf *gate = metal_alloc(ctx, (size_t)inter * sizeof(float), QW36_DTYPE_F32);
    qw36_gpu_buf *up = metal_alloc(ctx, (size_t)inter * sizeof(float), QW36_DTYPE_F32);
    if (!gate || !up) goto done;
    metal_matmul(ctx, gate, x, w_gate, 1, inter, hidden);
    metal_matmul(ctx, up,   x, w_up,   1, inter, hidden);
    metal_dispatch_1d(ctx, ctx->silu_mul, inter, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:gate->mtl offset:0 atIndex:0];
        [enc setBuffer:up->mtl   offset:0 atIndex:1];
        [enc setBytes:&inter length:sizeof(inter) atIndex:2];
    });
    metal_matmul(ctx, y, gate, w_down, 1, hidden, inter);

done:
    metal_free(ctx, up);
    metal_free(ctx, gate);
}

static void metal_residual_add(qw36_gpu_ctx *ctx, qw36_gpu_buf *x, qw36_gpu_buf *y, uint32_t n)
{
    if (!x || !y) return;
    metal_dispatch_1d(ctx, ctx->residual_add, n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:x->mtl offset:0 atIndex:0];
        [enc setBuffer:y->mtl offset:0 atIndex:1];
        [enc setBytes:&n length:sizeof(n) atIndex:2];
    });
}

static void metal_embedding_lookup(qw36_gpu_ctx *ctx, qw36_gpu_buf *y, qw36_gpu_buf *embed,
                                   uint32_t token, uint32_t hidden)
{
    if (!y || !embed) return;
    if (metal_dtype_is_host_dequant(embed->dtype)) {
        float *row = (float *)malloc((size_t)hidden * sizeof(float));
        if (!row) return;
        if (metal_dequant_row(embed, token, hidden, row) == 0)
            memcpy([y->mtl contents], row, (size_t)hidden * sizeof(float));
        free(row);
        (void)ctx;
        return;
    }

    metal_dispatch_1d(ctx, ctx->embedding_lookup, hidden, ^(id<MTLComputeCommandEncoder> enc) {
        uint32_t embed_dtype = (uint32_t)embed->dtype;
        [enc setBuffer:y->mtl     offset:0 atIndex:0];
        [enc setBuffer:embed->mtl offset:0 atIndex:1];
        [enc setBytes:&token  length:sizeof(token)  atIndex:2];
        [enc setBytes:&hidden length:sizeof(hidden) atIndex:3];
        [enc setBytes:&embed_dtype length:sizeof(embed_dtype) atIndex:4];
    });
}

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
    .moe_forward       = NULL,
    .residual_add      = metal_residual_add,
    .embedding_lookup  = metal_embedding_lookup,
};

qw36_gpu_backend *qw36_backend_create(void) { return &g_metal_backend; }
