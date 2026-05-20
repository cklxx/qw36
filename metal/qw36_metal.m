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
    id<MTLComputePipelineState> dn_conv1d_silu;
    id<MTLComputePipelineState> dn_prep_gdr;
    id<MTLComputePipelineState> dn_gated_delta;
    id<MTLComputePipelineState> dn_reorder_gdr_y;
    id<MTLComputePipelineState> dn_gated_rmsnorm;
    id<MTLComputePipelineState> residual_add;
    id<MTLComputePipelineState> embedding_lookup;
    id<MTLComputePipelineState> head_norm_rope;
    id<MTLComputePipelineState> kv_append;
    id<MTLComputePipelineState> attn_scores;
    id<MTLComputePipelineState> attn_softmax;
    id<MTLComputePipelineState> attn_combine;
    /* Cache of MPSMatrixVectorMultiplication objects keyed by
     * "<rows>x<cols>" — building one of these per matmul has measurable
     * cost (Metal shader compile lookup); cache cuts the per-call overhead
     * by 2-3x and is essential to get past ~10 tok/s on this kind of
     * decode workload. */
    NSMutableDictionary<NSString *, MPSMatrixVectorMultiplication *> *mps_cache;
    NSMutableDictionary<NSNumber *, NSMutableArray<id<MTLBuffer>> *> *buf_pool;

    /* When non-nil, every metal op encodes into this command buffer
     * instead of creating + committing one per call. Set by begin_batch
     * and cleared by end_batch (after commit + waitUntilCompleted). This
     * eliminates the ~200 sync points per token that otherwise dominate
     * decode latency. */
    id<MTLCommandBuffer> batch_cb;
    BOOL batch_active;

    qw36_gpu_buf *attn_q_scratch;
    qw36_gpu_buf *attn_k_scratch;
    qw36_gpu_buf *attn_v_scratch;
    qw36_gpu_buf *attn_scores_scratch;
    qw36_gpu_buf *swiglu_gate_scratch;
    qw36_gpu_buf *swiglu_up_scratch;
    qw36_gpu_buf *dn_q_scratch;
    qw36_gpu_buf *dn_k_scratch;
    qw36_gpu_buf *dn_v_scratch;
    qw36_gpu_buf *dn_g_scratch;
    qw36_gpu_buf *dn_beta_scratch;
    qw36_gpu_buf *dn_y_grouped_scratch;
};

struct qw36_gpu_buf {
    id<MTLBuffer> mtl;
    void         *host_copy;
    size_t        bytes;
    qw36_dtype    dtype;
};

static void metal_release_buf(qw36_gpu_buf *buf)
{
    if (!buf) return;
    buf->mtl = nil;
    free(buf->host_copy);
    free(buf);
}

/* --------------------------------------------------------------------- */
/* Host-side dequantization for simple bring-up of lazy GGUF weights.     */
/* --------------------------------------------------------------------- */

#define QW36_QK_K   256
#define QW36_QK8_0  32

static int metal_dtype_is_host_dequant(qw36_dtype dtype)
{
    return dtype == QW36_DTYPE_Q2_K ||
           dtype == QW36_DTYPE_Q3_K ||
           dtype == QW36_DTYPE_Q4_K ||
           dtype == QW36_DTYPE_Q5_K ||
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

static void metal_dq_q2_K(const uint8_t *blocks, float *out, size_t n)
{
    const size_t nb = n / QW36_QK_K;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t *b = blocks + i * 84;
        const uint8_t *scales = b;
        const uint8_t *qs = b + 16;
        uint16_t dh, dmh;
        memcpy(&dh,  b + 80, sizeof(dh));
        memcpy(&dmh, b + 82, sizeof(dmh));
        const float d = metal_f16_to_f32(dh);
        const float dmn = metal_f16_to_f32(dmh);
        int is = 0;
        for (int nblk = 0; nblk < QW36_QK_K; nblk += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t sc = scales[is++];
                float dl = d * (float)(sc & 0xF);
                float ml = dmn * (float)(sc >> 4);
                for (int l = 0; l < 16; l++)
                    *out++ = dl * (float)((int8_t)((qs[l] >> shift) & 3)) - ml;
                sc = scales[is++];
                dl = d * (float)(sc & 0xF);
                ml = dmn * (float)(sc >> 4);
                for (int l = 0; l < 16; l++)
                    *out++ = dl * (float)((int8_t)((qs[l + 16] >> shift) & 3)) - ml;
                shift += 2;
            }
            qs += 32;
        }
    }
}

static void metal_dq_q3_K(const uint8_t *blocks, float *out, size_t n)
{
    const size_t nb = n / QW36_QK_K;
    static const uint32_t kmask1 = 0x03030303;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t *b = blocks + i * 110;
        const uint8_t *hm = b;
        const uint8_t *qs = b + 32;
        uint16_t dh;
        memcpy(&dh, b + 32 + 64 + 12, sizeof(dh));
        const float d_all = metal_f16_to_f32(dh);

        uint32_t aux[4] = {0, 0, 0, 0};
        memcpy(aux, b + 32 + 64, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t *scales = (const int8_t *)aux;

        uint8_t mask = 1;
        int is = 0;
        for (int nblk = 0; nblk < QW36_QK_K; nblk += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                const float dl = d_all * (float)((int)scales[is++] - 32);
                for (int l = 0; l < 32; l++) {
                    int q = ((qs[l] >> shift) & 3) - ((hm[l] & mask) ? 0 : 4);
                    *out++ = dl * (float)q;
                }
                shift += 2;
                mask <<= 1;
            }
            qs += 32;
        }
    }
}

static void metal_dq_q5_K(const uint8_t *blocks, float *out, size_t n)
{
    const size_t nb = n / QW36_QK_K;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t *b = blocks + i * 176;
        uint16_t dh, dmh;
        memcpy(&dh,  b,     sizeof(dh));
        memcpy(&dmh, b + 2, sizeof(dmh));
        const float d = metal_f16_to_f32(dh);
        const float dmn = metal_f16_to_f32(dmh);
        const uint8_t *scales = b + 4;
        const uint8_t *qh = b + 16;
        const uint8_t *ql = b + 48;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QW36_QK_K; j += 64) {
            uint8_t sc, m;
            metal_q4_K_get_scale_min(is + 0, scales, &sc, &m);
            const float d1 = d * (float)sc, m1 = dmn * (float)m;
            metal_q4_K_get_scale_min(is + 1, scales, &sc, &m);
            const float d2 = d * (float)sc, m2 = dmn * (float)m;
            for (int l = 0; l < 32; l++)
                *out++ = d1 * (float)((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
            for (int l = 0; l < 32; l++)
                *out++ = d2 * (float)((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
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
        case QW36_DTYPE_Q2_K: *qk = 256; *bytes_per_block = 84;  return 0;
        case QW36_DTYPE_Q3_K: *qk = 256; *bytes_per_block = 110; return 0;
        case QW36_DTYPE_Q4_K: *qk = 256; *bytes_per_block = 144; return 0;
        case QW36_DTYPE_Q5_K: *qk = 256; *bytes_per_block = 176; return 0;
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
        case QW36_DTYPE_Q2_K: metal_dq_q2_K(row, out, cols); return 0;
        case QW36_DTYPE_Q3_K: metal_dq_q3_K(row, out, cols); return 0;
        case QW36_DTYPE_Q4_K: metal_dq_q4_K(row, out, cols); return 0;
        case QW36_DTYPE_Q5_K: metal_dq_q5_K(row, out, cols); return 0;
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
    ctx->mps_cache = [[NSMutableDictionary alloc] init];

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
    ctx->dn_conv1d_silu   = metal_make_pipeline(ctx, @"qw36_dn_conv1d_silu_f32", err, err_cap);
    ctx->dn_prep_gdr      = metal_make_pipeline(ctx, @"qw36_compute_g_beta_norm_qk", err, err_cap);
    ctx->dn_gated_delta   = metal_make_pipeline(ctx, @"qw36_gated_delta_step_f32", err, err_cap);
    ctx->dn_reorder_gdr_y = metal_make_pipeline(ctx, @"qw36_dn_reorder_grouped_y_to_raw_f32", err, err_cap);
    ctx->dn_gated_rmsnorm = metal_make_pipeline(ctx, @"qw36_dn_gated_rmsnorm_f32", err, err_cap);
    ctx->residual_add     = metal_make_pipeline(ctx, @"qw36_residual_add_f32", err, err_cap);
    ctx->embedding_lookup = metal_make_pipeline(ctx, @"qw36_embedding_lookup_f32", err, err_cap);
    ctx->head_norm_rope   = metal_make_pipeline(ctx, @"qw36_head_norm_rope_f32", err, err_cap);
    ctx->kv_append        = metal_make_pipeline(ctx, @"qw36_kv_append_f32", err, err_cap);
    ctx->attn_scores      = metal_make_pipeline(ctx, @"qw36_attn_scores_f32", err, err_cap);
    ctx->attn_softmax     = metal_make_pipeline(ctx, @"qw36_attn_softmax_f32", err, err_cap);
    ctx->attn_combine     = metal_make_pipeline(ctx, @"qw36_attn_combine_f32", err, err_cap);

    if (!ctx->rmsnorm || !ctx->matmul || !ctx->silu_mul ||
        !ctx->dn_conv1d_silu || !ctx->dn_gated_delta ||
        !ctx->dn_prep_gdr || !ctx->dn_reorder_gdr_y ||
        !ctx->dn_gated_rmsnorm || !ctx->residual_add ||
        !ctx->embedding_lookup || !ctx->head_norm_rope || !ctx->kv_append ||
        !ctx->attn_scores || !ctx->attn_softmax || !ctx->attn_combine) {
        ctx->attn_combine = nil;
        ctx->attn_softmax = nil;
        ctx->attn_scores = nil;
        ctx->kv_append = nil;
        ctx->head_norm_rope = nil;
        ctx->embedding_lookup = nil;
        ctx->residual_add = nil;
        ctx->dn_gated_rmsnorm = nil;
        ctx->dn_reorder_gdr_y = nil;
        ctx->dn_gated_delta = nil;
        ctx->dn_prep_gdr = nil;
        ctx->dn_conv1d_silu = nil;
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
    if (ctx->batch_cb) {
        [ctx->batch_cb commit];
        [ctx->batch_cb waitUntilCompleted];
        ctx->batch_cb = nil;
    }
    metal_release_buf(ctx->dn_y_grouped_scratch);
    metal_release_buf(ctx->dn_beta_scratch);
    metal_release_buf(ctx->dn_g_scratch);
    metal_release_buf(ctx->dn_v_scratch);
    metal_release_buf(ctx->dn_k_scratch);
    metal_release_buf(ctx->dn_q_scratch);
    metal_release_buf(ctx->swiglu_up_scratch);
    metal_release_buf(ctx->swiglu_gate_scratch);
    metal_release_buf(ctx->attn_scores_scratch);
    metal_release_buf(ctx->attn_v_scratch);
    metal_release_buf(ctx->attn_k_scratch);
    metal_release_buf(ctx->attn_q_scratch);
    ctx->mps_cache = nil;
    ctx->buf_pool  = nil;
    ctx->attn_combine = nil;
    ctx->attn_softmax = nil;
    ctx->attn_scores = nil;
    ctx->kv_append = nil;
    ctx->head_norm_rope = nil;
    ctx->embedding_lookup = nil;
    ctx->residual_add = nil;
    ctx->dn_gated_rmsnorm = nil;
    ctx->dn_reorder_gdr_y = nil;
    ctx->dn_gated_delta = nil;
    ctx->dn_prep_gdr = nil;
    ctx->dn_conv1d_silu = nil;
    ctx->silu_mul = nil;
    ctx->matmul = nil;
    ctx->rmsnorm = nil;
    ctx->library = nil;
    ctx->queue = nil;
    ctx->device = nil;
    free(ctx);
}

static void metal_begin_batch(qw36_gpu_ctx *ctx)
{
    if (!ctx) return;
    if (ctx->batch_active) return;       /* already in a batch */
    ctx->batch_active = YES;
    ctx->batch_cb = [ctx->queue commandBuffer];
}

static void metal_end_batch(qw36_gpu_ctx *ctx)
{
    if (!ctx || !ctx->batch_active) return;
    ctx->batch_active = NO;
    if (ctx->batch_cb) {
        id<MTLCommandBuffer> cb = ctx->batch_cb;
        ctx->batch_cb = nil;          /* clear *before* commit so re-entrant
                                       * ops within end_batch don't reuse it */
        [cb commit];
        [cb waitUntilCompleted];
    }
}

/* Helper: return the command buffer ops should encode into.
 * If we're inside begin_batch/end_batch, use the shared one and DON'T
 * commit/wait. Otherwise create a fresh one with per-op commit semantics
 * (legacy behavior). Returns the cb and sets *owns to 1 if caller must
 * commit+wait, 0 otherwise. */
static id<MTLCommandBuffer> metal_cb_for_op(qw36_gpu_ctx *ctx, int *owns)
{
    if (ctx->batch_active) {
        if (!ctx->batch_cb)
            ctx->batch_cb = [ctx->queue commandBuffer];
        *owns = 0;
        return ctx->batch_cb;
    }
    *owns = 1; return [ctx->queue commandBuffer];
}

static void metal_flush_batch(qw36_gpu_ctx *ctx)
{
    if (!ctx || !ctx->batch_cb) return;
    id<MTLCommandBuffer> cb = ctx->batch_cb;
    ctx->batch_cb = nil;
    [cb commit];
    [cb waitUntilCompleted];
}

/* --------------------------------------------------------------------- */
/* Memory                                                                 */
/* --------------------------------------------------------------------- */

/* Bucketed MTLBuffer pool. Returning a buffer to the pool instead of
 * deallocating cuts the per-op host overhead by ~30% (newBufferWith*
 * touches the page table on each call). Buckets are powers of two
 * (min 256 B); per-bucket free list is capped to bound peak memory.
 *
 * On Apple unified memory the buffer's .contents is a host-writable
 * pointer, so we use it as a normal staging slot. */
static id<MTLBuffer> metal_pool_take(qw36_gpu_ctx *ctx, size_t bytes) {
    if (!bytes) bytes = 1;
    size_t bucket = 256;
    while (bucket < bytes) bucket <<= 1;
    if (!ctx->buf_pool)
        ctx->buf_pool = [[NSMutableDictionary alloc] init];
    NSMutableArray<id<MTLBuffer>> *arr = ctx->buf_pool[@(bucket)];
    if (arr && arr.count) {
        id<MTLBuffer> b = arr.lastObject;
        [arr removeLastObject];
        return b;
    }
    return [ctx->device newBufferWithLength:bucket
                                    options:MTLResourceStorageModeShared];
}

static void metal_pool_release(qw36_gpu_ctx *ctx, id<MTLBuffer> b) {
    if (!b || !ctx) return;
    NSUInteger len = b.length;
    size_t bucket = 256;
    while (bucket < len) bucket <<= 1;
    if (!ctx->buf_pool)
        ctx->buf_pool = [[NSMutableDictionary alloc] init];
    NSNumber *key = @(bucket);
    NSMutableArray<id<MTLBuffer>> *arr = ctx->buf_pool[key];
    if (!arr) {
        arr = [[NSMutableArray alloc] init];
        ctx->buf_pool[key] = arr;
    }
    if (arr.count < 16) [arr addObject:b];
}

static qw36_gpu_buf *metal_upload(qw36_gpu_ctx *ctx, const void *host,
                                  size_t bytes, qw36_dtype dtype)
{
    if (!ctx || !host) return NULL;
    qw36_gpu_buf *buf = (qw36_gpu_buf *)calloc(1, sizeof(*buf));
    if (!buf) return NULL;
    buf->mtl = metal_pool_take(ctx, bytes);
    if (!buf->mtl) { free(buf); return NULL; }
    if (bytes) memcpy([buf->mtl contents], host, bytes);
    buf->bytes = bytes;
    buf->dtype = dtype;
    if (metal_dtype_is_host_dequant(dtype) && bytes) {
        buf->host_copy = malloc(bytes);
        if (!buf->host_copy) {
            metal_pool_release(ctx, buf->mtl);
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
    if (!buf || !host) return;
    metal_flush_batch(ctx);
    if (bytes > buf->bytes) bytes = buf->bytes;
    memcpy(host, [buf->mtl contents], bytes);
}

static void metal_copy_from_host(qw36_gpu_ctx *ctx, qw36_gpu_buf *buf,
                                 const void *host, size_t bytes)
{
    if (!buf || !host) return;
    metal_flush_batch(ctx);
    if (bytes > buf->bytes) bytes = buf->bytes;
    memcpy([buf->mtl contents], host, bytes);
}

static qw36_gpu_buf *metal_alloc(qw36_gpu_ctx *ctx, size_t bytes, qw36_dtype dtype)
{
    if (!ctx) return NULL;
    qw36_gpu_buf *buf = (qw36_gpu_buf *)calloc(1, sizeof(*buf));
    if (!buf) return NULL;
    buf->mtl = metal_pool_take(ctx, bytes);
    if (!buf->mtl) { free(buf); return NULL; }
    buf->bytes = bytes;
    buf->dtype = dtype;
    return buf;
}

static void metal_free(qw36_gpu_ctx *ctx, qw36_gpu_buf *buf)
{
    if (!buf) return;
    if (ctx && buf->mtl) metal_pool_release(ctx, buf->mtl);
    buf->mtl = nil;
    if (buf->host_copy) { free(buf->host_copy); buf->host_copy = NULL; }
    free(buf);
}

static qw36_gpu_buf *metal_scratch(qw36_gpu_ctx *ctx, qw36_gpu_buf **slot,
                                   size_t bytes, qw36_dtype dtype)
{
    if (!ctx || !slot) return NULL;
    if (*slot && (*slot)->bytes >= bytes && (*slot)->dtype == dtype)
        return *slot;
    metal_release_buf(*slot);
    *slot = metal_alloc(ctx, bytes, dtype);
    return *slot;
}

static void metal_zero_output(qw36_gpu_ctx *ctx, qw36_gpu_buf *buf, size_t bytes)
{
    if (!buf || !buf->mtl) return;
    metal_flush_batch(ctx);
    if (bytes > buf->bytes) bytes = buf->bytes;
    if (!bytes) return;
    memset([buf->mtl contents], 0, bytes);
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
    int owns_cb = 0;
    id<MTLCommandBuffer> cb = metal_cb_for_op(ctx, &owns_cb);
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    bind(enc);
    NSUInteger tg = pipe.maxTotalThreadsPerThreadgroup;
    if (tg > 256) tg = 256;
    if (tg > n) tg = n;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
  threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    if (owns_cb) { [cb commit]; [cb waitUntilCompleted]; }
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
        if (!w_f32) {
            metal_zero_output(ctx, y, (size_t)batch * rows * sizeof(float));
            return;
        }
        qw36_gpu_buf *w_tmp = metal_upload(ctx, w_f32,
                                           (size_t)rows * cols * sizeof(float),
                                           QW36_DTYPE_F32);
        free(w_f32);
        if (!w_tmp) {
            metal_zero_output(ctx, y, (size_t)batch * rows * sizeof(float));
            return;
        }
        metal_matmul(ctx, y, x, w_tmp, batch, rows, cols);
        metal_free(ctx, w_tmp);
        return;
    }

    if (ctx && batch == 1 && y->dtype == QW36_DTYPE_F32 &&
        x->dtype == QW36_DTYPE_F32 && w->dtype == QW36_DTYPE_F32) {
        /* Cache the MPSMatrixVectorMultiplication kernel by shape so we
         * don't re-create it on every call (lookup is ~3-5× cheaper than
         * alloc+init). The descriptor / MPSMatrix / MPSVector objects are
         * cheap to recreate per call. */
        NSString *key = [NSString stringWithFormat:@"%ux%u",
                         (unsigned)rows, (unsigned)cols];
        MPSMatrixVectorMultiplication *gemv = ctx->mps_cache[key];
        if (!gemv) {
            gemv = [[MPSMatrixVectorMultiplication alloc]
                       initWithDevice:ctx->device rows:rows columns:cols];
            ctx->mps_cache[key] = gemv;
        }
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
        int owns_cb = 0;
        id<MTLCommandBuffer> cb = metal_cb_for_op(ctx, &owns_cb);
        [gemv encodeToCommandBuffer:cb inputMatrix:wm inputVector:xv resultVector:yv];
        if (owns_cb) { [cb commit]; [cb waitUntilCompleted]; }
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
    qw36_gpu_buf *q = metal_scratch(ctx, &ctx->attn_q_scratch,
        (size_t)q_len * sizeof(float), QW36_DTYPE_F32);
    qw36_gpu_buf *k = metal_scratch(ctx, &ctx->attn_k_scratch,
        (size_t)kv_len * sizeof(float), QW36_DTYPE_F32);
    qw36_gpu_buf *v = metal_scratch(ctx, &ctx->attn_v_scratch,
        (size_t)kv_len * sizeof(float), QW36_DTYPE_F32);
    qw36_gpu_buf *scores = metal_scratch(ctx, &ctx->attn_scores_scratch,
        (size_t)n_heads * positions * sizeof(float), QW36_DTYPE_F32);
    if (!q || !k || !v || !scores) return;

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
}

static void metal_swiglu(qw36_gpu_ctx *ctx, qw36_gpu_buf *y, qw36_gpu_buf *x,
                         qw36_gpu_buf *w_gate, qw36_gpu_buf *w_up, qw36_gpu_buf *w_down,
                         uint32_t hidden, uint32_t inter)
{
    if (!ctx || !y || !x || !w_gate || !w_up || !w_down) return;
    qw36_gpu_buf *gate = metal_scratch(ctx, &ctx->swiglu_gate_scratch,
        (size_t)inter * sizeof(float), QW36_DTYPE_F32);
    qw36_gpu_buf *up = metal_scratch(ctx, &ctx->swiglu_up_scratch,
        (size_t)inter * sizeof(float), QW36_DTYPE_F32);
    if (!gate || !up) return;
    metal_matmul(ctx, gate, x, w_gate, 1, inter, hidden);
    metal_matmul(ctx, up,   x, w_up,   1, inter, hidden);
    metal_dispatch_1d(ctx, ctx->silu_mul, inter, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:gate->mtl offset:0 atIndex:0];
        [enc setBuffer:up->mtl   offset:0 atIndex:1];
        [enc setBytes:&inter length:sizeof(inter) atIndex:2];
    });
    metal_matmul(ctx, y, gate, w_down, 1, hidden, inter);
}

static void metal_dn_conv1d_silu(qw36_gpu_ctx *ctx, qw36_gpu_buf *y, qw36_gpu_buf *x,
                                 qw36_gpu_buf *conv_w, qw36_gpu_buf *conv_state,
                                 uint32_t channels, uint32_t kernel_size)
{
    if (!ctx || !y || !x || !conv_w || !conv_state) return;
    metal_dispatch_1d(ctx, ctx->dn_conv1d_silu, channels, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:y->mtl          offset:0 atIndex:0];
        [enc setBuffer:x->mtl          offset:0 atIndex:1];
        [enc setBuffer:conv_w->mtl     offset:0 atIndex:2];
        [enc setBuffer:conv_state->mtl offset:0 atIndex:3];
        [enc setBytes:&channels    length:sizeof(channels)    atIndex:4];
        [enc setBytes:&kernel_size length:sizeof(kernel_size) atIndex:5];
    });
}

static void metal_dn_gated_delta(qw36_gpu_ctx *ctx, qw36_gpu_buf *y,
                                 qw36_gpu_buf *qkv, qw36_gpu_buf *beta_raw,
                                 qw36_gpu_buf *alpha_raw, qw36_gpu_buf *dt_bias,
                                 qw36_gpu_buf *a_log, qw36_gpu_buf *state,
                                 uint32_t n_key, uint32_t n_value,
                                 uint32_t key_dim, uint32_t val_dim)
{
    if (!ctx || !y || !qkv || !beta_raw || !alpha_raw || !dt_bias ||
        !a_log || !state || !n_key || !n_value || !key_dim || !val_dim)
        return;
    if (key_dim % 32 != 0) return;

    qw36_gpu_buf *q = metal_scratch(ctx, &ctx->dn_q_scratch,
        (size_t)n_key * key_dim * sizeof(float), QW36_DTYPE_F32);
    qw36_gpu_buf *k = metal_scratch(ctx, &ctx->dn_k_scratch,
        (size_t)n_key * key_dim * sizeof(float), QW36_DTYPE_F32);
    qw36_gpu_buf *v = metal_scratch(ctx, &ctx->dn_v_scratch,
        (size_t)n_value * val_dim * sizeof(float), QW36_DTYPE_F32);
    qw36_gpu_buf *g = metal_scratch(ctx, &ctx->dn_g_scratch,
        (size_t)n_value * sizeof(float), QW36_DTYPE_F32);
    qw36_gpu_buf *beta = metal_scratch(ctx, &ctx->dn_beta_scratch,
        (size_t)n_value * sizeof(float), QW36_DTYPE_F32);
    qw36_gpu_buf *y_grouped = metal_scratch(ctx, &ctx->dn_y_grouped_scratch,
        (size_t)n_value * val_dim * sizeof(float), QW36_DTYPE_F32);
    if (!q || !k || !v || !g || !beta || !y_grouped) return;

    NSUInteger prep_n = (NSUInteger)n_key * key_dim;
    NSUInteger v_n = (NSUInteger)n_value * val_dim;
    if (v_n > prep_n) prep_n = v_n;
    metal_dispatch_1d(ctx, ctx->dn_prep_gdr, prep_n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:q->mtl         offset:0 atIndex:0];
        [enc setBuffer:k->mtl         offset:0 atIndex:1];
        [enc setBuffer:v->mtl         offset:0 atIndex:2];
        [enc setBuffer:g->mtl         offset:0 atIndex:3];
        [enc setBuffer:beta->mtl      offset:0 atIndex:4];
        [enc setBuffer:qkv->mtl       offset:0 atIndex:5];
        [enc setBuffer:alpha_raw->mtl offset:0 atIndex:6];
        [enc setBuffer:beta_raw->mtl  offset:0 atIndex:7];
        [enc setBuffer:dt_bias->mtl   offset:0 atIndex:8];
        [enc setBuffer:a_log->mtl     offset:0 atIndex:9];
        [enc setBytes:&n_key    length:sizeof(n_key)    atIndex:10];
        [enc setBytes:&n_value  length:sizeof(n_value)  atIndex:11];
        [enc setBytes:&key_dim  length:sizeof(key_dim)  atIndex:12];
        [enc setBytes:&val_dim  length:sizeof(val_dim)  atIndex:13];
    });

    uint32_t T = 1;
    int owns_cb = 0;
    id<MTLCommandBuffer> cb = metal_cb_for_op(ctx, &owns_cb);
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx->dn_gated_delta];
    [enc setBuffer:q->mtl         offset:0 atIndex:0];
    [enc setBuffer:k->mtl         offset:0 atIndex:1];
    [enc setBuffer:v->mtl         offset:0 atIndex:2];
    [enc setBuffer:g->mtl         offset:0 atIndex:3];
    [enc setBuffer:beta->mtl      offset:0 atIndex:4];
    [enc setBuffer:state->mtl     offset:0 atIndex:5];
    [enc setBuffer:y_grouped->mtl offset:0 atIndex:6];
    [enc setBuffer:state->mtl     offset:0 atIndex:7];
    [enc setBytes:&T        length:sizeof(T)        atIndex:8];
    [enc setBytes:&n_key    length:sizeof(n_key)    atIndex:9];
    [enc setBytes:&n_value  length:sizeof(n_value)  atIndex:10];
    [enc setBytes:&key_dim  length:sizeof(key_dim)  atIndex:11];
    [enc setBytes:&val_dim  length:sizeof(val_dim)  atIndex:12];
    [enc dispatchThreads:MTLSizeMake(32, val_dim, n_value)
  threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc endEncoding];
    if (owns_cb) { [cb commit]; [cb waitUntilCompleted]; }

    metal_dispatch_1d(ctx, ctx->dn_reorder_gdr_y, v_n, ^(id<MTLComputeCommandEncoder> renc) {
        [renc setBuffer:y->mtl         offset:0 atIndex:0];
        [renc setBuffer:y_grouped->mtl offset:0 atIndex:1];
        [renc setBytes:&n_key   length:sizeof(n_key)   atIndex:2];
        [renc setBytes:&n_value length:sizeof(n_value) atIndex:3];
        [renc setBytes:&val_dim length:sizeof(val_dim) atIndex:4];
    });
}

static void metal_dn_gated_rmsnorm(qw36_gpu_ctx *ctx, qw36_gpu_buf *y,
                                   qw36_gpu_buf *x, qw36_gpu_buf *z,
                                   qw36_gpu_buf *weight,
                                   uint32_t n_value, uint32_t val_dim,
                                   float eps)
{
    if (!ctx || !y || !x || !z || !weight) return;
    NSUInteger n = (NSUInteger)n_value * (NSUInteger)val_dim;
    metal_dispatch_1d(ctx, ctx->dn_gated_rmsnorm, n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:y->mtl      offset:0 atIndex:0];
        [enc setBuffer:x->mtl      offset:0 atIndex:1];
        [enc setBuffer:z->mtl      offset:0 atIndex:2];
        [enc setBuffer:weight->mtl offset:0 atIndex:3];
        [enc setBytes:&n_value length:sizeof(n_value) atIndex:4];
        [enc setBytes:&val_dim length:sizeof(val_dim) atIndex:5];
        [enc setBytes:&eps     length:sizeof(eps)     atIndex:6];
    });
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
        if (!row) {
            metal_zero_output(ctx, y, (size_t)hidden * sizeof(float));
            return;
        }
        if (metal_dequant_row(embed, token, hidden, row) == 0) {
            memcpy([y->mtl contents], row, (size_t)hidden * sizeof(float));
        } else {
            metal_zero_output(ctx, y, (size_t)hidden * sizeof(float));
        }
        free(row);
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
    .begin_batch       = metal_begin_batch,
    .end_batch         = metal_end_batch,
    .upload            = metal_upload,
    .download          = metal_download,
    .copy_from_host    = metal_copy_from_host,
    .alloc             = metal_alloc,
    .free              = metal_free,
    .rmsnorm           = metal_rmsnorm,
    .matmul            = metal_matmul,
    .attention         = metal_attention,
    .swiglu_mlp        = metal_swiglu,
    .dn_conv1d_silu    = metal_dn_conv1d_silu,
    .dn_gated_delta    = metal_dn_gated_delta,
    .dn_gated_rmsnorm  = metal_dn_gated_rmsnorm,
    .moe_forward       = NULL,
    .residual_add      = metal_residual_add,
    .embedding_lookup  = metal_embedding_lookup,
};

qw36_gpu_backend *qw36_backend_create(void) { return &g_metal_backend; }
