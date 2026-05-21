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

typedef struct {
    uint64_t calls;
    double total_gpu_us;
    double total_cpu_us;
    double max_gpu_us;
    double max_cpu_us;
} metal_perf_stat;

struct qw36_gpu_ctx {
    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary>      library;
    id<MTLComputePipelineState> rmsnorm;
    id<MTLComputePipelineState> residual_rmsnorm;
    id<MTLComputePipelineState> matmul;
    id<MTLComputePipelineState> f32_to_f16;
    id<MTLComputePipelineState> f16_to_f32;
    id<MTLComputePipelineState> silu_mul;
    id<MTLComputePipelineState> moe_route;
    id<MTLComputePipelineState> moe_gate_up;
    id<MTLComputePipelineState> moe_down_combine;
    id<MTLComputePipelineState> moe_scale_add;
    id<MTLComputePipelineState> dn_conv1d_silu;
    id<MTLComputePipelineState> dn_prep_gdr;
    id<MTLComputePipelineState> dn_prep_gdr_conv1d;
    id<MTLComputePipelineState> dn_gated_delta;
    id<MTLComputePipelineState> dn_reorder_gdr_y;
    id<MTLComputePipelineState> dn_gated_rmsnorm;
    id<MTLComputePipelineState> dn_gated_rmsnorm_f16;
    id<MTLComputePipelineState> dn_gated_rmsnorm_matmul;
    id<MTLComputePipelineState> residual_add;
    id<MTLComputePipelineState> embedding_lookup;
    id<MTLComputePipelineState> head_norm_rope;
    id<MTLComputePipelineState> kv_append;
    id<MTLComputePipelineState> attn_scores;
    id<MTLComputePipelineState> attn_softmax;
    id<MTLComputePipelineState> attn_combine;
    id<MTLComputePipelineState> attn_prep_decode;
    id<MTLComputePipelineState> attn_score_combine_tg;
    id<MTLComputePipelineState> attn_decode_fused;
    id<MTLComputePipelineState> attn_decode_fused_f16kv;
    id<MTLComputePipelineState> attn_decode_fused_f16kv_x4;
    id<MTLComputePipelineState> attn_decode_flash_f16kv;  /* opt-in flash-attn single-pass */
    /* GGUF-native quantised matmul (port of agent-infer's
     * gguf_q*_k_matmul). One threadgroup per output row, on-the-fly
     * dequant per element — replaces the f32→f16 MPS round-trip when the
     * weight is kept as Q4_K / Q5_K / Q6_K / Q8_0. */
    /* Cache for the f32→f16 input conversion that prefaces every MPS
     * fp16 gemv. Within a layer the same x_rms feeds q/k/v (and gate/up),
     * so converting once + reusing saves 2-3 dispatches per layer. The
     * cache is invalidated by every non-matmul compute kernel
     * (metal_dispatch_1d) since those may have written to x. */
    id<MTLBuffer> matmul_xh_src;
    NSUInteger    matmul_xh_cols;

    id<MTLComputePipelineState> matmul_qmv_quad_f16;
    id<MTLComputePipelineState> matmul_mma_f16;
    id<MTLComputePipelineState> matmul_q4_k;
    id<MTLComputePipelineState> matmul_q4_k_quad;
    id<MTLComputePipelineState> matmul_q4k_affine32;
    id<MTLComputePipelineState> matmul_q4k_affine32_mlx;
    id<MTLComputePipelineState> matmul_q5k_affine32;
    id<MTLComputePipelineState> matmul_q5k_affine32_mlx;
    id<MTLComputePipelineState> matmul_q6k_scale16;
    id<MTLComputePipelineState> matmul_q6k_scale16_mlx;
    id<MTLComputePipelineState> matmul_q5_k;
    id<MTLComputePipelineState> matmul_q6_k;
    id<MTLComputePipelineState> matmul_q8_0;
    /* Cache of MPSMatrixVectorMultiplication objects keyed by
     * "<rows>x<cols>" — building one of these per matmul has measurable
     * cost (Metal shader compile lookup); cache cuts the per-call overhead
     * by 2-3x and is essential to get past ~10 tok/s on this kind of
     * decode workload. */
    NSMutableDictionary<NSString *, MPSMatrixVectorMultiplication *> *mps_cache;
    NSMutableDictionary<NSString *, MPSMatrix *> *mps_matrix_cache;
    NSMutableDictionary<NSValue *, NSString *> *pipeline_labels;
    NSMutableDictionary<NSNumber *, NSMutableArray<id<MTLBuffer>> *> *buf_pool;

    /* When non-nil, every metal op encodes into this command buffer
     * instead of creating + committing one per call. Set by begin_batch
     * and cleared by end_batch (after commit + waitUntilCompleted). This
     * eliminates the ~200 sync points per token that otherwise dominate
     * decode latency. */
    id<MTLCommandBuffer> batch_cb;
    id<MTLComputeCommandEncoder> batch_compute_encoder;
    BOOL batch_active;
    BOOL perf_enabled;
    NSMutableDictionary<NSString *, NSValue *> *perf_stats;

    qw36_gpu_buf *matmul_x_f16_scratch;
    qw36_gpu_buf *matmul_y_f16_scratch;
    qw36_gpu_buf *moe_router_scratch;
    qw36_gpu_buf *moe_probs_scratch;
    qw36_gpu_buf *moe_idx_scratch;
    qw36_gpu_buf *moe_act_scratch;
    qw36_gpu_buf *moe_shared_scratch;
    qw36_gpu_buf *moe_shared_gate_scratch;
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

static void metal_flush_compute_encoder(qw36_gpu_ctx *ctx);
static double metal_perf_now_us(void);
static void metal_perf_record(qw36_gpu_ctx *ctx, NSString *label,
                              double cpu_us, double gpu_us);
static void metal_perf_print(qw36_gpu_ctx *ctx);
static void metal_perf_free(qw36_gpu_ctx *ctx);
static int metal_commit_wait_profile(qw36_gpu_ctx *ctx, id<MTLCommandBuffer> cb,
                                     int owns_cb, NSString *label,
                                     double start_us);
static NSString *metal_pipeline_label(qw36_gpu_ctx *ctx,
                                      id<MTLComputePipelineState> pipe);

static void metal_release_buf(qw36_gpu_buf *buf)
{
    if (!buf) return;
    buf->mtl = nil;
    free(buf->host_copy);
    free(buf);
}

static double metal_perf_now_us(void)
{
    return CFAbsoluteTimeGetCurrent() * 1000000.0;
}

static void metal_perf_record(qw36_gpu_ctx *ctx, NSString *label,
                              double cpu_us, double gpu_us)
{
    if (!ctx || !ctx->perf_enabled || !label) return;
    if (!ctx->perf_stats)
        ctx->perf_stats = [[NSMutableDictionary alloc] init];
    NSValue *val = ctx->perf_stats[label];
    metal_perf_stat *st = val ? (metal_perf_stat *)[val pointerValue] : NULL;
    if (!st) {
        st = (metal_perf_stat *)calloc(1, sizeof(*st));
        if (!st) return;
        ctx->perf_stats[label] = [NSValue valueWithPointer:st];
    }
    st->calls++;
    st->total_cpu_us += cpu_us;
    st->total_gpu_us += gpu_us;
    if (cpu_us > st->max_cpu_us) st->max_cpu_us = cpu_us;
    if (gpu_us > st->max_gpu_us) st->max_gpu_us = gpu_us;
}

static void metal_perf_print(qw36_gpu_ctx *ctx)
{
    if (!ctx || !ctx->perf_enabled || !ctx->perf_stats.count) return;
    NSArray<NSString *> *keys = [ctx->perf_stats allKeys];
    keys = [keys sortedArrayUsingComparator:^NSComparisonResult(NSString *a,
                                                                NSString *b) {
        metal_perf_stat *sa = (metal_perf_stat *)[ctx->perf_stats[a] pointerValue];
        metal_perf_stat *sb = (metal_perf_stat *)[ctx->perf_stats[b] pointerValue];
        double ga = sa ? sa->total_gpu_us : 0.0;
        double gb = sb ? sb->total_gpu_us : 0.0;
        if (ga < gb) return NSOrderedDescending;
        if (ga > gb) return NSOrderedAscending;
        return [a compare:b];
    }];

    double total_gpu = 0.0, total_cpu = 0.0;
    uint64_t total_calls = 0;
    for (NSString *k in keys) {
        metal_perf_stat *st = (metal_perf_stat *)[ctx->perf_stats[k] pointerValue];
        if (!st) continue;
        total_calls += st->calls;
        total_gpu += st->total_gpu_us;
        total_cpu += st->total_cpu_us;
    }

    fprintf(stderr,
            "[metal-perf] standalone profiling mode: %llu calls, gpu %.3fms, cpu-wall %.3fms\n",
            (unsigned long long)total_calls, total_gpu / 1000.0,
            total_cpu / 1000.0);
    fprintf(stderr,
            "[metal-perf] %-38s %8s %10s %10s %10s %10s\n",
            "pipeline", "calls", "gpu_ms", "avg_us", "max_us", "cpu_ms");
    for (NSString *k in keys) {
        metal_perf_stat *st = (metal_perf_stat *)[ctx->perf_stats[k] pointerValue];
        if (!st || !st->calls) continue;
        fprintf(stderr,
                "[metal-perf] %-38s %8llu %10.3f %10.2f %10.2f %10.3f\n",
                [k UTF8String],
                (unsigned long long)st->calls,
                st->total_gpu_us / 1000.0,
                st->total_gpu_us / (double)st->calls,
                st->max_gpu_us,
                st->total_cpu_us / 1000.0);
    }
}

static void metal_perf_free(qw36_gpu_ctx *ctx)
{
    if (!ctx || !ctx->perf_stats) return;
    for (NSString *k in ctx->perf_stats) {
        metal_perf_stat *st =
            (metal_perf_stat *)[ctx->perf_stats[k] pointerValue];
        free(st);
    }
    ctx->perf_stats = nil;
}

static NSString *metal_pipeline_label(qw36_gpu_ctx *ctx,
                                      id<MTLComputePipelineState> pipe)
{
    if (!ctx || !pipe) return @"compute";
    NSString *label = ctx->pipeline_labels[[NSValue valueWithNonretainedObject:pipe]];
    return label ? label : @"compute";
}

static int metal_commit_wait_profile(qw36_gpu_ctx *ctx, id<MTLCommandBuffer> cb,
                                     int owns_cb, NSString *label,
                                     double start_us)
{
    if (!cb) return -1;
    if (!owns_cb) return 0;
    [cb commit];
    [cb waitUntilCompleted];
    if (ctx && ctx->perf_enabled && label) {
        double cpu_us = metal_perf_now_us() - start_us;
        double gpu_us = 0.0;
        if ([cb respondsToSelector:@selector(GPUStartTime)] &&
            [cb respondsToSelector:@selector(GPUEndTime)]) {
            CFTimeInterval gs = [cb GPUStartTime];
            CFTimeInterval ge = [cb GPUEndTime];
            if (ge > gs) gpu_us = (ge - gs) * 1000000.0;
        }
        metal_perf_record(ctx, label, cpu_us, gpu_us);
    }
    return [cb status] == MTLCommandBufferStatusError ? -1 : 0;
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
           dtype == QW36_DTYPE_Q4K_AFFINE32 ||
           dtype == QW36_DTYPE_Q5_K ||
           dtype == QW36_DTYPE_Q5K_AFFINE32 ||
           dtype == QW36_DTYPE_Q6_K ||
           dtype == QW36_DTYPE_Q6K_SCALE16 ||
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

static void metal_dq_q4k_affine32(const uint8_t *blocks, float *out, size_t n)
{
    const size_t nb = n / QW36_QK_K;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t *b = blocks + i * 160;
        const uint16_t *scales = (const uint16_t *)b;
        const uint16_t *biases = (const uint16_t *)(b + 16);
        const uint8_t *qs = b + 32;
        for (int sub = 0; sub < 8; sub++) {
            const float scale = metal_f16_to_f32(scales[sub]);
            const float bias = metal_f16_to_f32(biases[sub]);
            const uint8_t *qg = qs + sub * 16;
            for (int j = 0; j < 16; j++) {
                uint8_t byte = qg[j];
                *out++ = scale * (float)(byte & 0x0F) + bias;
                *out++ = scale * (float)(byte >> 4) + bias;
            }
        }
    }
}

static uint8_t metal_unpack_5bit8(const uint8_t *src, int idx)
{
    uint64_t bits = 0;
    for (int i = 0; i < 5; i++)
        bits |= ((uint64_t)src[i]) << (8 * i);
    return (uint8_t)((bits >> (5 * idx)) & 31u);
}

static void metal_dq_q5k_affine32(const uint8_t *blocks, float *out, size_t n)
{
    const size_t nb = n / QW36_QK_K;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t *b = blocks + i * 192;
        const uint16_t *scales = (const uint16_t *)b;
        const uint16_t *biases = (const uint16_t *)(b + 16);
        const uint8_t *qs = b + 32;
        for (int sub = 0; sub < 8; sub++) {
            const float scale = metal_f16_to_f32(scales[sub]);
            const float bias = metal_f16_to_f32(biases[sub]);
            const uint8_t *qg = qs + sub * 20;
            for (int p = 0; p < 4; p++) {
                const uint8_t *pack = qg + p * 5;
                for (int j = 0; j < 8; j++)
                    *out++ = scale * (float)metal_unpack_5bit8(pack, j) + bias;
            }
        }
    }
}

static uint8_t metal_unpack_6bit16(const uint8_t *src, int idx)
{
    int bit = 6 * idx;
    int byte = bit >> 3;
    int shift = bit & 7;
    uint16_t x = (uint16_t)src[byte];
    if (byte + 1 < 12)
        x |= (uint16_t)src[byte + 1] << 8;
    return (uint8_t)((x >> shift) & 63u);
}

static void metal_dq_q6k_scale16(const uint8_t *blocks, float *out, size_t n)
{
    const size_t nb = n / QW36_QK_K;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t *b = blocks + i * 224;
        const uint16_t *scales = (const uint16_t *)b;
        const uint8_t *qs = b + 32;
        for (int group = 0; group < 16; group++) {
            const float scale = metal_f16_to_f32(scales[group]);
            const uint8_t *qg = qs + group * 12;
            for (int j = 0; j < 16; j++)
                *out++ = scale * ((float)metal_unpack_6bit16(qg, j) - 32.0f);
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
        case QW36_DTYPE_Q4K_AFFINE32: *qk = 256; *bytes_per_block = 160; return 0;
        case QW36_DTYPE_Q5_K: *qk = 256; *bytes_per_block = 176; return 0;
        case QW36_DTYPE_Q5K_AFFINE32: *qk = 256; *bytes_per_block = 192; return 0;
        case QW36_DTYPE_Q6_K: *qk = 256; *bytes_per_block = 210; return 0;
        case QW36_DTYPE_Q6K_SCALE16: *qk = 256; *bytes_per_block = 224; return 0;
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
        case QW36_DTYPE_Q4K_AFFINE32: metal_dq_q4k_affine32(row, out, cols); return 0;
        case QW36_DTYPE_Q5_K: metal_dq_q5_K(row, out, cols); return 0;
        case QW36_DTYPE_Q5K_AFFINE32: metal_dq_q5k_affine32(row, out, cols); return 0;
        case QW36_DTYPE_Q6_K: metal_dq_q6_K(row, out, cols); return 0;
        case QW36_DTYPE_Q6K_SCALE16: metal_dq_q6k_scale16(row, out, cols); return 0;
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
    if (pipe && ctx->pipeline_labels)
        ctx->pipeline_labels[[NSValue valueWithNonretainedObject:pipe]] = name;
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
    ctx->mps_matrix_cache = [[NSMutableDictionary alloc] init];
    ctx->pipeline_labels = [[NSMutableDictionary alloc] init];
    ctx->perf_enabled = getenv("QW36_METAL_PERF") != NULL;
    if (ctx->perf_enabled)
        ctx->perf_stats = [[NSMutableDictionary alloc] init];

    ctx->library = metal_load_library(ctx->device, err, err_cap);
    if (!ctx->library) {
        ctx->queue = nil;
        ctx->device = nil;
        free(ctx);
        return NULL;
    }

    ctx->rmsnorm          = metal_make_pipeline(ctx, @"qw36_rmsnorm_f32", err, err_cap);
    ctx->matmul           = metal_make_pipeline(ctx, @"qw36_matmul_f32", err, err_cap);
    ctx->f32_to_f16       = metal_make_pipeline(ctx, @"qw36_f32_to_f16_f32", err, err_cap);
    ctx->f16_to_f32       = metal_make_pipeline(ctx, @"qw36_f16_to_f32_f32", err, err_cap);
    ctx->silu_mul         = metal_make_pipeline(ctx, @"qw36_silu_mul_f32", err, err_cap);
    ctx->moe_route        = metal_make_pipeline(ctx, @"qw36_moe_route_f32", err, err_cap);
    ctx->moe_gate_up      = metal_make_pipeline(ctx, @"qw36_moe_gate_up_f32", err, err_cap);
    ctx->moe_down_combine = metal_make_pipeline(ctx, @"qw36_moe_down_combine_f32", err, err_cap);
    ctx->moe_scale_add    = metal_make_pipeline(ctx, @"qw36_moe_scale_add_f32", err, err_cap);
    ctx->dn_conv1d_silu   = metal_make_pipeline(ctx, @"qw36_dn_conv1d_silu_f32", err, err_cap);
    ctx->dn_prep_gdr      = metal_make_pipeline(ctx, @"qw36_compute_g_beta_norm_qk", err, err_cap);
    ctx->dn_prep_gdr_conv1d = metal_make_pipeline(ctx, @"qw36_compute_g_beta_norm_qk_conv1d", err, err_cap);
    ctx->dn_gated_delta   = metal_make_pipeline(ctx, @"qw36_gated_delta_step_f32", err, err_cap);
    ctx->dn_reorder_gdr_y = metal_make_pipeline(ctx, @"qw36_dn_reorder_grouped_y_to_raw_f32", err, err_cap);
    ctx->dn_gated_rmsnorm = metal_make_pipeline(ctx, @"qw36_dn_gated_rmsnorm_f32", err, err_cap);
    ctx->dn_gated_rmsnorm_f16 = metal_make_pipeline(ctx, @"qw36_dn_gated_rmsnorm_f16_f32", err, err_cap);
    ctx->dn_gated_rmsnorm_matmul = metal_make_pipeline(ctx, @"qw36_dn_gated_rmsnorm_matmul_f32", err, err_cap);
    ctx->residual_add     = metal_make_pipeline(ctx, @"qw36_residual_add_f32", err, err_cap);
    ctx->residual_rmsnorm = metal_make_pipeline(ctx, @"qw36_residual_rmsnorm_f32", err, err_cap);
    ctx->embedding_lookup = metal_make_pipeline(ctx, @"qw36_embedding_lookup_f32", err, err_cap);
    ctx->head_norm_rope   = metal_make_pipeline(ctx, @"qw36_head_norm_rope_f32", err, err_cap);
    ctx->kv_append        = metal_make_pipeline(ctx, @"qw36_kv_append_f32", err, err_cap);
    ctx->attn_scores      = metal_make_pipeline(ctx, @"qw36_attn_scores_f32", err, err_cap);
    ctx->attn_softmax     = metal_make_pipeline(ctx, @"qw36_attn_softmax_f32", err, err_cap);
    ctx->attn_combine     = metal_make_pipeline(ctx, @"qw36_attn_combine_f32", err, err_cap);
    ctx->attn_prep_decode = metal_make_pipeline(ctx, @"qw36_attn_prep_decode_f32", err, err_cap);
    ctx->attn_score_combine_tg = metal_make_pipeline(ctx, @"qw36_attn_score_combine_tg_f32", err, err_cap);
    ctx->attn_decode_fused = metal_make_pipeline(ctx, @"qw36_attn_decode_fused_f32", err, err_cap);
    ctx->attn_decode_fused_f16kv = metal_make_pipeline(ctx, @"qw36_attn_decode_fused_f16kv_f32", err, err_cap);
    ctx->attn_decode_fused_f16kv_x4 = metal_make_pipeline(ctx, @"qw36_attn_decode_fused_f16kv_x4_f32", err, err_cap);
    ctx->attn_decode_flash_f16kv = metal_make_pipeline(ctx, @"qw36_attn_decode_flash_f16kv_f32", err, err_cap);
    ctx->matmul_qmv_quad_f16 = metal_make_pipeline(ctx, @"qw36_matmul_qmv_quad_f16", err, err_cap);
    ctx->matmul_mma_f16 = metal_make_pipeline(ctx, @"qw36_matmul_mma_f16_f32", err, err_cap);
    ctx->matmul_q4_k = metal_make_pipeline(ctx, @"qw36_matmul_q4_k_f32", err, err_cap);
    ctx->matmul_q4_k_quad = metal_make_pipeline(ctx, @"qw36_matmul_q4_k_qmv_quad_f32", err, err_cap);
    ctx->matmul_q4k_affine32 = metal_make_pipeline(ctx, @"qw36_matmul_q4k_affine32_qmv_fast_f32", err, err_cap);
    ctx->matmul_q4k_affine32_mlx = metal_make_pipeline(ctx, @"qw36_matmul_q4k_affine32_qmv_mlx_f32", err, err_cap);
    ctx->matmul_q5k_affine32 = metal_make_pipeline(ctx, @"qw36_matmul_q5k_affine32_qmv_fast_f32", err, err_cap);
    ctx->matmul_q5k_affine32_mlx = metal_make_pipeline(ctx, @"qw36_matmul_q5k_affine32_qmv_mlx_f32", err, err_cap);
    ctx->matmul_q6k_scale16 = metal_make_pipeline(ctx, @"qw36_matmul_q6k_scale16_qmv_fast_f32", err, err_cap);
    ctx->matmul_q6k_scale16_mlx = metal_make_pipeline(ctx, @"qw36_matmul_q6k_scale16_qmv_mlx_f32", err, err_cap);
    ctx->matmul_q5_k = metal_make_pipeline(ctx, @"qw36_matmul_q5_k_f32", err, err_cap);
    ctx->matmul_q6_k = metal_make_pipeline(ctx, @"qw36_matmul_q6_k_f32", err, err_cap);
    ctx->matmul_q8_0 = metal_make_pipeline(ctx, @"qw36_matmul_q8_0_f32", err, err_cap);

    if (!ctx->rmsnorm || !ctx->matmul || !ctx->f32_to_f16 ||
        !ctx->f16_to_f32 || !ctx->silu_mul ||
        !ctx->moe_route || !ctx->moe_gate_up || !ctx->moe_down_combine ||
        !ctx->moe_scale_add ||
        !ctx->dn_conv1d_silu || !ctx->dn_gated_delta ||
        !ctx->dn_prep_gdr || !ctx->dn_prep_gdr_conv1d ||
        !ctx->dn_reorder_gdr_y ||
        !ctx->dn_gated_rmsnorm || !ctx->dn_gated_rmsnorm_f16 ||
        !ctx->dn_gated_rmsnorm_matmul ||
        !ctx->residual_add ||
        !ctx->embedding_lookup || !ctx->head_norm_rope || !ctx->kv_append ||
        !ctx->attn_scores || !ctx->attn_softmax || !ctx->attn_combine ||
        !ctx->attn_prep_decode || !ctx->attn_score_combine_tg ||
        !ctx->attn_decode_fused || !ctx->attn_decode_fused_f16kv ||
        !ctx->matmul_mma_f16 || !ctx->matmul_q4k_affine32 ||
        !ctx->matmul_q5k_affine32 || !ctx->matmul_q6k_scale16) {
        ctx->attn_decode_fused_f16kv = nil;
        ctx->attn_decode_fused = nil;
        ctx->attn_score_combine_tg = nil;
        ctx->matmul_q4k_affine32 = nil;
        ctx->matmul_q5k_affine32 = nil;
        ctx->matmul_q6k_scale16 = nil;
        ctx->matmul_mma_f16 = nil;
        ctx->attn_prep_decode = nil;
        ctx->attn_combine = nil;
        ctx->attn_softmax = nil;
        ctx->attn_scores = nil;
        ctx->kv_append = nil;
        ctx->head_norm_rope = nil;
        ctx->embedding_lookup = nil;
        ctx->residual_add = nil;
        ctx->dn_gated_rmsnorm_matmul = nil;
        ctx->dn_gated_rmsnorm_f16 = nil;
        ctx->dn_gated_rmsnorm = nil;
        ctx->dn_reorder_gdr_y = nil;
        ctx->dn_gated_delta = nil;
        ctx->dn_prep_gdr_conv1d = nil;
        ctx->dn_prep_gdr = nil;
        ctx->dn_conv1d_silu = nil;
        ctx->moe_down_combine = nil;
        ctx->moe_scale_add = nil;
        ctx->moe_gate_up = nil;
        ctx->moe_route = nil;
        ctx->silu_mul = nil;
        ctx->f16_to_f32 = nil;
        ctx->f32_to_f16 = nil;
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
        metal_flush_compute_encoder(ctx);
        [ctx->batch_cb commit];
        [ctx->batch_cb waitUntilCompleted];
        ctx->batch_cb = nil;
    }
    ctx->batch_compute_encoder = nil;
    metal_perf_print(ctx);
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
    metal_release_buf(ctx->matmul_y_f16_scratch);
    metal_release_buf(ctx->matmul_x_f16_scratch);
    metal_release_buf(ctx->moe_shared_gate_scratch);
    metal_release_buf(ctx->moe_shared_scratch);
    metal_release_buf(ctx->moe_act_scratch);
    metal_release_buf(ctx->moe_idx_scratch);
    metal_release_buf(ctx->moe_probs_scratch);
    metal_release_buf(ctx->moe_router_scratch);
    ctx->mps_cache = nil;
    ctx->mps_matrix_cache = nil;
    ctx->pipeline_labels = nil;
    ctx->buf_pool  = nil;
    metal_perf_free(ctx);
    ctx->attn_combine = nil;
    ctx->attn_prep_decode = nil;
    ctx->attn_score_combine_tg = nil;
    ctx->attn_decode_fused = nil;
    ctx->attn_decode_fused_f16kv = nil;
    ctx->attn_decode_fused_f16kv_x4 = nil;
    ctx->attn_decode_flash_f16kv = nil;
    ctx->matmul_mma_f16 = nil;
    ctx->matmul_q4k_affine32 = nil;
    ctx->matmul_q4k_affine32_mlx = nil;
    ctx->matmul_q5k_affine32 = nil;
    ctx->matmul_q5k_affine32_mlx = nil;
    ctx->matmul_q6k_scale16 = nil;
    ctx->matmul_q6k_scale16_mlx = nil;
    ctx->matmul_q4_k_quad = nil;
    ctx->attn_softmax = nil;
    ctx->attn_scores = nil;
    ctx->kv_append = nil;
    ctx->head_norm_rope = nil;
    ctx->embedding_lookup = nil;
    ctx->residual_add = nil;
    ctx->dn_gated_rmsnorm_matmul = nil;
    ctx->dn_gated_rmsnorm_f16 = nil;
    ctx->dn_gated_rmsnorm = nil;
    ctx->dn_reorder_gdr_y = nil;
    ctx->dn_gated_delta = nil;
    ctx->dn_prep_gdr_conv1d = nil;
    ctx->dn_prep_gdr = nil;
    ctx->dn_conv1d_silu = nil;
    ctx->moe_down_combine = nil;
    ctx->moe_scale_add = nil;
    ctx->moe_gate_up = nil;
    ctx->moe_route = nil;
    ctx->silu_mul = nil;
    ctx->f16_to_f32 = nil;
    ctx->f32_to_f16 = nil;
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
    if (ctx->perf_enabled) return;        /* per-op GPU profiling needs
                                          * standalone command buffers. */
    if (ctx->batch_active) return;       /* already in a batch */
    ctx->batch_active = YES;
    ctx->batch_cb = [ctx->queue commandBuffer];
    ctx->batch_compute_encoder = nil;
}

static void metal_end_batch(qw36_gpu_ctx *ctx)
{
    if (!ctx || !ctx->batch_active) return;
    ctx->batch_active = NO;
    if (ctx->batch_cb) {
        metal_flush_compute_encoder(ctx);
        id<MTLCommandBuffer> cb = ctx->batch_cb;
        ctx->batch_cb = nil;          /* clear *before* commit so re-entrant
                                       * ops within end_batch don't reuse it */
        /* Per-step timing (QW36_METAL_TIMING=1):
         *   commit ≈ 0.01ms (CPU encoding), gpu ≈ 11.9ms (waitUntilCompleted).
         * We are 100% GPU-bound; CPU dispatch overhead is negligible. To
         * reach 200 tok/s we need to actually reduce GPU work (kernel
         * fusion or faster GEMV), not lower dispatch overhead. */
        if (getenv("QW36_METAL_TIMING") != NULL) {
            static int hits = 0;
            static double t_encode = 0, t_gpu = 0;
            double t1 = CFAbsoluteTimeGetCurrent();
            [cb commit];
            double t2 = CFAbsoluteTimeGetCurrent();
            [cb waitUntilCompleted];
            double t3 = CFAbsoluteTimeGetCurrent();
            t_encode += (t2 - t1); t_gpu += (t3 - t2);
            if (++hits == 12 || hits == 64 || hits == 256) {
                fprintf(stderr, "[timing] hits=%d  commit=%.2fms gpu=%.2fms per-step\n",
                    hits, 1000.0*t_encode/hits, 1000.0*t_gpu/hits);
            }
        } else {
            [cb commit];
            [cb waitUntilCompleted];
        }
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

static void metal_flush_compute_encoder(qw36_gpu_ctx *ctx)
{
    if (!ctx || !ctx->batch_compute_encoder) return;
    [ctx->batch_compute_encoder endEncoding];
    ctx->batch_compute_encoder = nil;
}

static id<MTLComputeCommandEncoder>
metal_compute_encoder_for_op(qw36_gpu_ctx *ctx, id<MTLCommandBuffer> cb,
                             int owns_cb)
{
    if (!ctx || owns_cb || !ctx->batch_active) {
        return [cb computeCommandEncoder];
    }
    if (!ctx->batch_compute_encoder) {
        ctx->batch_compute_encoder = [cb computeCommandEncoder];
    }
    return ctx->batch_compute_encoder;
}

static void metal_finish_compute_encoder(qw36_gpu_ctx *ctx,
                                         id<MTLComputeCommandEncoder> enc,
                                         int owns_cb)
{
    if (!enc) return;
    if (ctx && ctx->batch_active && !owns_cb) return;
    [enc endEncoding];
}

static void metal_flush_batch(qw36_gpu_ctx *ctx)
{
    if (!ctx || !ctx->batch_cb) return;
    metal_flush_compute_encoder(ctx);
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
    /* Don't recycle buffers while a command buffer is still being encoded:
     * a later op in the same batch could otherwise alias a resource that an
     * earlier encoded op has not consumed yet. */
    if (ctx && buf->mtl && !ctx->batch_active) metal_pool_release(ctx, buf->mtl);
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

static void metal_invalidate_matmul_xh(qw36_gpu_ctx *ctx)
{
    if (!ctx) return;
    ctx->matmul_xh_src = nil;
    ctx->matmul_xh_cols = 0;
}

static void metal_dispatch_1d(qw36_gpu_ctx *ctx,
                              id<MTLComputePipelineState> pipe,
                              NSUInteger n,
                              void (^bind)(id<MTLComputeCommandEncoder> enc))
{
    if (!ctx || !pipe || n == 0) return;
    double start_us = ctx->perf_enabled ? metal_perf_now_us() : 0.0;
    int owns_cb = 0;
    id<MTLCommandBuffer> cb = metal_cb_for_op(ctx, &owns_cb);
    id<MTLComputeCommandEncoder> enc =
        metal_compute_encoder_for_op(ctx, cb, owns_cb);
    [enc setComputePipelineState:pipe];
    bind(enc);
    NSUInteger tg = pipe.maxTotalThreadsPerThreadgroup;
    if (tg > 256) tg = 256;
    if (tg > n) tg = n;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
  threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    metal_finish_compute_encoder(ctx, enc, owns_cb);
    metal_commit_wait_profile(ctx, cb, owns_cb,
                              metal_pipeline_label(ctx, pipe),
                              start_us);
}

static void metal_invalidate_matmul_xh(qw36_gpu_ctx *ctx);

static void metal_rmsnorm(qw36_gpu_ctx *ctx, qw36_gpu_buf *out, qw36_gpu_buf *x,
                          qw36_gpu_buf *w, uint32_t hidden, float eps)
{
    if (!out || !x || !w) return;
    /* rmsnorm writes to `out`, which is fed directly into the next matmul
     * — drop any cached fp16 input that was derived from a previous
     * residual stream snapshot. */
    metal_invalidate_matmul_xh(ctx);
    metal_dispatch_1d(ctx, ctx->rmsnorm, hidden, ^(id<MTLComputeCommandEncoder> enc) {
        uint32_t x_dtype = (uint32_t)x->dtype;
        uint32_t w_dtype = (uint32_t)w->dtype;
        uint32_t out_dtype = (uint32_t)out->dtype;
        [enc setBuffer:out->mtl offset:0 atIndex:0];
        [enc setBuffer:x->mtl   offset:0 atIndex:1];
        [enc setBuffer:w->mtl   offset:0 atIndex:2];
        [enc setBytes:&hidden length:sizeof(hidden) atIndex:3];
        [enc setBytes:&eps    length:sizeof(eps)    atIndex:4];
        [enc setBytes:&x_dtype length:sizeof(x_dtype) atIndex:5];
        [enc setBytes:&w_dtype length:sizeof(w_dtype) atIndex:6];
        [enc setBytes:&out_dtype length:sizeof(out_dtype) atIndex:7];
    });
}

static MPSMatrix *metal_cached_weight_matrix(qw36_gpu_ctx *ctx,
                                             qw36_gpu_buf *w,
                                             uint32_t rows,
                                             uint32_t cols,
                                             MPSDataType data_type,
                                             NSUInteger elem_bytes,
                                             const char *suffix)
{
    if (!ctx || !w || !w->mtl) return nil;
    NSString *key = [NSString stringWithFormat:@"%p:%ux%u:%s",
                     (__bridge void *)w->mtl,
                     (unsigned)rows, (unsigned)cols, suffix];
    MPSMatrix *wm = ctx->mps_matrix_cache[key];
    if (wm) return wm;
    MPSMatrixDescriptor *desc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                              columns:cols
                                             rowBytes:(NSUInteger)cols * elem_bytes
                                             dataType:data_type];
    wm = [[MPSMatrix alloc] initWithBuffer:w->mtl descriptor:desc];
    ctx->mps_matrix_cache[key] = wm;
    return wm;
}

static int metal_matmul(qw36_gpu_ctx *ctx, qw36_gpu_buf *y, qw36_gpu_buf *x,
                        qw36_gpu_buf *w, uint32_t batch, uint32_t rows, uint32_t cols)
{
    if (!y || !x || !w) return -1;

    if (ctx && batch == 1 && y->dtype == QW36_DTYPE_F32 &&
        x->dtype == QW36_DTYPE_F32 &&
        (w->dtype == QW36_DTYPE_Q4K_AFFINE32 ||
         w->dtype == QW36_DTYPE_Q5K_AFFINE32 ||
         w->dtype == QW36_DTYPE_Q6K_SCALE16) &&
        w->mtl && (cols % 512u) == 0)
    {
        /* Q4K MLX-style qmv is faster than the bit-shift fast variant on
         * sustained essay benches (~+5% at n=256 in 2026-05-21 5-rep
         * median). Wash on short EOS-bounded prompts. Default it on; opt
         * out via QW36_METAL_Q4K_AFFINE32_MLX=0. */
        static int q4k_mlx_env_cached = -1;
        if (q4k_mlx_env_cached < 0) {
            const char *e = getenv("QW36_METAL_Q4K_AFFINE32_MLX");
            q4k_mlx_env_cached = e ? (atoi(e) != 0 ? 1 : 0) : 1;
        }
        /* Q6K MLX-style qmv is faster than the bit-shift fast variant on
         * this host (+3-4% e2e, see commit 7550375). Default it on; opt out
         * with QW36_METAL_Q6K_SCALE16_MLX=0 for bisecting. */
        static int q6k_mlx_env_cached = -1;
        if (q6k_mlx_env_cached < 0) {
            const char *e = getenv("QW36_METAL_Q6K_SCALE16_MLX");
            q6k_mlx_env_cached = e ? (atoi(e) != 0 ? 1 : 0) : 1;
        }
        static int q5k_mlx_env_cached = -1;
        if (q5k_mlx_env_cached < 0) {
            const char *e = getenv("QW36_METAL_Q5K_AFFINE32_MLX");
            q5k_mlx_env_cached = (e && atoi(e) != 0) ? 1 : 0;
        }
        id<MTLComputePipelineState> affine_pipe = nil;
        NSString *affine_name = nil;
        if (w->dtype == QW36_DTYPE_Q4K_AFFINE32) {
            affine_pipe = q4k_mlx_env_cached && ctx->matmul_q4k_affine32_mlx
                ? ctx->matmul_q4k_affine32_mlx
                : ctx->matmul_q4k_affine32;
            affine_name = @"q4k_affine32";
        } else if (w->dtype == QW36_DTYPE_Q5K_AFFINE32) {
            affine_pipe = q5k_mlx_env_cached && ctx->matmul_q5k_affine32_mlx
                ? ctx->matmul_q5k_affine32_mlx
                : ctx->matmul_q5k_affine32;
            affine_name = q5k_mlx_env_cached && ctx->matmul_q5k_affine32_mlx
                ? @"q5k_affine32_mlx"
                : @"q5k_affine32";
        } else {
            affine_pipe = q6k_mlx_env_cached && ctx->matmul_q6k_scale16_mlx
                ? ctx->matmul_q6k_scale16_mlx
                : ctx->matmul_q6k_scale16;
            affine_name = q6k_mlx_env_cached && ctx->matmul_q6k_scale16_mlx
                ? @"q6k_scale16_mlx"
                : @"q6k_scale16";
        }
        if (affine_pipe) {
            double start_us = ctx->perf_enabled ? metal_perf_now_us() : 0.0;
            NSString *perf_label = ctx->perf_enabled
                ? [NSString stringWithFormat:@"%@_%ux%u",
                   affine_name, (unsigned)rows, (unsigned)cols]
                : nil;
            int owns_cb = 0;
            id<MTLCommandBuffer> cb = metal_cb_for_op(ctx, &owns_cb);
            id<MTLComputeCommandEncoder> enc =
                metal_compute_encoder_for_op(ctx, cb, owns_cb);
            [enc setComputePipelineState:affine_pipe];
            [enc setBuffer:y->mtl offset:0 atIndex:0];
            [enc setBuffer:x->mtl offset:0 atIndex:1];
            [enc setBuffer:w->mtl offset:0 atIndex:2];
            [enc setBytes:&cols length:sizeof(cols) atIndex:3];
            [enc setBytes:&rows length:sizeof(rows) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((rows + 15u) / 16u, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            metal_finish_compute_encoder(ctx, enc, owns_cb);
            return metal_commit_wait_profile(ctx, cb, owns_cb, perf_label, start_us);
        }
    }

    /* GGUF-native quantised matmul: when the weight is still in its packed
     * Q4_K / Q5_K / Q6_K / Q8_0 layout on the GPU, the per-row dequant kernel
     * does on-the-fly unpack + accumulation, skipping the fp16 / fp32 stage. */
    if (ctx && batch == 1 && y->dtype == QW36_DTYPE_F32 &&
        x->dtype == QW36_DTYPE_F32 && w->mtl) {
        id<MTLComputePipelineState> qpipe = nil;
        switch (w->dtype) {
            case QW36_DTYPE_Q4_K: qpipe = ctx->matmul_q4_k; break;
            case QW36_DTYPE_Q5_K: qpipe = ctx->matmul_q5_k; break;
            case QW36_DTYPE_Q6_K: qpipe = ctx->matmul_q6_k; break;
            case QW36_DTYPE_Q8_0: qpipe = ctx->matmul_q8_0; break;
            default: break;
        }
        if (qpipe) {
            static int q4k_quad = -1;
            if (q4k_quad < 0) {
                const char *e = getenv("QW36_METAL_Q4K_QUAD");
                q4k_quad = (e && atoi(e)) ? 1 : 0;
            }
            if (q4k_quad && w->dtype == QW36_DTYPE_Q4_K &&
                ctx->matmul_q4_k_quad && (cols % 256u) == 0) {
                double start_us = ctx->perf_enabled ? metal_perf_now_us() : 0.0;
                NSString *perf_label = ctx->perf_enabled
                    ? [NSString stringWithFormat:@"q4k_quad_%ux%u",
                       (unsigned)rows, (unsigned)cols]
                    : nil;
                int owns_cb = 0;
                id<MTLCommandBuffer> cb = metal_cb_for_op(ctx, &owns_cb);
                id<MTLComputeCommandEncoder> enc =
                    metal_compute_encoder_for_op(ctx, cb, owns_cb);
                [enc setComputePipelineState:ctx->matmul_q4_k_quad];
                [enc setBuffer:y->mtl offset:0 atIndex:0];
                [enc setBuffer:x->mtl offset:0 atIndex:1];
                [enc setBuffer:w->mtl offset:0 atIndex:2];
                [enc setBytes:&cols length:sizeof(cols) atIndex:3];
                [enc setBytes:&rows length:sizeof(rows) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(1, (rows + 63u) / 64u, 1)
                     threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                metal_finish_compute_encoder(ctx, enc, owns_cb);
                return metal_commit_wait_profile(ctx, cb, owns_cb, perf_label, start_us);
            }

            /* Q4_K kernel caches per-block scales in threadgroup memory
             * (18 floats per Q4_K block + 8 SIMD-partial floats). Others
             * still use the simple 256-float reduction scratch. */
            NSUInteger tg_bytes = 256 * sizeof(float);
            if (w->dtype == QW36_DTYPE_Q4_K) {
                NSUInteger k_blocks = cols / 256u;
                tg_bytes = (18u * k_blocks + 8u) * sizeof(float);
                if (tg_bytes < 256 * sizeof(float))
                    tg_bytes = 256 * sizeof(float);
            }
            double start_us = ctx->perf_enabled ? metal_perf_now_us() : 0.0;
            NSString *perf_label = ctx->perf_enabled
                ? [NSString stringWithFormat:@"%@_%ux%u",
                   metal_pipeline_label(ctx, qpipe),
                   (unsigned)rows, (unsigned)cols]
                : nil;
            int owns_cb = 0;
            id<MTLCommandBuffer> cb = metal_cb_for_op(ctx, &owns_cb);
            id<MTLComputeCommandEncoder> enc =
                metal_compute_encoder_for_op(ctx, cb, owns_cb);
            [enc setComputePipelineState:qpipe];
            [enc setBuffer:y->mtl offset:0 atIndex:0];
            [enc setBuffer:x->mtl offset:0 atIndex:1];
            [enc setBuffer:w->mtl offset:0 atIndex:2];
            [enc setBytes:&cols length:sizeof(cols) atIndex:3];
            [enc setBytes:&rows length:sizeof(rows) atIndex:4];
            [enc setThreadgroupMemoryLength:tg_bytes atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            metal_finish_compute_encoder(ctx, enc, owns_cb);
            return metal_commit_wait_profile(ctx, cb, owns_cb, perf_label, start_us);
        }
    }

    if (metal_dtype_is_host_dequant(w->dtype)) {
        float *w_f32 = metal_dequant_matrix(w, rows, cols);
        if (!w_f32) return -1;
        qw36_gpu_buf *w_tmp = metal_upload(ctx, w_f32,
                                           (size_t)rows * cols * sizeof(float),
                                           QW36_DTYPE_F32);
        free(w_f32);
        if (!w_tmp) return -1;
        int rc = metal_matmul(ctx, y, x, w_tmp, batch, rows, cols);
        metal_free(ctx, w_tmp);
        return rc;
    }

    /* fp16 MPS GEMV. Weights are F16; x and y can independently be F32
     * or F16. When state buffers move to fp16 (#46), x and y land as
     * F16 here directly, eliminating both f32↔f16 conversion dispatches
     * per matmul — that is the bulk of the to-200-tok/s gain. */
    if (ctx && batch == 1 && w->dtype == QW36_DTYPE_F16 &&
        (x->dtype == QW36_DTYPE_F32 || x->dtype == QW36_DTYPE_F16) &&
        (y->dtype == QW36_DTYPE_F32 || y->dtype == QW36_DTYPE_F16))
    {
        static int mma_f16gemv = -1;
        static uint32_t mma_min_rows = 0;
        static uint32_t mma_max_rows = UINT32_MAX;
        if (mma_f16gemv < 0) {
            const char *e = getenv("QW36_METAL_MMA_GEMV");
            mma_f16gemv = (e && atoi(e)) ? 1 : 0;
            const char *mn = getenv("QW36_METAL_MMA_GEMV_MIN_ROWS");
            const char *mx = getenv("QW36_METAL_MMA_GEMV_MAX_ROWS");
            int parsed_min = mn ? atoi(mn) : 0;
            int parsed_max = mx ? atoi(mx) : 0;
            mma_min_rows = parsed_min > 0 ? (uint32_t)parsed_min : 0u;
            mma_max_rows = parsed_max > 0 ? (uint32_t)parsed_max : UINT32_MAX;
        }
        if (mma_f16gemv && rows >= mma_min_rows && rows <= mma_max_rows &&
            ctx->matmul_mma_f16 && w->mtl && (cols % 8u) == 0) {
            double start_us = ctx->perf_enabled ? metal_perf_now_us() : 0.0;
            NSString *perf_label = ctx->perf_enabled
                ? [NSString stringWithFormat:@"mma_f16_gemv_%ux%u",
                   (unsigned)rows, (unsigned)cols]
                : nil;
            int owns_cb = 0;
            id<MTLCommandBuffer> cb = metal_cb_for_op(ctx, &owns_cb);
            id<MTLComputeCommandEncoder> enc =
                metal_compute_encoder_for_op(ctx, cb, owns_cb);
            [enc setComputePipelineState:ctx->matmul_mma_f16];
            uint32_t x_dtype = (uint32_t)x->dtype;
            uint32_t y_dtype = (uint32_t)y->dtype;
            [enc setBuffer:y->mtl offset:0 atIndex:0];
            [enc setBuffer:x->mtl offset:0 atIndex:1];
            [enc setBuffer:w->mtl offset:0 atIndex:2];
            [enc setBytes:&cols length:sizeof(cols) atIndex:3];
            [enc setBytes:&rows length:sizeof(rows) atIndex:4];
            [enc setBytes:&x_dtype length:sizeof(x_dtype) atIndex:5];
            [enc setBytes:&y_dtype length:sizeof(y_dtype) atIndex:6];
            [enc setThreadgroupMemoryLength:(NSUInteger)(8u * 128u * sizeof(uint16_t))
                                    atIndex:0];
            [enc setThreadgroupMemoryLength:(NSUInteger)(8u * 64u * sizeof(float))
                                    atIndex:1];
            [enc dispatchThreadgroups:MTLSizeMake((rows + 63u) / 64u, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            metal_finish_compute_encoder(ctx, enc, owns_cb);
            return metal_commit_wait_profile(ctx, cb, owns_cb, perf_label, start_us);
        }

        /* Opt-in qmv_quad-style GEMV: one SIMD threadgroup produces up to
         * 64 output rows using 8 quadgroups. This avoids MPS GEMV object
         * encoding overhead on the fp16 weight path while retaining fp32
         * accumulation inside the shader. */
        static int qmv_quad_f16gemv = -1;
        static uint32_t qmv_quad_max_rows = 0;
        if (qmv_quad_f16gemv < 0) {
            const char *e = getenv("QW36_METAL_F16_GEMV_QUAD");
            qmv_quad_f16gemv = (e && atoi(e)) ? 1 : 0;
            const char *mr = getenv("QW36_METAL_F16_GEMV_QUAD_MAX_ROWS");
            int parsed = mr ? atoi(mr) : 512;
            qmv_quad_max_rows = parsed > 0 ? (uint32_t)parsed : UINT32_MAX;
        }
        if (qmv_quad_f16gemv && rows <= qmv_quad_max_rows &&
            ctx->matmul_qmv_quad_f16 && w->mtl) {
            double start_us = ctx->perf_enabled ? metal_perf_now_us() : 0.0;
            NSString *perf_label = ctx->perf_enabled
                ? [NSString stringWithFormat:@"qmv_quad_f16_%ux%u",
                   (unsigned)rows, (unsigned)cols]
                : nil;
            int owns_cb = 0;
            id<MTLCommandBuffer> cb = metal_cb_for_op(ctx, &owns_cb);
            id<MTLComputeCommandEncoder> enc =
                metal_compute_encoder_for_op(ctx, cb, owns_cb);
            [enc setComputePipelineState:ctx->matmul_qmv_quad_f16];
            uint32_t x_dtype = (uint32_t)x->dtype;
            uint32_t y_dtype = (uint32_t)y->dtype;
            [enc setBuffer:y->mtl offset:0 atIndex:0];
            [enc setBuffer:x->mtl offset:0 atIndex:1];
            [enc setBuffer:w->mtl offset:0 atIndex:2];
            [enc setBytes:&cols length:sizeof(cols) atIndex:3];
            [enc setBytes:&rows length:sizeof(rows) atIndex:4];
            [enc setBytes:&x_dtype length:sizeof(x_dtype) atIndex:5];
            [enc setBytes:&y_dtype length:sizeof(y_dtype) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake(1, (rows + 63u) / 64u, 1)
                 threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            metal_finish_compute_encoder(ctx, enc, owns_cb);
            return metal_commit_wait_profile(ctx, cb, owns_cb, perf_label, start_us);
        }
        const int x_is_f16 = (x->dtype == QW36_DTYPE_F16);
        const int y_is_f16 = (y->dtype == QW36_DTYPE_F16);
        qw36_gpu_buf *xh = x_is_f16 ? x : metal_scratch(ctx,
            &ctx->matmul_x_f16_scratch,
            (size_t)cols * sizeof(uint16_t), QW36_DTYPE_F16);
        qw36_gpu_buf *yh = y_is_f16 ? y : metal_scratch(ctx,
            &ctx->matmul_y_f16_scratch,
            (size_t)rows * sizeof(uint16_t), QW36_DTYPE_F16);
        if (!xh || !yh) return -1;
        /* xh dedup: skip the f32→f16 conversion when this matmul's x is
         * the same buffer + cols that we converted on the last fp16
         * matmul (no intervening compute dispatch invalidated it).
         * Only needed when x came in as F32. */
        if (x_is_f16) {
            /* Direct fp16 input — the previous xh scratch is irrelevant
             * to anything that follows. Invalidate so the next f32-input
             * matmul re-converts instead of trusting stale cache. */
            metal_invalidate_matmul_xh(ctx);
        } else if (ctx->matmul_xh_src != x->mtl ||
                   ctx->matmul_xh_cols != (NSUInteger)cols)
        {
            metal_dispatch_1d(ctx, ctx->f32_to_f16, cols, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:xh->mtl offset:0 atIndex:0];
                [enc setBuffer:x->mtl  offset:0 atIndex:1];
                [enc setBytes:&cols length:sizeof(cols) atIndex:2];
            });
            ctx->matmul_xh_src = x->mtl;
            ctx->matmul_xh_cols = (NSUInteger)cols;
        }

        NSString *key = [NSString stringWithFormat:@"%ux%u:h",
                         (unsigned)rows, (unsigned)cols];
        MPSMatrixVectorMultiplication *gemv = ctx->mps_cache[key];
        if (!gemv) {
            gemv = [[MPSMatrixVectorMultiplication alloc]
                       initWithDevice:ctx->device rows:rows columns:cols];
            ctx->mps_cache[key] = gemv;
        }
        MPSVectorDescriptor *x_desc =
            [MPSVectorDescriptor vectorDescriptorWithLength:cols
                                                   dataType:MPSDataTypeFloat16];
        MPSVectorDescriptor *y_desc =
            [MPSVectorDescriptor vectorDescriptorWithLength:rows
                                                   dataType:MPSDataTypeFloat16];
        MPSMatrix *wm = metal_cached_weight_matrix(ctx, w, rows, cols,
                                                   MPSDataTypeFloat16,
                                                   sizeof(uint16_t), "h");
        MPSVector *xv = [[MPSVector alloc] initWithBuffer:xh->mtl descriptor:x_desc];
        MPSVector *yv = [[MPSVector alloc] initWithBuffer:yh->mtl descriptor:y_desc];
        double start_us = ctx->perf_enabled ? metal_perf_now_us() : 0.0;
        NSString *perf_label = ctx->perf_enabled
            ? [NSString stringWithFormat:@"mps_f16_gemv_%ux%u",
               (unsigned)rows, (unsigned)cols]
            : nil;
        int owns_cb = 0;
        id<MTLCommandBuffer> cb = metal_cb_for_op(ctx, &owns_cb);
        metal_flush_compute_encoder(ctx);
        [gemv encodeToCommandBuffer:cb inputMatrix:wm inputVector:xv resultVector:yv];
        int rc = metal_commit_wait_profile(ctx, cb, owns_cb, perf_label, start_us);
        if (rc) return rc;
        if (!y_is_f16) {
            metal_dispatch_1d(ctx, ctx->f16_to_f32, rows, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:y->mtl  offset:0 atIndex:0];
                [enc setBuffer:yh->mtl offset:0 atIndex:1];
                [enc setBytes:&rows length:sizeof(rows) atIndex:2];
            });
        }
        return 0;
    }

    if (ctx && batch == 1 && y->dtype == QW36_DTYPE_F32 &&
        x->dtype == QW36_DTYPE_F32 && w->dtype == QW36_DTYPE_F32) {
        /* Cache the MPSMatrixVectorMultiplication kernel by shape so we
         * don't re-create it on every call (lookup is ~3-5× cheaper than
         * alloc+init). The descriptor / MPSMatrix / MPSVector objects are
         * cheap to recreate per call. */
        NSString *key = [NSString stringWithFormat:@"%ux%u:f",
                         (unsigned)rows, (unsigned)cols];
        MPSMatrixVectorMultiplication *gemv = ctx->mps_cache[key];
        if (!gemv) {
            gemv = [[MPSMatrixVectorMultiplication alloc]
                       initWithDevice:ctx->device rows:rows columns:cols];
            ctx->mps_cache[key] = gemv;
        }
        MPSVectorDescriptor *x_desc =
            [MPSVectorDescriptor vectorDescriptorWithLength:cols
                                                   dataType:MPSDataTypeFloat32];
        MPSVectorDescriptor *y_desc =
            [MPSVectorDescriptor vectorDescriptorWithLength:rows
                                                   dataType:MPSDataTypeFloat32];
        MPSMatrix *wm = metal_cached_weight_matrix(ctx, w, rows, cols,
                                                   MPSDataTypeFloat32,
                                                   sizeof(float), "f");
        MPSVector *xv = [[MPSVector alloc] initWithBuffer:x->mtl descriptor:x_desc];
        MPSVector *yv = [[MPSVector alloc] initWithBuffer:y->mtl descriptor:y_desc];
        double start_us = ctx->perf_enabled ? metal_perf_now_us() : 0.0;
        NSString *perf_label = ctx->perf_enabled
            ? [NSString stringWithFormat:@"mps_f32_gemv_%ux%u",
               (unsigned)rows, (unsigned)cols]
            : nil;
        int owns_cb = 0;
        id<MTLCommandBuffer> cb = metal_cb_for_op(ctx, &owns_cb);
        metal_flush_compute_encoder(ctx);
        [gemv encodeToCommandBuffer:cb inputMatrix:wm inputVector:xv resultVector:yv];
        return metal_commit_wait_profile(ctx, cb, owns_cb, perf_label, start_us);
    }

    if (!ctx || !ctx->matmul) return -1;
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
    return 0;
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
    const int fused_qkv = wq && !wk && !wv;
    if (!ctx || !y || !x || !wq || (!fused_qkv && (!wk || !wv)) ||
        !q_norm || !k_norm ||
        !k_cache || !v_cache || n_heads == 0 || n_kv == 0 || head_dim == 0)
        return;

    const uint32_t q_rows = metal_matrix_rows_from_bytes(wq, hidden);
    /* Qwen3.5/3.6 vanilla q_proj output is n_heads * head_dim * 2 when the
     * attention has a Q-gate (per-head [Q(hd) | gate(hd)] concat). Detect
     * by comparing wq rows against the caller-supplied n_heads. */
    uint32_t q_len = n_heads * head_dim;
    uint32_t kv_len = n_kv * head_dim;
    uint32_t q_has_gate = 0;
    uint32_t fused_rows = 0;
    if (fused_qkv && q_rows == 2u * q_len + 2u * kv_len) {
        q_has_gate = 1;
        fused_rows = q_rows;
    } else if (fused_qkv && q_rows == q_len + 2u * kv_len) {
        fused_rows = q_rows;
    } else if (q_rows == 2u * n_heads * head_dim) {
        q_has_gate = 1;
    } else if (q_rows && q_rows % head_dim == 0) {
        const uint32_t inferred_heads = q_rows / head_dim;
        if (inferred_heads) n_heads = inferred_heads;
    }

    q_len = n_heads * head_dim;
    uint32_t q_proj_out_len = q_has_gate ? (2u * q_len) : q_len;
    if (fused_rows && fused_rows != q_proj_out_len + 2u * kv_len) return;
    uint32_t positions = seq_pos + 1;
    if (y->bytes < (size_t)q_len * sizeof(float)) return;
    const uint32_t q_scratch_len = fused_rows ? fused_rows : q_proj_out_len;
    qw36_gpu_buf *q = metal_scratch(ctx, &ctx->attn_q_scratch,
        (size_t)q_scratch_len * sizeof(float), QW36_DTYPE_F32);
    qw36_gpu_buf *k = q;
    qw36_gpu_buf *v = q;
    NSUInteger k_byte_offset = (NSUInteger)q_proj_out_len * sizeof(float);
    NSUInteger v_byte_offset = (NSUInteger)(q_proj_out_len + kv_len) * sizeof(float);
    uint32_t q_elem_offset = 0;
    uint32_t k_elem_offset = q_proj_out_len;
    uint32_t v_elem_offset = q_proj_out_len + kv_len;
    if (!fused_rows) {
        k = metal_scratch(ctx, &ctx->attn_k_scratch,
            (size_t)kv_len * sizeof(float), QW36_DTYPE_F32);
        v = metal_scratch(ctx, &ctx->attn_v_scratch,
            (size_t)kv_len * sizeof(float), QW36_DTYPE_F32);
        k_byte_offset = 0;
        v_byte_offset = 0;
        k_elem_offset = 0;
        v_elem_offset = 0;
    }
    if (!q || !k || !v) return;

    if (metal_matmul(ctx, q, x, wq, 1, q_scratch_len, hidden)) return;
    if (!fused_rows) {
        if (metal_matmul(ctx, k, x, wk, 1, kv_len, hidden)) return;
        if (metal_matmul(ctx, v, x, wv, 1, kv_len, hidden)) return;
    }

    float rms_eps = 1.0e-6f;
    uint32_t k_cache_dtype = (uint32_t)k_cache->dtype;
    uint32_t v_cache_dtype = (uint32_t)v_cache->dtype;
    uint32_t tg_size = 1;
    while (tg_size < head_dim) tg_size <<= 1;

    /* Fused decode path. The decode kernel consumes q/k/v as fp32, so any
     * matmul that lands fp32 is OK upstream. Accept both fp16 weights (MPS
     * gemv) and native quantised weights (our on-device dequant+gemv,
     * including the repacked Q4_K/Q5_K/Q6_K experiments). */
    int wq_ok = wq->dtype == QW36_DTYPE_F16 ||
                wq->dtype == QW36_DTYPE_Q4_K ||
                wq->dtype == QW36_DTYPE_Q4K_AFFINE32 ||
                wq->dtype == QW36_DTYPE_Q5_K ||
                wq->dtype == QW36_DTYPE_Q5K_AFFINE32 ||
                wq->dtype == QW36_DTYPE_Q6_K ||
                wq->dtype == QW36_DTYPE_Q6K_SCALE16 ||
                wq->dtype == QW36_DTYPE_Q8_0;
    int wk_ok = fused_rows ? wq_ok :
                (wk->dtype == QW36_DTYPE_F16 ||
                 wk->dtype == QW36_DTYPE_Q4_K ||
                 wk->dtype == QW36_DTYPE_Q4K_AFFINE32 ||
                 wk->dtype == QW36_DTYPE_Q5_K ||
                 wk->dtype == QW36_DTYPE_Q5K_AFFINE32 ||
                 wk->dtype == QW36_DTYPE_Q6_K ||
                 wk->dtype == QW36_DTYPE_Q6K_SCALE16 ||
                 wk->dtype == QW36_DTYPE_Q8_0);
    int wv_ok = fused_rows ? wq_ok :
                (wv->dtype == QW36_DTYPE_F16 ||
                 wv->dtype == QW36_DTYPE_Q4_K ||
                 wv->dtype == QW36_DTYPE_Q4K_AFFINE32 ||
                 wv->dtype == QW36_DTYPE_Q5_K ||
                 wv->dtype == QW36_DTYPE_Q5K_AFFINE32 ||
                 wv->dtype == QW36_DTYPE_Q6_K ||
                 wv->dtype == QW36_DTYPE_Q6K_SCALE16 ||
                 wv->dtype == QW36_DTYPE_Q8_0);
    /* QW36_METAL_KV_TRANSPOSED tri-state:
     *   unset / "auto"          → flip on when seq_capacity > 512
     *   "0" / "off" / "false"   → always legacy [t][head][dim]
     *   anything else           → always transposed [head][dim][t]
     *
     * Auto threshold matches the bench in commit b4bb6f6: transposed wins
     * +24% at n=2048 and +15% at n=1024 but regresses 10% at n=64. The
     * seq_capacity == 512 boundary is the breakeven on this host —
     * sessions sized for long context get the long-context optimisation
     * by default, short interactive sessions keep the cheap write path. */
    enum { KV_AUTO = -1, KV_OFF = 0, KV_ON = 1 };
    static int kv_transposed_mode_cached = -2;
    if (kv_transposed_mode_cached == -2) {
        const char *e = getenv("QW36_METAL_KV_TRANSPOSED");
        if (!e || !*e || strcasecmp(e, "auto") == 0) {
            kv_transposed_mode_cached = KV_AUTO;
        } else if (strcasecmp(e, "0") == 0 || strcasecmp(e, "off") == 0 ||
                   strcasecmp(e, "false") == 0) {
            kv_transposed_mode_cached = KV_OFF;
        } else {
            kv_transposed_mode_cached = KV_ON;
        }
    }
    uint32_t kv_transposed = (kv_transposed_mode_cached == KV_ON) ? 1u :
                              (kv_transposed_mode_cached == KV_OFF) ? 0u :
                              (seq_capacity > 512u) ? 1u : 0u;
    if (wq_ok && wk_ok && wv_ok &&
        tg_size <= 256 &&
        tg_size <= ctx->attn_decode_fused.maxTotalThreadsPerThreadgroup &&
        positions <= 4096) {
        const int kv16_kv =
            k_cache->dtype == v_cache->dtype &&
            (k_cache->dtype == QW36_DTYPE_F16 ||
             k_cache->dtype == QW36_DTYPE_BF16);
        /* x4 batched scoring variant: cuts barriers 4× per K reduction
         * iteration but measured win is within noise (~0-2% at long
         * context) because attention is K-cache-bandwidth-bound on this
         * host, not synchronization-bound.  Kept as opt-in research
         * artifact (QW36_METAL_ATTN_X4=1). Threshold positions>=128 so
         * short-context can't regress on shape edge cases. */
        static int attn_x4_env_cached = -1;
        if (attn_x4_env_cached < 0) {
            const char *e = getenv("QW36_METAL_ATTN_X4");
            attn_x4_env_cached = (e && atoi(e) != 0) ? 1 : 0;
        }
        /* QW36_METAL_FLASH_ATTN=1 opts into the online-softmax single-pass
         * decode kernel. Requires kv16_kv (fp16 or bf16 KV) — same
         * dispatch surface as the existing fused kernel. Defaults off
         * until precision smoke + perf gate run against it. */
        static int flash_attn_env_cached = -1;
        if (flash_attn_env_cached < 0) {
            const char *e = getenv("QW36_METAL_FLASH_ATTN");
            flash_attn_env_cached = (e && atoi(e) != 0) ? 1 : 0;
        }
        const int use_flash = kv16_kv && flash_attn_env_cached &&
                              ctx->attn_decode_flash_f16kv;
        const int use_x4 = !use_flash && kv16_kv && attn_x4_env_cached &&
                           ctx->attn_decode_fused_f16kv_x4 &&
                           positions >= 128u;
        double start_us = ctx->perf_enabled ? metal_perf_now_us() : 0.0;
        NSString *perf_label = use_flash
            ? @"qw36_attn_decode_flash_f16kv_f32"
            : (use_x4
                ? @"qw36_attn_decode_fused_f16kv_x4_f32"
                : (kv16_kv ? @"qw36_attn_decode_fused_f16kv_f32"
                           : @"qw36_attn_decode_fused_f32"));
        int owns_cb = 0;
        id<MTLCommandBuffer> cb = metal_cb_for_op(ctx, &owns_cb);
        id<MTLComputeCommandEncoder> enc =
            metal_compute_encoder_for_op(ctx, cb, owns_cb);
        uint32_t q_w_dtype = (uint32_t)q_norm->dtype;
        uint32_t k_w_dtype = (uint32_t)k_norm->dtype;
        [enc setComputePipelineState:use_flash
            ? ctx->attn_decode_flash_f16kv
            : (use_x4 ? ctx->attn_decode_fused_f16kv_x4
                : (kv16_kv ? ctx->attn_decode_fused_f16kv
                           : ctx->attn_decode_fused))];
        [enc setBuffer:y->mtl       offset:0 atIndex:0];
        [enc setBuffer:q->mtl       offset:0 atIndex:1];
        [enc setBuffer:k->mtl       offset:0 atIndex:2];
        [enc setBuffer:v->mtl       offset:0 atIndex:3];
        [enc setBuffer:k_cache->mtl offset:0 atIndex:4];
        [enc setBuffer:v_cache->mtl offset:0 atIndex:5];
        [enc setBuffer:q_norm->mtl  offset:0 atIndex:6];
        [enc setBuffer:k_norm->mtl  offset:0 atIndex:7];
        [enc setBytes:&n_heads length:sizeof(n_heads) atIndex:8];
        [enc setBytes:&n_kv length:sizeof(n_kv) atIndex:9];
        [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:10];
        [enc setBytes:&seq_pos length:sizeof(seq_pos) atIndex:11];
        [enc setBytes:&seq_capacity length:sizeof(seq_capacity) atIndex:12];
        [enc setBytes:&rope_theta length:sizeof(rope_theta) atIndex:13];
        [enc setBytes:&partial_rotary_factor length:sizeof(partial_rotary_factor) atIndex:14];
        [enc setBytes:&rms_eps length:sizeof(rms_eps) atIndex:15];
        [enc setBytes:&q_w_dtype length:sizeof(q_w_dtype) atIndex:16];
        [enc setBytes:&k_w_dtype length:sizeof(k_w_dtype) atIndex:17];
        [enc setBytes:&tg_size length:sizeof(tg_size) atIndex:18];
        uint32_t y_dtype = (uint32_t)y->dtype;
        if (!kv16_kv) {
            [enc setBytes:&k_cache_dtype length:sizeof(k_cache_dtype) atIndex:19];
            [enc setBytes:&v_cache_dtype length:sizeof(v_cache_dtype) atIndex:20];
            [enc setBytes:&q_has_gate length:sizeof(q_has_gate) atIndex:21];
            [enc setBytes:&y_dtype length:sizeof(y_dtype) atIndex:22];
            [enc setBytes:&q_elem_offset length:sizeof(q_elem_offset) atIndex:23];
            [enc setBytes:&k_elem_offset length:sizeof(k_elem_offset) atIndex:24];
            [enc setBytes:&v_elem_offset length:sizeof(v_elem_offset) atIndex:25];
            [enc setBytes:&kv_transposed length:sizeof(kv_transposed) atIndex:26];
        } else {
            [enc setBytes:&q_has_gate length:sizeof(q_has_gate) atIndex:19];
            [enc setBytes:&y_dtype length:sizeof(y_dtype) atIndex:20];
            [enc setBytes:&q_elem_offset length:sizeof(q_elem_offset) atIndex:21];
            [enc setBytes:&k_elem_offset length:sizeof(k_elem_offset) atIndex:22];
            [enc setBytes:&v_elem_offset length:sizeof(v_elem_offset) atIndex:23];
            [enc setBytes:&kv_transposed length:sizeof(kv_transposed) atIndex:24];
            [enc setBytes:&k_cache_dtype length:sizeof(k_cache_dtype) atIndex:25];
        }
        NSUInteger scratch_floats = (NSUInteger)tg_size + positions;
        if (kv16_kv && kv_transposed && !use_x4)
            scratch_floats += (NSUInteger)tg_size;
        [enc setThreadgroupMemoryLength:scratch_floats * sizeof(float)
                                atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        metal_finish_compute_encoder(ctx, enc, owns_cb);
        metal_commit_wait_profile(ctx, cb, owns_cb, perf_label, start_us);
        return;
    }

    NSUInteger prep_n = (NSUInteger)n_heads + n_kv;
    metal_dispatch_1d(ctx, ctx->attn_prep_decode, prep_n, ^(id<MTLComputeCommandEncoder> enc) {
        uint32_t q_w_dtype = (uint32_t)q_norm->dtype;
        uint32_t k_w_dtype = (uint32_t)k_norm->dtype;
        [enc setBuffer:q->mtl       offset:0 atIndex:0];
        [enc setBuffer:k->mtl       offset:k_byte_offset atIndex:1];
        [enc setBuffer:v->mtl       offset:v_byte_offset atIndex:2];
        [enc setBuffer:k_cache->mtl offset:0 atIndex:3];
        [enc setBuffer:v_cache->mtl offset:0 atIndex:4];
        [enc setBuffer:q_norm->mtl  offset:0 atIndex:5];
        [enc setBuffer:k_norm->mtl  offset:0 atIndex:6];
        [enc setBytes:&n_heads length:sizeof(n_heads) atIndex:7];
        [enc setBytes:&n_kv length:sizeof(n_kv) atIndex:8];
        [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:9];
        [enc setBytes:&seq_pos length:sizeof(seq_pos) atIndex:10];
        [enc setBytes:&seq_capacity length:sizeof(seq_capacity) atIndex:11];
        [enc setBytes:&rope_theta length:sizeof(rope_theta) atIndex:12];
        [enc setBytes:&partial_rotary_factor length:sizeof(partial_rotary_factor) atIndex:13];
        [enc setBytes:&rms_eps length:sizeof(rms_eps) atIndex:14];
        [enc setBytes:&q_w_dtype length:sizeof(q_w_dtype) atIndex:15];
        [enc setBytes:&k_w_dtype length:sizeof(k_w_dtype) atIndex:16];
        [enc setBytes:&k_cache_dtype length:sizeof(k_cache_dtype) atIndex:17];
        [enc setBytes:&v_cache_dtype length:sizeof(v_cache_dtype) atIndex:18];
        [enc setBytes:&kv_transposed length:sizeof(kv_transposed) atIndex:19];
    });

    if (tg_size <= 256 &&
        tg_size <= ctx->attn_score_combine_tg.maxTotalThreadsPerThreadgroup &&
        positions <= 4096) {
        double start_us = ctx->perf_enabled ? metal_perf_now_us() : 0.0;
        int owns_cb = 0;
        id<MTLCommandBuffer> cb = metal_cb_for_op(ctx, &owns_cb);
        id<MTLComputeCommandEncoder> enc =
            metal_compute_encoder_for_op(ctx, cb, owns_cb);
        [enc setComputePipelineState:ctx->attn_score_combine_tg];
        [enc setBuffer:y->mtl       offset:0 atIndex:0];
        [enc setBuffer:q->mtl       offset:0 atIndex:1];
        [enc setBuffer:k_cache->mtl offset:0 atIndex:2];
        [enc setBuffer:v_cache->mtl offset:0 atIndex:3];
        [enc setBytes:&n_heads length:sizeof(n_heads) atIndex:4];
        [enc setBytes:&n_kv length:sizeof(n_kv) atIndex:5];
        [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:6];
        [enc setBytes:&seq_pos length:sizeof(seq_pos) atIndex:7];
        [enc setBytes:&seq_capacity length:sizeof(seq_capacity) atIndex:8];
        [enc setBytes:&tg_size length:sizeof(tg_size) atIndex:9];
        [enc setBytes:&k_cache_dtype length:sizeof(k_cache_dtype) atIndex:10];
        [enc setBytes:&v_cache_dtype length:sizeof(v_cache_dtype) atIndex:11];
        [enc setBytes:&kv_transposed length:sizeof(kv_transposed) atIndex:12];
        [enc setThreadgroupMemoryLength:((NSUInteger)tg_size + positions) * sizeof(float)
                                atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        metal_finish_compute_encoder(ctx, enc, owns_cb);
        metal_commit_wait_profile(ctx, cb, owns_cb,
                                  @"qw36_attn_score_combine_tg_f32",
                                  start_us);
        return;
    }

    qw36_gpu_buf *scores = metal_scratch(ctx, &ctx->attn_scores_scratch,
        (size_t)n_heads * positions * sizeof(float), QW36_DTYPE_F32);
    if (!scores) return;

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
        [enc setBytes:&k_cache_dtype length:sizeof(k_cache_dtype) atIndex:8];
        [enc setBytes:&kv_transposed length:sizeof(kv_transposed) atIndex:9];
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
        [enc setBytes:&v_cache_dtype length:sizeof(v_cache_dtype) atIndex:8];
        [enc setBytes:&kv_transposed length:sizeof(kv_transposed) atIndex:9];
    });
}

static void metal_swiglu(qw36_gpu_ctx *ctx, qw36_gpu_buf *y, qw36_gpu_buf *x,
                         qw36_gpu_buf *w_gate, qw36_gpu_buf *w_up, qw36_gpu_buf *w_down,
                         uint32_t hidden, uint32_t inter)
{
    if (!ctx || !y || !x || !w_gate || !w_down) return;
    if (!w_up) {
        qw36_gpu_buf *gate_up = metal_scratch(ctx, &ctx->swiglu_gate_scratch,
            (size_t)inter * 2u * sizeof(float), QW36_DTYPE_F32);
        if (!gate_up) return;
        if (metal_matmul(ctx, gate_up, x, w_gate, 1, inter * 2u, hidden)) return;
        metal_dispatch_1d(ctx, ctx->silu_mul, inter, ^(id<MTLComputeCommandEncoder> enc) {
            uint32_t dtype = (uint32_t)gate_up->dtype;
            uint32_t gate_offset = 0;
            uint32_t up_offset = inter;
            [enc setBuffer:gate_up->mtl offset:0 atIndex:0];
            [enc setBuffer:gate_up->mtl offset:0 atIndex:1];
            [enc setBytes:&inter length:sizeof(inter) atIndex:2];
            [enc setBytes:&dtype length:sizeof(dtype) atIndex:3];
            [enc setBytes:&dtype length:sizeof(dtype) atIndex:4];
            [enc setBytes:&gate_offset length:sizeof(gate_offset) atIndex:5];
            [enc setBytes:&up_offset length:sizeof(up_offset) atIndex:6];
        });
        if (metal_matmul(ctx, y, gate_up, w_down, 1, hidden, inter)) return;
        return;
    }
    qw36_gpu_buf *gate = metal_scratch(ctx, &ctx->swiglu_gate_scratch,
        (size_t)inter * sizeof(float), QW36_DTYPE_F32);
    qw36_gpu_buf *up = metal_scratch(ctx, &ctx->swiglu_up_scratch,
        (size_t)inter * sizeof(float), QW36_DTYPE_F32);
    if (!gate || !up) return;
    if (metal_matmul(ctx, gate, x, w_gate, 1, inter, hidden)) return;
    if (metal_matmul(ctx, up,   x, w_up,   1, inter, hidden)) return;
    metal_dispatch_1d(ctx, ctx->silu_mul, inter, ^(id<MTLComputeCommandEncoder> enc) {
        uint32_t gate_dtype = (uint32_t)gate->dtype;
        uint32_t up_dtype = (uint32_t)up->dtype;
        uint32_t gate_offset = 0;
        uint32_t up_offset = 0;
        [enc setBuffer:gate->mtl offset:0 atIndex:0];
        [enc setBuffer:up->mtl   offset:0 atIndex:1];
        [enc setBytes:&inter length:sizeof(inter) atIndex:2];
        [enc setBytes:&gate_dtype length:sizeof(gate_dtype) atIndex:3];
        [enc setBytes:&up_dtype length:sizeof(up_dtype) atIndex:4];
        [enc setBytes:&gate_offset length:sizeof(gate_offset) atIndex:5];
        [enc setBytes:&up_offset length:sizeof(up_offset) atIndex:6];
    });
    if (metal_matmul(ctx, y, gate, w_down, 1, hidden, inter)) return;
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
                                 uint32_t key_dim, uint32_t val_dim,
                                 uint32_t alpha_offset,
                                 uint32_t beta_offset)
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
    if (!q || !k || !v || !g || !beta) return;

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
        [enc setBuffer:alpha_raw->mtl offset:(NSUInteger)alpha_offset * sizeof(float) atIndex:6];
        [enc setBuffer:beta_raw->mtl  offset:(NSUInteger)beta_offset * sizeof(float) atIndex:7];
        [enc setBuffer:dt_bias->mtl   offset:0 atIndex:8];
        [enc setBuffer:a_log->mtl     offset:0 atIndex:9];
        [enc setBytes:&n_key    length:sizeof(n_key)    atIndex:10];
        [enc setBytes:&n_value  length:sizeof(n_value)  atIndex:11];
        [enc setBytes:&key_dim  length:sizeof(key_dim)  atIndex:12];
        [enc setBytes:&val_dim  length:sizeof(val_dim)  atIndex:13];
    });

    const int reorder_v = (n_key && n_value > n_key && n_value % n_key == 0);
    qw36_gpu_buf *y_grouped = y;
    if (reorder_v) {
        y_grouped = metal_scratch(ctx, &ctx->dn_y_grouped_scratch,
            (size_t)n_value * val_dim * sizeof(float), QW36_DTYPE_F32);
        if (!y_grouped) return;
    }

    uint32_t T = 1;
    double start_us = ctx->perf_enabled ? metal_perf_now_us() : 0.0;
    int owns_cb = 0;
    id<MTLCommandBuffer> cb = metal_cb_for_op(ctx, &owns_cb);
    id<MTLComputeCommandEncoder> enc =
        metal_compute_encoder_for_op(ctx, cb, owns_cb);
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
    if (reorder_v) {
        NSUInteger total = (NSUInteger)n_value * val_dim;
        [enc setComputePipelineState:ctx->dn_reorder_gdr_y];
        [enc setBuffer:y->mtl         offset:0 atIndex:0];
        [enc setBuffer:y_grouped->mtl offset:0 atIndex:1];
        [enc setBytes:&n_key   length:sizeof(n_key)   atIndex:2];
        [enc setBytes:&n_value length:sizeof(n_value) atIndex:3];
        [enc setBytes:&val_dim length:sizeof(val_dim) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(total < 256 ? total : 256, 1, 1)];
    }
    metal_finish_compute_encoder(ctx, enc, owns_cb);
    metal_commit_wait_profile(ctx, cb, owns_cb,
                              @"qw36_gated_delta_step_f32", start_us);
}

static void metal_dn_gated_delta_conv1d(qw36_gpu_ctx *ctx, qw36_gpu_buf *y,
                                        qw36_gpu_buf *qkv_raw,
                                        qw36_gpu_buf *beta_raw,
                                        qw36_gpu_buf *alpha_raw,
                                        qw36_gpu_buf *dt_bias,
                                        qw36_gpu_buf *a_log,
                                        qw36_gpu_buf *state,
                                        qw36_gpu_buf *conv_w,
                                        qw36_gpu_buf *conv_state,
                                        uint32_t n_key, uint32_t n_value,
                                        uint32_t key_dim, uint32_t val_dim,
                                        uint32_t alpha_offset,
                                        uint32_t beta_offset,
                                        uint32_t kernel_size)
{
    if (!ctx || !ctx->dn_prep_gdr_conv1d || !y || !qkv_raw ||
        !beta_raw || !alpha_raw || !dt_bias || !a_log || !state ||
        !conv_w || !conv_state || !n_key || !n_value || !key_dim ||
        !val_dim)
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
    if (!q || !k || !v || !g || !beta) return;

    const uint32_t channels = n_key * key_dim * 2 + n_value * val_dim;
    const uint32_t heads = n_key > n_value ? n_key : n_value;
    const uint32_t tg_size = 256;
    double start_us = ctx->perf_enabled ? metal_perf_now_us() : 0.0;
    int owns_cb = 0;
    id<MTLCommandBuffer> cb = metal_cb_for_op(ctx, &owns_cb);
    id<MTLComputeCommandEncoder> enc =
        metal_compute_encoder_for_op(ctx, cb, owns_cb);
    [enc setComputePipelineState:ctx->dn_prep_gdr_conv1d];
    [enc setBuffer:q->mtl          offset:0 atIndex:0];
    [enc setBuffer:k->mtl          offset:0 atIndex:1];
    [enc setBuffer:v->mtl          offset:0 atIndex:2];
    [enc setBuffer:g->mtl          offset:0 atIndex:3];
    [enc setBuffer:beta->mtl       offset:0 atIndex:4];
    [enc setBuffer:qkv_raw->mtl    offset:0 atIndex:5];
    [enc setBuffer:alpha_raw->mtl  offset:(NSUInteger)alpha_offset * sizeof(float) atIndex:6];
    [enc setBuffer:beta_raw->mtl   offset:(NSUInteger)beta_offset * sizeof(float) atIndex:7];
    [enc setBuffer:dt_bias->mtl    offset:0 atIndex:8];
    [enc setBuffer:a_log->mtl      offset:0 atIndex:9];
    [enc setBuffer:conv_w->mtl     offset:0 atIndex:10];
    [enc setBuffer:conv_state->mtl offset:0 atIndex:11];
    [enc setBytes:&n_key       length:sizeof(n_key)       atIndex:12];
    [enc setBytes:&n_value     length:sizeof(n_value)     atIndex:13];
    [enc setBytes:&key_dim     length:sizeof(key_dim)     atIndex:14];
    [enc setBytes:&val_dim     length:sizeof(val_dim)     atIndex:15];
    [enc setBytes:&kernel_size length:sizeof(kernel_size) atIndex:16];
    [enc setBytes:&channels    length:sizeof(channels)    atIndex:17];
    [enc setBytes:&tg_size     length:sizeof(tg_size)     atIndex:18];
    [enc setThreadgroupMemoryLength:(NSUInteger)(2 * tg_size * sizeof(float)) atIndex:0];
    [enc dispatchThreads:MTLSizeMake(tg_size, heads, 1)
  threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];

    const int reorder_v = (n_key && n_value > n_key && n_value % n_key == 0);
    qw36_gpu_buf *y_grouped = y;
    if (reorder_v) {
        y_grouped = metal_scratch(ctx, &ctx->dn_y_grouped_scratch,
            (size_t)n_value * val_dim * sizeof(float), QW36_DTYPE_F32);
        if (!y_grouped) return;
    }

    uint32_t T = 1;
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
    if (reorder_v) {
        NSUInteger total = (NSUInteger)n_value * val_dim;
        [enc setComputePipelineState:ctx->dn_reorder_gdr_y];
        [enc setBuffer:y->mtl         offset:0 atIndex:0];
        [enc setBuffer:y_grouped->mtl offset:0 atIndex:1];
        [enc setBytes:&n_key   length:sizeof(n_key)   atIndex:2];
        [enc setBytes:&n_value length:sizeof(n_value) atIndex:3];
        [enc setBytes:&val_dim length:sizeof(val_dim) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(total < 256 ? total : 256, 1, 1)];
    }
    metal_finish_compute_encoder(ctx, enc, owns_cb);
    metal_commit_wait_profile(ctx, cb, owns_cb,
                              @"dn_conv1d_prep+gated_delta", start_us);
}

static void metal_dn_gated_rmsnorm(qw36_gpu_ctx *ctx, qw36_gpu_buf *y,
                                   qw36_gpu_buf *x, qw36_gpu_buf *z,
                                   qw36_gpu_buf *weight,
                                   uint32_t n_value, uint32_t val_dim,
                                   uint32_t z_offset, float eps)
{
    if (!ctx || !y || !x || !z || !weight) return;
    NSUInteger n = (NSUInteger)n_value * (NSUInteger)val_dim;
    metal_dispatch_1d(ctx, ctx->dn_gated_rmsnorm, n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:y->mtl      offset:0 atIndex:0];
        [enc setBuffer:x->mtl      offset:0 atIndex:1];
        [enc setBuffer:z->mtl      offset:(NSUInteger)z_offset * sizeof(float) atIndex:2];
        [enc setBuffer:weight->mtl offset:0 atIndex:3];
        [enc setBytes:&n_value length:sizeof(n_value) atIndex:4];
        [enc setBytes:&val_dim length:sizeof(val_dim) atIndex:5];
        [enc setBytes:&eps     length:sizeof(eps)     atIndex:6];
    });
}

static void metal_dn_gated_rmsnorm_matmul(qw36_gpu_ctx *ctx,
                                          qw36_gpu_buf *y,
                                          qw36_gpu_buf *x,
                                          qw36_gpu_buf *z,
                                          qw36_gpu_buf *weight,
                                          qw36_gpu_buf *w_out,
                                          uint32_t n_value,
                                          uint32_t val_dim,
                                          uint32_t out_rows,
                                          uint32_t z_offset,
                                          float eps)
{
    if (!ctx || !ctx->dn_gated_rmsnorm_matmul || !y || !x || !z ||
        !weight || !w_out || !n_value || !val_dim || !out_rows)
        return;
    const uint32_t in_cols = n_value * val_dim;
    if (w_out->dtype != QW36_DTYPE_F16 && w_out->dtype != QW36_DTYPE_F32)
        return;
    static int direct_matmul = -1;
    if (direct_matmul < 0) {
        const char *e = getenv("QW36_METAL_DN_TAIL_DIRECT");
        direct_matmul = (e && atoi(e) != 0) ? 1 : 0;
    }
    if (!direct_matmul && w_out->dtype == QW36_DTYPE_F16 &&
        ctx->dn_gated_rmsnorm_f16) {
        qw36_gpu_buf *xh = metal_scratch(ctx, &ctx->matmul_x_f16_scratch,
            (size_t)in_cols * sizeof(uint16_t), QW36_DTYPE_F16);
        if (!xh) return;
        metal_invalidate_matmul_xh(ctx);
        metal_dispatch_1d(ctx, ctx->dn_gated_rmsnorm_f16, in_cols,
            ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:xh->mtl     offset:0 atIndex:0];
                [enc setBuffer:x->mtl      offset:0 atIndex:1];
                [enc setBuffer:z->mtl      offset:(NSUInteger)z_offset * sizeof(float) atIndex:2];
                [enc setBuffer:weight->mtl offset:0 atIndex:3];
                [enc setBytes:&n_value length:sizeof(n_value) atIndex:4];
                [enc setBytes:&val_dim length:sizeof(val_dim) atIndex:5];
                [enc setBytes:&eps     length:sizeof(eps)     atIndex:6];
            });
        if (metal_matmul(ctx, y, xh, w_out, 1, out_rows, in_cols)) return;
        return;
    }
    metal_invalidate_matmul_xh(ctx);
    const uint32_t tg_size = 256;
    double start_us = ctx->perf_enabled ? metal_perf_now_us() : 0.0;
    int owns_cb = 0;
    id<MTLCommandBuffer> cb = metal_cb_for_op(ctx, &owns_cb);
    id<MTLComputeCommandEncoder> enc =
        metal_compute_encoder_for_op(ctx, cb, owns_cb);
    [enc setComputePipelineState:ctx->dn_gated_rmsnorm_matmul];
    uint32_t y_dtype = (uint32_t)y->dtype;
    uint32_t w_dtype = (uint32_t)w_out->dtype;
    [enc setBuffer:y->mtl      offset:0 atIndex:0];
    [enc setBuffer:x->mtl      offset:0 atIndex:1];
    [enc setBuffer:z->mtl      offset:(NSUInteger)z_offset * sizeof(float) atIndex:2];
    [enc setBuffer:weight->mtl offset:0 atIndex:3];
    [enc setBuffer:w_out->mtl  offset:0 atIndex:4];
    [enc setBytes:&n_value length:sizeof(n_value) atIndex:5];
    [enc setBytes:&val_dim length:sizeof(val_dim) atIndex:6];
    [enc setBytes:&out_rows length:sizeof(out_rows) atIndex:7];
    [enc setBytes:&in_cols length:sizeof(in_cols) atIndex:8];
    [enc setBytes:&eps length:sizeof(eps) atIndex:9];
    [enc setBytes:&w_dtype length:sizeof(w_dtype) atIndex:10];
    [enc setBytes:&y_dtype length:sizeof(y_dtype) atIndex:11];
    [enc setBytes:&tg_size length:sizeof(tg_size) atIndex:12];
    [enc setThreadgroupMemoryLength:((NSUInteger)n_value + tg_size) * sizeof(float)
                            atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(out_rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    metal_finish_compute_encoder(ctx, enc, owns_cb);
    metal_commit_wait_profile(ctx, cb, owns_cb,
                              @"qw36_dn_gated_rmsnorm_matmul_f32",
                              start_us);
}

static void metal_residual_add(qw36_gpu_ctx *ctx, qw36_gpu_buf *x,
                               qw36_gpu_buf *y, uint32_t n);

static void metal_moe_forward(qw36_gpu_ctx *ctx,
                              qw36_gpu_buf *y, qw36_gpu_buf *x,
                              qw36_gpu_buf *router,
                              qw36_gpu_buf *expert_gate,
                              qw36_gpu_buf *expert_up,
                              qw36_gpu_buf *expert_down,
                              qw36_gpu_buf *shared_gate,
                              qw36_gpu_buf *shared_up,
                              qw36_gpu_buf *shared_down,
                              qw36_gpu_buf *shared_gate_inp,
                              uint32_t hidden, uint32_t inter,
                              uint32_t shared_inter,
                              uint32_t num_experts, uint32_t experts_per_tok,
                              uint8_t norm_topk)
{
    (void)norm_topk;
    if (!ctx || !y || !x || !router || !expert_gate || !expert_up ||
        !expert_down || !hidden || !inter || !num_experts || !experts_per_tok)
        return;
    if (experts_per_tok > num_experts) experts_per_tok = num_experts;

    qw36_gpu_buf *r = metal_scratch(ctx, &ctx->moe_router_scratch,
        (size_t)num_experts * sizeof(float), QW36_DTYPE_F32);
    qw36_gpu_buf *p = metal_scratch(ctx, &ctx->moe_probs_scratch,
        (size_t)experts_per_tok * sizeof(float), QW36_DTYPE_F32);
    qw36_gpu_buf *idx = metal_scratch(ctx, &ctx->moe_idx_scratch,
        (size_t)experts_per_tok * sizeof(uint32_t), QW36_DTYPE_F32);
    qw36_gpu_buf *act = metal_scratch(ctx, &ctx->moe_act_scratch,
        (size_t)experts_per_tok * inter * sizeof(float), QW36_DTYPE_F32);
    if (!r || !p || !idx || !act) return;

    if (metal_matmul(ctx, r, x, router, 1, num_experts, hidden)) return;
    metal_dispatch_1d(ctx, ctx->moe_route, 1, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:r->mtl   offset:0 atIndex:0];
        [enc setBuffer:p->mtl   offset:0 atIndex:1];
        [enc setBuffer:idx->mtl offset:0 atIndex:2];
        [enc setBytes:&num_experts length:sizeof(num_experts) atIndex:3];
        [enc setBytes:&experts_per_tok length:sizeof(experts_per_tok) atIndex:4];
    });

    NSUInteger act_n = (NSUInteger)experts_per_tok * inter;
    metal_dispatch_1d(ctx, ctx->moe_gate_up, act_n, ^(id<MTLComputeCommandEncoder> enc) {
        uint32_t gate_dtype = (uint32_t)expert_gate->dtype;
        uint32_t up_dtype = (uint32_t)expert_up->dtype;
        [enc setBuffer:act->mtl         offset:0 atIndex:0];
        [enc setBuffer:x->mtl           offset:0 atIndex:1];
        [enc setBuffer:expert_gate->mtl offset:0 atIndex:2];
        [enc setBuffer:expert_up->mtl   offset:0 atIndex:3];
        [enc setBuffer:idx->mtl         offset:0 atIndex:4];
        [enc setBytes:&hidden length:sizeof(hidden) atIndex:5];
        [enc setBytes:&inter length:sizeof(inter) atIndex:6];
        [enc setBytes:&experts_per_tok length:sizeof(experts_per_tok) atIndex:7];
        [enc setBytes:&gate_dtype length:sizeof(gate_dtype) atIndex:8];
        [enc setBytes:&up_dtype length:sizeof(up_dtype) atIndex:9];
    });

    qw36_gpu_buf *shared = NULL;
    qw36_gpu_buf *shared_gate_scalar = NULL;
    if (shared_gate && shared_up && shared_down) {
        if (!shared_inter) shared_inter = inter;
        shared = metal_scratch(ctx, &ctx->moe_shared_scratch,
            (size_t)hidden * sizeof(float), QW36_DTYPE_F32);
        if (!shared) return;
        metal_swiglu(ctx, shared, x, shared_gate, shared_up, shared_down,
                     hidden, shared_inter);
        if (shared_gate_inp) {
            shared_gate_scalar = metal_scratch(ctx, &ctx->moe_shared_gate_scratch,
                sizeof(float), QW36_DTYPE_F32);
            if (!shared_gate_scalar) return;
            if (metal_matmul(ctx, shared_gate_scalar, x, shared_gate_inp,
                             1, 1, hidden)) return;
        }
    }

    metal_invalidate_matmul_xh(ctx);
    metal_dispatch_1d(ctx, ctx->moe_down_combine, hidden, ^(id<MTLComputeCommandEncoder> enc) {
        uint32_t down_dtype = (uint32_t)expert_down->dtype;
        [enc setBuffer:y->mtl           offset:0 atIndex:0];
        [enc setBuffer:act->mtl         offset:0 atIndex:1];
        [enc setBuffer:expert_down->mtl offset:0 atIndex:2];
        [enc setBuffer:p->mtl           offset:0 atIndex:3];
        [enc setBuffer:idx->mtl         offset:0 atIndex:4];
        [enc setBytes:&hidden length:sizeof(hidden) atIndex:5];
        [enc setBytes:&inter length:sizeof(inter) atIndex:6];
        [enc setBytes:&experts_per_tok length:sizeof(experts_per_tok) atIndex:7];
        [enc setBytes:&down_dtype length:sizeof(down_dtype) atIndex:8];
    });

    if (shared) {
        uint32_t has_scale = shared_gate_scalar ? 1u : 0u;
        metal_invalidate_matmul_xh(ctx);
        metal_dispatch_1d(ctx, ctx->moe_scale_add, hidden, ^(id<MTLComputeCommandEncoder> enc) {
            uint32_t y_dtype = (uint32_t)y->dtype;
            [enc setBuffer:y->mtl      offset:0 atIndex:0];
            [enc setBuffer:shared->mtl offset:0 atIndex:1];
            [enc setBuffer:(shared_gate_scalar ? shared_gate_scalar : shared)->mtl
                    offset:0 atIndex:2];
            [enc setBytes:&hidden length:sizeof(hidden) atIndex:3];
            [enc setBytes:&has_scale length:sizeof(has_scale) atIndex:4];
            [enc setBytes:&y_dtype length:sizeof(y_dtype) atIndex:5];
        });
    }
}

static void metal_residual_add(qw36_gpu_ctx *ctx, qw36_gpu_buf *x, qw36_gpu_buf *y, uint32_t n)
{
    if (!x || !y) return;
    /* residual_add writes to x, which is also the typical x source for
     * the next matmul — drop the cached fp16 input copy. */
    metal_invalidate_matmul_xh(ctx);
    metal_dispatch_1d(ctx, ctx->residual_add, n, ^(id<MTLComputeCommandEncoder> enc) {
        uint32_t x_dtype = (uint32_t)x->dtype;
        uint32_t y_dtype = (uint32_t)y->dtype;
        [enc setBuffer:x->mtl offset:0 atIndex:0];
        [enc setBuffer:y->mtl offset:0 atIndex:1];
        [enc setBytes:&n length:sizeof(n) atIndex:2];
        [enc setBytes:&x_dtype length:sizeof(x_dtype) atIndex:3];
        [enc setBytes:&y_dtype length:sizeof(y_dtype) atIndex:4];
    });
}

/* Fused residual_add + rmsnorm:
 *   x += y; out = x * rsqrt(mean(x^2) + eps) * w_buf
 * One TG covers the full hidden vector; up to 256 threads cooperate.
 * Skips the per-pair dispatch overhead that hits twice per layer. */
static void metal_residual_rmsnorm(qw36_gpu_ctx *ctx,
                                   qw36_gpu_buf *x, qw36_gpu_buf *y,
                                   qw36_gpu_buf *out, qw36_gpu_buf *w_buf,
                                   uint32_t hidden, float eps)
{
    if (!ctx || !x || !y || !out || !w_buf || !hidden) return;
    metal_invalidate_matmul_xh(ctx);
    double start_us = ctx->perf_enabled ? metal_perf_now_us() : 0.0;
    int owns_cb = 0;
    id<MTLCommandBuffer> cb = metal_cb_for_op(ctx, &owns_cb);
    id<MTLComputeCommandEncoder> enc =
        metal_compute_encoder_for_op(ctx, cb, owns_cb);
    [enc setComputePipelineState:ctx->residual_rmsnorm];
    uint32_t x_dtype = (uint32_t)x->dtype;
    uint32_t y_dtype = (uint32_t)y->dtype;
    uint32_t out_dtype = (uint32_t)out->dtype;
    uint32_t w_dtype = (uint32_t)w_buf->dtype;
    [enc setBuffer:x->mtl     offset:0 atIndex:0];
    [enc setBuffer:y->mtl     offset:0 atIndex:1];
    [enc setBuffer:out->mtl   offset:0 atIndex:2];
    [enc setBuffer:w_buf->mtl offset:0 atIndex:3];
    [enc setBytes:&hidden length:sizeof(hidden) atIndex:4];
    [enc setBytes:&eps    length:sizeof(eps)    atIndex:5];
    [enc setBytes:&x_dtype   length:sizeof(x_dtype)   atIndex:6];
    [enc setBytes:&y_dtype   length:sizeof(y_dtype)   atIndex:7];
    [enc setBytes:&out_dtype length:sizeof(out_dtype) atIndex:8];
    [enc setBytes:&w_dtype   length:sizeof(w_dtype)   atIndex:9];

    NSUInteger tg = ctx->residual_rmsnorm.maxTotalThreadsPerThreadgroup;
    if (tg > 256) tg = 256;
    if (tg > hidden) tg = hidden;
    NSUInteger simd_count = (tg + 31u) >> 5;
    [enc setThreadgroupMemoryLength:simd_count * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    metal_finish_compute_encoder(ctx, enc, owns_cb);
    metal_commit_wait_profile(ctx, cb, owns_cb,
                              @"qw36_residual_rmsnorm_f32", start_us);
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
        uint32_t y_dtype = (uint32_t)y->dtype;
        [enc setBuffer:y->mtl     offset:0 atIndex:0];
        [enc setBuffer:embed->mtl offset:0 atIndex:1];
        [enc setBytes:&token  length:sizeof(token)  atIndex:2];
        [enc setBytes:&hidden length:sizeof(hidden) atIndex:3];
        [enc setBytes:&embed_dtype length:sizeof(embed_dtype) atIndex:4];
        [enc setBytes:&y_dtype length:sizeof(y_dtype) atIndex:5];
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
    .dn_gated_delta_conv1d = metal_dn_gated_delta_conv1d,
    .dn_gated_rmsnorm  = metal_dn_gated_rmsnorm,
    .dn_gated_rmsnorm_matmul = metal_dn_gated_rmsnorm_matmul,
    .moe_forward       = metal_moe_forward,
    .residual_add      = metal_residual_add,
    .residual_rmsnorm  = metal_residual_rmsnorm,
    .embedding_lookup  = metal_embedding_lookup,
};

qw36_gpu_backend *qw36_backend_create(void) { return &g_metal_backend; }
