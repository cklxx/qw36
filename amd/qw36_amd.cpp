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
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

struct qw36_gpu_ctx {
    int device;
    hipStream_t stream;
    rocblas_handle blas;
};

struct qw36_gpu_buf {
    void   *dptr;
    void   *host_copy;
    size_t  bytes;
    qw36_dtype dtype;
};

/* --------------------------------------------------------------------- */
/* Host-side dequantization for simple bring-up of lazy GGUF weights.     */
/* --------------------------------------------------------------------- */

#define QW36_QK_K   256
#define QW36_QK8_0  32

static int amd_dtype_is_host_dequant(qw36_dtype dtype)
{
    return dtype == QW36_DTYPE_Q2_K ||
           dtype == QW36_DTYPE_Q3_K ||
           dtype == QW36_DTYPE_Q4_K ||
           dtype == QW36_DTYPE_Q5_K ||
           dtype == QW36_DTYPE_Q6_K ||
           dtype == QW36_DTYPE_Q8_0;
}

static float amd_host_f16_to_f32(uint16_t h)
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

static void amd_dq_q8_0(const uint8_t *blocks, float *out, size_t n)
{
    const size_t nb = n / QW36_QK8_0;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t *b = blocks + i * 34;
        uint16_t dh;
        std::memcpy(&dh, b, sizeof(dh));
        const float d = amd_host_f16_to_f32(dh);
        const int8_t *qs = (const int8_t *)(b + 2);
        for (int j = 0; j < QW36_QK8_0; j++) *out++ = d * (float)qs[j];
    }
}

static void amd_q4_K_get_scale_min(int j, const uint8_t *q,
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

static void amd_dq_q4_K(const uint8_t *blocks, float *out, size_t n)
{
    const size_t nb = n / QW36_QK_K;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t *b = blocks + i * 144;
        uint16_t dh, dmh;
        std::memcpy(&dh,  b,     sizeof(dh));
        std::memcpy(&dmh, b + 2, sizeof(dmh));
        const float d = amd_host_f16_to_f32(dh);
        const float dmn = amd_host_f16_to_f32(dmh);
        const uint8_t *scales = b + 4;
        const uint8_t *qs = b + 16;
        int is = 0;
        for (int j = 0; j < QW36_QK_K; j += 64) {
            uint8_t sc, m;
            amd_q4_K_get_scale_min(is + 0, scales, &sc, &m);
            const float d1 = d * (float)sc, m1 = dmn * (float)m;
            amd_q4_K_get_scale_min(is + 1, scales, &sc, &m);
            const float d2 = d * (float)sc, m2 = dmn * (float)m;
            for (int l = 0; l < 32; l++) *out++ = d1 * (float)(qs[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++) *out++ = d2 * (float)(qs[l] >> 4) - m2;
            qs += 32;
            is += 2;
        }
    }
}

static void amd_dq_q2_K(const uint8_t *blocks, float *out, size_t n)
{
    const size_t nb = n / QW36_QK_K;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t *b = blocks + i * 84;
        const uint8_t *scales = b;
        const uint8_t *qs = b + 16;
        uint16_t dh, dmh;
        std::memcpy(&dh,  b + 80, sizeof(dh));
        std::memcpy(&dmh, b + 82, sizeof(dmh));
        const float d = amd_host_f16_to_f32(dh);
        const float dmn = amd_host_f16_to_f32(dmh);
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

static void amd_dq_q3_K(const uint8_t *blocks, float *out, size_t n)
{
    const size_t nb = n / QW36_QK_K;
    static const uint32_t kmask1 = 0x03030303;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t *b = blocks + i * 110;
        const uint8_t *hm = b;
        const uint8_t *qs = b + 32;
        uint16_t dh;
        std::memcpy(&dh, b + 32 + 64 + 12, sizeof(dh));
        const float d_all = amd_host_f16_to_f32(dh);

        uint32_t aux[4] = {0, 0, 0, 0};
        std::memcpy(aux, b + 32 + 64, 12);
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

static void amd_dq_q5_K(const uint8_t *blocks, float *out, size_t n)
{
    const size_t nb = n / QW36_QK_K;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t *b = blocks + i * 176;
        uint16_t dh, dmh;
        std::memcpy(&dh,  b,     sizeof(dh));
        std::memcpy(&dmh, b + 2, sizeof(dmh));
        const float d = amd_host_f16_to_f32(dh);
        const float dmn = amd_host_f16_to_f32(dmh);
        const uint8_t *scales = b + 4;
        const uint8_t *qh = b + 16;
        const uint8_t *ql = b + 48;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QW36_QK_K; j += 64) {
            uint8_t sc, m;
            amd_q4_K_get_scale_min(is + 0, scales, &sc, &m);
            const float d1 = d * (float)sc, m1 = dmn * (float)m;
            amd_q4_K_get_scale_min(is + 1, scales, &sc, &m);
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

static void amd_dq_q6_K(const uint8_t *blocks, float *out, size_t n)
{
    const size_t nb = n / QW36_QK_K;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t *b = blocks + i * 210;
        const uint8_t *ql = b;
        const uint8_t *qh = b + 128;
        const int8_t *sc = (const int8_t *)(b + 128 + 64);
        uint16_t dh;
        std::memcpy(&dh, b + 128 + 64 + 16, sizeof(dh));
        const float d = amd_host_f16_to_f32(dh);
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

static int amd_quant_geom(qw36_dtype dtype, size_t *qk, size_t *bytes_per_block)
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

static size_t amd_dtype_bytes(qw36_dtype dtype)
{
    switch (dtype) {
        case QW36_DTYPE_F32:  return 4;
        case QW36_DTYPE_F16:  return 2;
        case QW36_DTYPE_BF16: return 2;
        default: return 0;
    }
}

static uint32_t amd_matrix_rows_from_bytes(const qw36_gpu_buf *buf, uint32_t cols)
{
    if (!buf || cols == 0) return 0;

    size_t row_bytes = 0;
    if (amd_dtype_is_host_dequant(buf->dtype)) {
        size_t qk, bpb;
        if (amd_quant_geom(buf->dtype, &qk, &bpb) || cols % qk != 0)
            return 0;
        row_bytes = ((size_t)cols / qk) * bpb;
    } else {
        const size_t elem_bytes = amd_dtype_bytes(buf->dtype);
        if (elem_bytes == 0) return 0;
        row_bytes = (size_t)cols * elem_bytes;
    }

    if (row_bytes == 0 || buf->bytes % row_bytes != 0) return 0;
    const size_t rows = buf->bytes / row_bytes;
    return rows <= UINT32_MAX ? (uint32_t)rows : 0;
}

static int amd_dequant_row(const qw36_gpu_buf *buf, size_t row_idx,
                           size_t cols, float *out)
{
    if (!buf || !buf->host_copy) return -1;
    size_t qk, bpb;
    if (amd_quant_geom(buf->dtype, &qk, &bpb) || cols % qk != 0) return -1;
    const size_t blocks_per_row = cols / qk;
    const uint8_t *row = (const uint8_t *)buf->host_copy
                       + row_idx * blocks_per_row * bpb;
    switch (buf->dtype) {
        case QW36_DTYPE_Q2_K: amd_dq_q2_K(row, out, cols); return 0;
        case QW36_DTYPE_Q3_K: amd_dq_q3_K(row, out, cols); return 0;
        case QW36_DTYPE_Q4_K: amd_dq_q4_K(row, out, cols); return 0;
        case QW36_DTYPE_Q5_K: amd_dq_q5_K(row, out, cols); return 0;
        case QW36_DTYPE_Q6_K: amd_dq_q6_K(row, out, cols); return 0;
        case QW36_DTYPE_Q8_0: amd_dq_q8_0(row, out, cols); return 0;
        default: return -1;
    }
}

static float *amd_dequant_matrix(const qw36_gpu_buf *buf,
                                 uint32_t rows, uint32_t cols)
{
    float *out = (float *)std::malloc((size_t)rows * cols * sizeof(float));
    if (!out) return nullptr;
    for (uint32_t r = 0; r < rows; r++) {
        if (amd_dequant_row(buf, r, cols, out + (size_t)r * cols)) {
            std::free(out);
            return nullptr;
        }
    }
    return out;
}

/* --------------------------------------------------------------------- */
/* Lifecycle                                                              */
/* --------------------------------------------------------------------- */

static qw36_gpu_ctx *amd_init(char *err, size_t err_cap)
{
    qw36_gpu_ctx *ctx = (qw36_gpu_ctx *)std::calloc(1, sizeof(*ctx));
    if (!ctx) { if (err && err_cap) std::snprintf(err, err_cap, "oom"); return nullptr; }

    hipError_t st = hipGetDevice(&ctx->device);
    if (st != hipSuccess) {
        if (err && err_cap) std::snprintf(err, err_cap, "hipGetDevice: %s",
                                          hipGetErrorString(st));
        std::free(ctx);
        return nullptr;
    }
    st = hipStreamCreate(&ctx->stream);
    if (st != hipSuccess) {
        if (err && err_cap) std::snprintf(err, err_cap, "hipStreamCreate: %s",
                                          hipGetErrorString(st));
        std::free(ctx);
        return nullptr;
    }
    rocblas_status bs = rocblas_create_handle(&ctx->blas);
    if (bs != rocblas_status_success) {
        if (err && err_cap) std::snprintf(err, err_cap, "rocblas_create_handle: %d", (int)bs);
        hipStreamDestroy(ctx->stream);
        std::free(ctx);
        return nullptr;
    }
    bs = rocblas_set_stream(ctx->blas, ctx->stream);
    if (bs != rocblas_status_success) {
        if (err && err_cap) std::snprintf(err, err_cap, "rocblas_set_stream: %d", (int)bs);
        rocblas_destroy_handle(ctx->blas);
        hipStreamDestroy(ctx->stream);
        std::free(ctx);
        return nullptr;
    }
    return ctx;
}

static void amd_destroy(qw36_gpu_ctx *ctx)
{
    if (!ctx) return;
    if (ctx->blas) rocblas_destroy_handle(ctx->blas);
    if (ctx->stream) hipStreamDestroy(ctx->stream);
    std::free(ctx);
}

/* --------------------------------------------------------------------- */
/* Memory                                                                 */
/* --------------------------------------------------------------------- */

static qw36_gpu_buf *amd_upload(qw36_gpu_ctx *ctx, const void *host,
                                size_t bytes, qw36_dtype dtype)
{
    if (!ctx || !host) return nullptr;
    qw36_gpu_buf *buf = (qw36_gpu_buf *)std::calloc(1, sizeof(*buf));
    if (!buf) return nullptr;
    hipError_t st = hipMalloc(&buf->dptr, bytes ? bytes : 1);
    if (st != hipSuccess) { std::free(buf); return nullptr; }
    if (bytes) {
        st = hipMemcpyAsync(buf->dptr, host, bytes, hipMemcpyHostToDevice, ctx->stream);
        if (st == hipSuccess) st = hipStreamSynchronize(ctx->stream);
        if (st != hipSuccess) { hipFree(buf->dptr); std::free(buf); return nullptr; }
    }
    buf->bytes = bytes;
    buf->dtype = dtype;
    if (amd_dtype_is_host_dequant(dtype) && bytes) {
        buf->host_copy = std::malloc(bytes);
        if (!buf->host_copy) {
            hipFree(buf->dptr);
            std::free(buf);
            return nullptr;
        }
        std::memcpy(buf->host_copy, host, bytes);
    }
    return buf;
}

static void amd_download(qw36_gpu_ctx *ctx, qw36_gpu_buf *buf,
                         void *host, size_t bytes)
{
    if (!ctx || !buf || !host) return;
    if (bytes > buf->bytes) bytes = buf->bytes;
    if (!bytes) return;
    hipMemcpyAsync(host, buf->dptr, bytes, hipMemcpyDeviceToHost, ctx->stream);
    hipStreamSynchronize(ctx->stream);
}

static void amd_copy_from_host(qw36_gpu_ctx *ctx, qw36_gpu_buf *buf,
                               const void *host, size_t bytes)
{
    if (!ctx || !buf || !host) return;
    if (bytes > buf->bytes) bytes = buf->bytes;
    if (!bytes) return;
    hipMemcpyAsync(buf->dptr, host, bytes, hipMemcpyHostToDevice, ctx->stream);
    hipStreamSynchronize(ctx->stream);
}

static qw36_gpu_buf *amd_alloc(qw36_gpu_ctx *ctx, size_t bytes, qw36_dtype dtype)
{
    if (!ctx) return nullptr;
    qw36_gpu_buf *buf = (qw36_gpu_buf *)std::calloc(1, sizeof(*buf));
    if (!buf) return nullptr;
    hipError_t st = hipMalloc(&buf->dptr, bytes ? bytes : 1);
    if (st != hipSuccess) { std::free(buf); return nullptr; }
    buf->bytes = bytes;
    buf->dtype = dtype;
    return buf;
}

static void amd_free(qw36_gpu_ctx *ctx, qw36_gpu_buf *buf)
{
    (void)ctx;
    if (!buf) return;
    if (buf->dptr) hipFree(buf->dptr);
    std::free(buf->host_copy);
    std::free(buf);
}

/* --------------------------------------------------------------------- */
/* Kernels                                                               */
/* --------------------------------------------------------------------- */

__device__ static float qw36_bf16_to_f32(uint16_t v)
{
    union {
        uint32_t u;
        float f;
    } bits;
    bits.u = ((uint32_t)v) << 16;
    return bits.f;
}

__device__ static float qw36_load_scalar(const void *ptr, qw36_dtype dtype, size_t i)
{
    switch (dtype) {
    case QW36_DTYPE_F32:
        return ((const float *)ptr)[i];
    case QW36_DTYPE_F16:
        return __half2float(((const __half *)ptr)[i]);
    case QW36_DTYPE_BF16:
        return qw36_bf16_to_f32(((const uint16_t *)ptr)[i]);
    default:
        return 0.0f;
    }
}

__global__ static void qw36_rmsnorm_kernel(float *out, const void *x,
                                           const void *w, qw36_dtype x_dtype,
                                           qw36_dtype w_dtype,
                                           uint32_t hidden, float eps)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= hidden) return;
    float ss = 0.0f;
    for (uint32_t i = 0; i < hidden; ++i) {
        float v = qw36_load_scalar(x, x_dtype, i);
        ss += v * v;
    }
    float scale = rsqrtf(ss / (float)hidden + eps);
    out[tid] = qw36_load_scalar(x, x_dtype, tid) * scale *
               qw36_load_scalar(w, w_dtype, tid);
}

__global__ static void qw36_matmul_kernel(float *y, const void *x, const void *w,
                                          qw36_dtype x_dtype, qw36_dtype w_dtype,
                                          uint32_t batch, uint32_t rows,
                                          uint32_t cols)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = batch * rows;
    if (gid >= total) return;
    uint32_t b = gid / rows;
    uint32_t r = gid - b * rows;
    size_t x_base = (size_t)b * cols;
    size_t w_base = (size_t)r * cols;
    float acc = 0.0f;
    for (uint32_t c = 0; c < cols; ++c)
        acc += qw36_load_scalar(x, x_dtype, x_base + c) *
               qw36_load_scalar(w, w_dtype, w_base + c);
    y[gid] = acc;
}

__global__ static void qw36_silu_mul_kernel(float *gate_io, const float *up,
                                            uint32_t n)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    float g = gate_io[tid];
    gate_io[tid] = (g / (1.0f + expf(-g))) * up[tid];
}

__global__ static void qw36_residual_add_kernel(float *x, const float *y,
                                                uint32_t n)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) x[tid] += y[tid];
}

__global__ static void qw36_embedding_lookup_kernel(float *y, const void *embed,
                                                    qw36_dtype embed_dtype,
                                                    uint32_t token,
                                                    uint32_t hidden)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < hidden)
        y[tid] = qw36_load_scalar(embed, embed_dtype, (size_t)token * hidden + tid);
}

__global__ static void qw36_head_norm_rope_kernel(float *x, const void *w,
                                                  qw36_dtype w_dtype,
                                                  uint32_t heads,
                                                  uint32_t head_dim,
                                                  uint32_t seq_pos,
                                                  float theta,
                                                  float partial,
                                                  float eps)
{
    uint32_t h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= heads) return;
    float *xh = x + (size_t)h * head_dim;
    float ss = 0.0f;
    for (uint32_t d = 0; d < head_dim; ++d) ss += xh[d] * xh[d];
    float scale = rsqrtf(ss / (float)head_dim + eps);

    uint32_t rot_dim = (uint32_t)((float)head_dim * partial);
    if (rot_dim > head_dim) rot_dim = head_dim;
    rot_dim &= ~1u;
    uint32_t half_dim = rot_dim / 2;
    for (uint32_t pair = 0; pair < half_dim; ++pair) {
        uint32_t d0 = pair;
        uint32_t d1 = pair + half_dim;
        float x0 = xh[d0] * scale * qw36_load_scalar(w, w_dtype, d0);
        float x1 = xh[d1] * scale * qw36_load_scalar(w, w_dtype, d1);
        float inv_freq = 1.0f / powf(theta, (2.0f * (float)pair) / (float)rot_dim);
        float angle = (float)seq_pos * inv_freq;
        float c = cosf(angle);
        float s = sinf(angle);
        xh[d0] = x0 * c - x1 * s;
        xh[d1] = x0 * s + x1 * c;
    }
    for (uint32_t d = rot_dim; d < head_dim; ++d)
        xh[d] *= scale * qw36_load_scalar(w, w_dtype, d);
}

__global__ static void qw36_kv_append_kernel(float *k_cache, float *v_cache,
                                             const float *k, const float *v,
                                             uint32_t seq_pos,
                                             uint32_t seq_capacity,
                                             uint32_t kv_len)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kv_len || seq_pos >= seq_capacity) return;
    size_t off = (size_t)seq_pos * kv_len + tid;
    k_cache[off] = k[tid];
    v_cache[off] = v[tid];
}

__global__ static void qw36_attn_scores_kernel(float *scores, const float *q,
                                               const float *k_cache,
                                               uint32_t n_heads,
                                               uint32_t n_kv,
                                               uint32_t head_dim,
                                               uint32_t seq_pos,
                                               uint32_t seq_capacity)
{
    uint32_t count = seq_pos + 1;
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = n_heads * count;
    if (gid >= total || count > seq_capacity) return;
    uint32_t h = gid / count;
    uint32_t pos = gid - h * count;
    uint32_t kv_h = h * n_kv / n_heads;
    uint32_t kv_len = n_kv * head_dim;
    const float *qh = q + (size_t)h * head_dim;
    const float *kh = k_cache + (size_t)pos * kv_len + (size_t)kv_h * head_dim;
    float acc = 0.0f;
    for (uint32_t d = 0; d < head_dim; ++d) acc += qh[d] * kh[d];
    scores[gid] = acc / sqrtf((float)head_dim);
}

__global__ static void qw36_attn_softmax_kernel(float *scores,
                                                uint32_t n_heads,
                                                uint32_t seq_pos)
{
    uint32_t h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n_heads) return;
    uint32_t count = seq_pos + 1;
    float *row = scores + (size_t)h * count;
    float maxv = row[0];
    for (uint32_t i = 1; i < count; ++i) maxv = fmaxf(maxv, row[i]);
    float sum = 0.0f;
    for (uint32_t i = 0; i < count; ++i) {
        float e = expf(row[i] - maxv);
        row[i] = e;
        sum += e;
    }
    float inv = 1.0f / sum;
    for (uint32_t i = 0; i < count; ++i) row[i] *= inv;
}

__global__ static void qw36_attn_combine_kernel(float *y, const float *scores,
                                                const float *v_cache,
                                                uint32_t n_heads,
                                                uint32_t n_kv,
                                                uint32_t head_dim,
                                                uint32_t seq_pos,
                                                uint32_t seq_capacity)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t hidden = n_heads * head_dim;
    if (gid >= hidden || seq_pos >= seq_capacity) return;
    uint32_t h = gid / head_dim;
    uint32_t d = gid - h * head_dim;
    uint32_t kv_h = h * n_kv / n_heads;
    uint32_t count = seq_pos + 1;
    uint32_t kv_len = n_kv * head_dim;
    const float *row = scores + (size_t)h * count;
    float acc = 0.0f;
    for (uint32_t pos = 0; pos < count; ++pos) {
        size_t off = (size_t)pos * kv_len + (size_t)kv_h * head_dim + d;
        acc += row[pos] * v_cache[off];
    }
    y[gid] = acc;
}

static int amd_blocks(size_t n)
{
    return (int)((n + 255) / 256);
}

static void amd_sync(qw36_gpu_ctx *ctx)
{
    if (ctx) hipStreamSynchronize(ctx->stream);
}

static void amd_zero_output(qw36_gpu_ctx *ctx, qw36_gpu_buf *buf, size_t bytes)
{
    if (!ctx || !buf || !buf->dptr) return;
    if (bytes > buf->bytes) bytes = buf->bytes;
    if (!bytes) return;
    hipMemsetAsync(buf->dptr, 0, bytes, ctx->stream);
    amd_sync(ctx);
}

static void amd_rmsnorm(qw36_gpu_ctx *ctx, qw36_gpu_buf *out, qw36_gpu_buf *x,
                        qw36_gpu_buf *w, uint32_t hidden, float eps)
{
    if (!ctx || !out || !x || !w || hidden == 0) return;
    hipLaunchKernelGGL(qw36_rmsnorm_kernel, dim3(amd_blocks(hidden)), dim3(256),
                       0, ctx->stream, (float *)out->dptr, x->dptr, w->dptr,
                       x->dtype, w->dtype, hidden, eps);
    amd_sync(ctx);
}

static void amd_matmul(qw36_gpu_ctx *ctx, qw36_gpu_buf *y, qw36_gpu_buf *x,
                       qw36_gpu_buf *w, uint32_t batch, uint32_t rows, uint32_t cols)
{
    if (!ctx || !y || !x || !w || batch == 0 || rows == 0 || cols == 0) return;
    if (amd_dtype_is_host_dequant(w->dtype)) {
        float *w_f32 = amd_dequant_matrix(w, rows, cols);
        if (!w_f32) {
            amd_zero_output(ctx, y, (size_t)batch * rows * sizeof(float));
            return;
        }
        qw36_gpu_buf *w_tmp = amd_upload(ctx, w_f32,
                                         (size_t)rows * cols * sizeof(float),
                                         QW36_DTYPE_F32);
        std::free(w_f32);
        if (!w_tmp) {
            amd_zero_output(ctx, y, (size_t)batch * rows * sizeof(float));
            return;
        }
        amd_matmul(ctx, y, x, w_tmp, batch, rows, cols);
        amd_free(ctx, w_tmp);
        return;
    }

    if (ctx->blas && y->dtype == QW36_DTYPE_F32 &&
        x->dtype == QW36_DTYPE_F32 && w->dtype == QW36_DTYPE_F32) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        rocblas_status bs = rocblas_sgemm(ctx->blas,
                                          rocblas_operation_transpose,
                                          rocblas_operation_none,
                                          (int)rows, (int)batch, (int)cols,
                                          &alpha,
                                          (const float *)w->dptr, (int)cols,
                                          (const float *)x->dptr, (int)cols,
                                          &beta,
                                          (float *)y->dptr, (int)rows);
        if (bs == rocblas_status_success) {
            amd_sync(ctx);
            return;
        }
    }

    size_t n = (size_t)batch * rows;
    hipLaunchKernelGGL(qw36_matmul_kernel, dim3(amd_blocks(n)), dim3(256),
                       0, ctx->stream, (float *)y->dptr, x->dptr, w->dptr,
                       x->dtype, w->dtype, batch, rows, cols);
    amd_sync(ctx);
}

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
    if (!ctx || !y || !x || !wq || !wk || !wv || !q_norm || !k_norm ||
        !k_cache || !v_cache || n_heads == 0 || n_kv == 0 || head_dim == 0)
        return;

    const uint32_t q_rows = amd_matrix_rows_from_bytes(wq, hidden);
    if (q_rows && q_rows % head_dim == 0) {
        const uint32_t inferred_heads = q_rows / head_dim;
        if (inferred_heads) n_heads = inferred_heads;
    }

    uint32_t q_len = n_heads * head_dim;
    uint32_t kv_len = n_kv * head_dim;
    uint32_t positions = seq_pos + 1;
    if (y->bytes < (size_t)q_len * sizeof(float)) return;
    qw36_gpu_buf *q = amd_alloc(ctx, (size_t)q_len * sizeof(float), QW36_DTYPE_F32);
    qw36_gpu_buf *k = amd_alloc(ctx, (size_t)kv_len * sizeof(float), QW36_DTYPE_F32);
    qw36_gpu_buf *v = amd_alloc(ctx, (size_t)kv_len * sizeof(float), QW36_DTYPE_F32);
    qw36_gpu_buf *scores = amd_alloc(ctx, (size_t)n_heads * positions * sizeof(float),
                                     QW36_DTYPE_F32);
    if (!q || !k || !v || !scores) goto done;

    amd_matmul(ctx, q, x, wq, 1, q_len, hidden);
    amd_matmul(ctx, k, x, wk, 1, kv_len, hidden);
    amd_matmul(ctx, v, x, wv, 1, kv_len, hidden);

    float rms_eps = 1.0e-6f;
    hipLaunchKernelGGL(qw36_head_norm_rope_kernel, dim3(amd_blocks(n_heads)), dim3(256),
                       0, ctx->stream, (float *)q->dptr, q_norm->dptr, q_norm->dtype,
                       n_heads, head_dim, seq_pos, rope_theta,
                       partial_rotary_factor, rms_eps);
    hipLaunchKernelGGL(qw36_head_norm_rope_kernel, dim3(amd_blocks(n_kv)), dim3(256),
                       0, ctx->stream, (float *)k->dptr, k_norm->dptr, k_norm->dtype,
                       n_kv, head_dim, seq_pos, rope_theta,
                       partial_rotary_factor, rms_eps);
    hipLaunchKernelGGL(qw36_kv_append_kernel, dim3(amd_blocks(kv_len)), dim3(256),
                       0, ctx->stream, (float *)k_cache->dptr, (float *)v_cache->dptr,
                       (const float *)k->dptr, (const float *)v->dptr,
                       seq_pos, seq_capacity, kv_len);
    hipLaunchKernelGGL(qw36_attn_scores_kernel,
                       dim3(amd_blocks((size_t)n_heads * positions)), dim3(256),
                       0, ctx->stream, (float *)scores->dptr,
                       (const float *)q->dptr, (const float *)k_cache->dptr,
                       n_heads, n_kv, head_dim, seq_pos, seq_capacity);
    hipLaunchKernelGGL(qw36_attn_softmax_kernel, dim3(amd_blocks(n_heads)), dim3(256),
                       0, ctx->stream, (float *)scores->dptr, n_heads, seq_pos);
    hipLaunchKernelGGL(qw36_attn_combine_kernel, dim3(amd_blocks(q_len)), dim3(256),
                       0, ctx->stream, (float *)y->dptr, (const float *)scores->dptr,
                       (const float *)v_cache->dptr, n_heads, n_kv, head_dim,
                       seq_pos, seq_capacity);
    amd_sync(ctx);

done:
    amd_free(ctx, scores);
    amd_free(ctx, v);
    amd_free(ctx, k);
    amd_free(ctx, q);
}

static void amd_swiglu(qw36_gpu_ctx *ctx, qw36_gpu_buf *y, qw36_gpu_buf *x,
                       qw36_gpu_buf *w_gate, qw36_gpu_buf *w_up, qw36_gpu_buf *w_down,
                       uint32_t hidden, uint32_t inter)
{
    if (!ctx || !y || !x || !w_gate || !w_up || !w_down || inter == 0) return;
    qw36_gpu_buf *gate = amd_alloc(ctx, (size_t)inter * sizeof(float), QW36_DTYPE_F32);
    qw36_gpu_buf *up = amd_alloc(ctx, (size_t)inter * sizeof(float), QW36_DTYPE_F32);
    if (!gate || !up) goto done;
    amd_matmul(ctx, gate, x, w_gate, 1, inter, hidden);
    amd_matmul(ctx, up, x, w_up, 1, inter, hidden);
    hipLaunchKernelGGL(qw36_silu_mul_kernel, dim3(amd_blocks(inter)), dim3(256),
                       0, ctx->stream, (float *)gate->dptr, (const float *)up->dptr, inter);
    amd_sync(ctx);
    amd_matmul(ctx, y, gate, w_down, 1, hidden, inter);

done:
    amd_free(ctx, up);
    amd_free(ctx, gate);
}

/* Gated Delta Rule step — HIP fp32 port matching cuda_gdr_step.
 * See cuda/qw36_cuda.cu for the algorithm reference. */
__global__ static void qw36_gated_delta_step_kernel(
    const float *q, const float *k, const float *v,
    const float *g, const float *beta,
    const float *state_in,
    float *y, float *state_out,
    int T, int Hk, int Hv, int Dk, int Dv)
{
    int n      = blockIdx.x;
    int b_idx  = n / Hv;
    int hv_idx = n % Hv;
    int hk_idx = hv_idx / (Hv / Hk);
    int dv_idx = threadIdx.x;
    if (dv_idx >= Dv) return;

    const float *q_ = q + (size_t)b_idx * T * Hk * Dk + hk_idx * Dk;
    const float *k_ = k + (size_t)b_idx * T * Hk * Dk + hk_idx * Dk;
    const float *v_ = v + (size_t)b_idx * T * Hv * Dv + hv_idx * Dv;
    float       *y_ = y + (size_t)b_idx * T * Hv * Dv + hv_idx * Dv;
    const float *g_    = g    + (size_t)b_idx * T * Hv;
    const float *beta_ = beta + (size_t)b_idx * T * Hv;

    float state[256];
    const float *i_state = state_in + ((size_t)n * Dv + dv_idx) * Dk;
    for (int j = 0; j < Dk; j++) state[j] = i_state[j];

    for (int t = 0; t < T; t++) {
        float g_val = g_[hv_idx];
        float kv_mem = 0.0f;
        for (int j = 0; j < Dk; j++) {
            state[j] *= g_val;
            kv_mem  += state[j] * k_[j];
        }
        float delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

        float out = 0.0f;
        for (int j = 0; j < Dk; j++) {
            state[j] += k_[j] * delta;
            out     += state[j] * q_[j];
        }
        y_[dv_idx] = out;

        q_    += Hk * Dk; k_    += Hk * Dk;
        v_    += Hv * Dv; y_    += Hv * Dv;
        g_    += Hv;      beta_ += Hv;
    }

    float *o_state = state_out + ((size_t)n * Dv + dv_idx) * Dk;
    for (int j = 0; j < Dk; j++) o_state[j] = state[j];
}

static void amd_gdr_step(qw36_gpu_ctx *ctx,
                         qw36_gpu_buf *y, qw36_gpu_buf *state_out,
                         qw36_gpu_buf *q, qw36_gpu_buf *k, qw36_gpu_buf *v,
                         qw36_gpu_buf *g, qw36_gpu_buf *beta,
                         qw36_gpu_buf *state_in,
                         uint32_t T, uint32_t Hk, uint32_t Hv,
                         uint32_t Dk, uint32_t Dv)
{
    if (!ctx || !y || !state_out || !q || !k || !v || !g || !beta || !state_in) return;
    qw36_gated_delta_step_kernel<<<(int)Hv, (int)Dv, 0, ctx->stream>>>(
        (const float *)q->dptr, (const float *)k->dptr, (const float *)v->dptr,
        (const float *)g->dptr, (const float *)beta->dptr,
        (const float *)state_in->dptr,
        (float *)y->dptr, (float *)state_out->dptr,
        (int)T, (int)Hk, (int)Hv, (int)Dk, (int)Dv);
    hipStreamSynchronize(ctx->stream);
}

static void amd_residual_add(qw36_gpu_ctx *ctx, qw36_gpu_buf *x, qw36_gpu_buf *y, uint32_t n)
{
    if (!ctx || !x || !y || n == 0) return;
    hipLaunchKernelGGL(qw36_residual_add_kernel, dim3(amd_blocks(n)), dim3(256),
                       0, ctx->stream, (float *)x->dptr, (const float *)y->dptr, n);
    amd_sync(ctx);
}

static void amd_embedding_lookup(qw36_gpu_ctx *ctx, qw36_gpu_buf *y, qw36_gpu_buf *embed,
                                 uint32_t token, uint32_t hidden)
{
    if (!ctx || !y || !embed || hidden == 0) return;
    if (amd_dtype_is_host_dequant(embed->dtype)) {
        float *row = (float *)std::malloc((size_t)hidden * sizeof(float));
        if (!row) {
            amd_zero_output(ctx, y, (size_t)hidden * sizeof(float));
            return;
        }
        if (amd_dequant_row(embed, token, hidden, row) == 0) {
            hipMemcpyAsync(y->dptr, row, (size_t)hidden * sizeof(float),
                           hipMemcpyHostToDevice, ctx->stream);
            amd_sync(ctx);
        } else {
            amd_zero_output(ctx, y, (size_t)hidden * sizeof(float));
        }
        std::free(row);
        return;
    }

    hipLaunchKernelGGL(qw36_embedding_lookup_kernel, dim3(amd_blocks(hidden)), dim3(256),
                       0, ctx->stream, (float *)y->dptr, embed->dptr,
                       embed->dtype, token, hidden);
    amd_sync(ctx);
}

/* --------------------------------------------------------------------- */
/* Vtable                                                                 */
/* --------------------------------------------------------------------- */

static qw36_gpu_backend g_amd_backend = {
    /* name              */ "amd",
    /* init              */ amd_init,
    /* destroy           */ amd_destroy,
    /* begin_batch       */ nullptr,
    /* end_batch         */ nullptr,
    /* upload            */ amd_upload,
    /* download          */ amd_download,
    /* copy_from_host    */ amd_copy_from_host,
    /* alloc             */ amd_alloc,
    /* free              */ amd_free,
    /* rmsnorm           */ amd_rmsnorm,
    /* matmul            */ amd_matmul,
    /* attention         */ amd_attention,
    /* swiglu_mlp        */ amd_swiglu,
    /* dn_conv1d_silu    */ nullptr,
    /* dn_gated_delta    */ nullptr,
    /* dn_gated_rmsnorm  */ nullptr,
    /* moe_forward       */ nullptr,
    /* residual_add      */ amd_residual_add,
    /* embedding_lookup  */ amd_embedding_lookup,
};

extern "C" qw36_gpu_backend *qw36_backend_create(void) { return &g_amd_backend; }
