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
    for (uint32_t pair = 0; pair < rot_dim / 2; ++pair) {
        uint32_t d0 = 2 * pair;
        uint32_t d1 = d0 + 1;
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
    uint32_t kv_h = h % n_kv;
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
    uint32_t kv_h = h % n_kv;
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

    uint32_t q_len = n_heads * head_dim;
    uint32_t kv_len = n_kv * head_dim;
    uint32_t positions = seq_pos + 1;
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
    /* upload            */ amd_upload,
    /* download          */ amd_download,
    /* alloc             */ amd_alloc,
    /* free              */ amd_free,
    /* rmsnorm           */ amd_rmsnorm,
    /* matmul            */ amd_matmul,
    /* attention         */ amd_attention,
    /* swiglu_mlp        */ amd_swiglu,
    /* moe_forward       */ nullptr,
    /* residual_add      */ amd_residual_add,
    /* embedding_lookup  */ amd_embedding_lookup,
};

extern "C" qw36_gpu_backend *qw36_backend_create(void) { return &g_amd_backend; }
