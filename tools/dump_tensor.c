/* dump_tensor.c — print the first N fp32 elements of a tensor from a
 * GGUF file, using qw36's dequant path. Compare against llama.cpp's
 * `gguf-dump` to verify our Q4_K/Q5_K/Q6_K loaders are byte-correct.
 *
 * Build (run from repo root):
 *   cc -O2 -std=c11 -Icommon tools/dump_tensor.c \
 *      common/qw36.c common/qw36_gguf.c common/qw36_tokenizer.c \
 *      cpu/qw36_cpu_stub.c -lm -o qw36_dump_tensor
 *
 * Usage:
 *   ./qw36_dump_tensor <model.gguf> <tensor name> [N]
 * Example:
 *   ./qw36_dump_tensor Qwen3.5-0.8B-Q4_K_M.gguf blk.3.attn_q.weight 16
 *   ./qw36_dump_tensor Qwen3.5-0.8B-Q4_K_M.gguf token_embd.weight 1024
 */

#include "qw36.h"
#include "qw36_gguf.h"
#include "qw36_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

qw36_gpu_backend *qw36_backend_create(void) { return NULL; }

/* Subset of dq_q*_K reimplemented locally for the tool — keeps it
 * independent of qw36.c's static linkage. */
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp_ = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;
    uint32_t f;
    if (exp_ == 0) {
        if (mant == 0) { f = sign << 31; }
        else {
            int e = 1;
            while (!(mant & 0x400u)) { mant <<= 1; e--; }
            mant &= 0x3FFu;
            f = (sign << 31) | ((uint32_t)(e + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp_ == 0x1F) {
        f = (sign << 31) | (0xFFu << 23) | (mant << 13);
    } else {
        f = (sign << 31) | ((exp_ + (127 - 15)) << 23) | (mant << 13);
    }
    union { uint32_t u; float f; } u; u.u = f; return u.f;
}

#define QK_K 256

static void dq_q4_K(const uint8_t *blocks, float *out, size_t n) {
    const size_t nb = n / QK_K;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t *b = blocks + i * 144;
        uint16_t dh, dmh;
        memcpy(&dh, b, 2); memcpy(&dmh, b + 2, 2);
        float d = f16_to_f32(dh), dmn = f16_to_f32(dmh);
        const uint8_t *scales = b + 4, *qs = b + 16;
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            #define GS(jj) do { \
                if ((jj) < 4) { sc = scales[(jj)] & 63; m = scales[(jj)+4] & 63; } \
                else { sc = (scales[(jj)+4] & 0xF) | ((scales[(jj)-4] >> 6) << 4); \
                       m  = (scales[(jj)+4] >>  4) | ((scales[(jj)-0] >> 6) << 4); } \
            } while (0)
            GS(is + 0);
            float d1 = d * sc, m1 = dmn * m;
            GS(is + 1);
            float d2 = d * sc, m2 = dmn * m;
            #undef GS
            for (int l = 0; l < 32; l++) *out++ = d1 * (float)(qs[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++) *out++ = d2 * (float)(qs[l] >>  4) - m2;
            qs += 32; is += 2;
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <tensor name> [N=16]\n", argv[0]);
        return 2;
    }
    int N = (argc >= 4) ? atoi(argv[3]) : 16;

    char err[256] = {0};
    qw36_gguf_file *f = qw36_gguf_open(argv[1], err, sizeof(err));
    if (!f) { fprintf(stderr, "open: %s\n", err); return 1; }
    qw36_gguf_tensor t;
    if (qw36_gguf_get_tensor(f, argv[2], &t)) {
        fprintf(stderr, "tensor not found: %s\n", argv[2]);
        return 1;
    }
    printf("name: %s\n", argv[2]);
    printf("ggml_type: %u  dtype: %d\n", t.ggml_type, t.dtype);
    printf("dims: [%llu, %llu, %llu, %llu] (innermost first)\n",
        (unsigned long long)t.dims[0], (unsigned long long)t.dims[1],
        (unsigned long long)t.dims[2], (unsigned long long)t.dims[3]);

    size_t numel = 1;
    for (uint32_t d = 0; d < t.n_dims; d++) numel *= (size_t)t.dims[d];
    if ((size_t)N > numel) N = (int)numel;

    float *buf = (float *)calloc(N, sizeof(float));
    if (!buf) { perror("calloc"); return 1; }

    if (t.dtype == QW36_DTYPE_F32) {
        memcpy(buf, t.data, N * sizeof(float));
    } else if (t.dtype == QW36_DTYPE_Q4_K) {
        size_t blocks_needed = (size_t)((N + QK_K - 1) / QK_K);
        size_t big_n = blocks_needed * QK_K;
        float *big = (float *)calloc(big_n, sizeof(float));
        dq_q4_K((const uint8_t *)t.data, big, big_n);
        memcpy(buf, big, N * sizeof(float));
        free(big);
    } else {
        fprintf(stderr, "dtype %d not implemented in this dump tool — "
                "extend dq_q*K from common/qw36.c\n", t.dtype);
        return 1;
    }

    printf("first %d fp32:\n", N);
    for (int i = 0; i < N; i++) {
        printf("  [%4d] %12.6f\n", i, buf[i]);
    }

    free(buf);
    qw36_gguf_close(f);
    return 0;
}
