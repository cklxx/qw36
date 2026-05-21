/* dump_tensor.c - print one dequantized tensor row using qw36's row path.
 *
 * Build (run from repo root):
 *   cc -O2 -std=c11 -Icommon tools/dump_tensor.c \
 *      common/qw36_gguf.c common/qw36_dequant.c -lm -o qw36_dump_tensor
 *
 * Usage:
 *   ./qw36_dump_tensor <model.gguf> <tensor name> [N=16] [row=0]
 */

#include "qw36_internal.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <tensor name> [N=16] [row=0]\n",
                argv[0]);
        return 2;
    }

    int N = (argc >= 4) ? atoi(argv[3]) : 16;
    if (N <= 0) N = 16;
    size_t row = (argc >= 5) ? strtoull(argv[4], NULL, 10) : 0;

    char err[256] = {0};
    qw36_gguf_file *f = qw36_gguf_open(argv[1], err, sizeof(err));
    if (!f) {
        fprintf(stderr, "open: %s\n", err);
        return 1;
    }

    qw36_gguf_tensor t;
    if (qw36_gguf_get_tensor(f, argv[2], &t)) {
        fprintf(stderr, "tensor not found: %s\n", argv[2]);
        qw36_gguf_close(f);
        return 1;
    }

    size_t cols = t.n_dims >= 1 ? (size_t)t.dims[0] : 1;
    size_t rows = t.n_dims >= 2 ? (size_t)t.dims[1] : 1;
    size_t extra = t.n_dims >= 3 ? (size_t)t.dims[2] : 0;
    size_t total_rows = rows * (extra ? extra : 1u);
    if (row >= total_rows) {
        fprintf(stderr, "row %zu exceeds tensor row count %zu\n", row, total_rows);
        qw36_gguf_close(f);
        return 1;
    }
    if ((size_t)N > cols) N = (int)cols;

    qw36_lazy_w lw = {
        .data = t.data,
        .dtype = t.dtype,
        .ggml_type = t.ggml_type,
        .rows = rows,
        .cols = cols,
        .n_extra = extra,
        .gpu_buf = NULL,
    };

    float *buf = (float *)calloc(cols, sizeof(float));
    if (!buf) {
        perror("calloc");
        qw36_gguf_close(f);
        return 1;
    }
    if (qw36__dequant_row(&lw, row, buf)) {
        fprintf(stderr, "dtype %d is unsupported for row dequant\n", t.dtype);
        free(buf);
        qw36_gguf_close(f);
        return 1;
    }

    printf("name: %s\n", argv[2]);
    printf("ggml_type: %u  dtype: %d\n", t.ggml_type, t.dtype);
    printf("dims: [%llu, %llu, %llu, %llu] (innermost first)\n",
           (unsigned long long)t.dims[0], (unsigned long long)t.dims[1],
           (unsigned long long)t.dims[2], (unsigned long long)t.dims[3]);
    printf("(row %zu offset)\n", row);
    printf("first %d fp32:\n", N);
    for (int i = 0; i < N; i++) {
        printf("  [%4d] %.9g\n", i, buf[i]);
    }

    free(buf);
    qw36_gguf_close(f);
    return 0;
}
