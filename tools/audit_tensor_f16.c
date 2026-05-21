/* audit_tensor_f16.c - materialize one tensor through qw36's f16 row path.
 *
 * Usage:
 *   ./qw36_audit_tensor_f16 <model.gguf> <tensor name> [expected_rows]
 */

#include "qw36_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int check_row(const qw36_lazy_w *w, const uint16_t *f16,
                     size_t row_idx, size_t cols)
{
    float *ref = (float *)malloc(cols * sizeof(float));
    if (!ref) return -1;
    if (qw36__dequant_row(w, row_idx, ref)) {
        free(ref);
        return -1;
    }
    const uint16_t *got = f16 + row_idx * cols;
    float max_abs = 0.0f;
    for (size_t c = 0; c < cols; c++) {
        float d = fabsf(qw36__f16_to_f32(got[c]) - ref[c]);
        if (d > max_abs) max_abs = d;
    }
    free(ref);
    printf("ok: row %zu f16 materialize max_abs %.6g\n", row_idx, max_abs);
    return max_abs <= 5.0e-5f ? 0 : -1;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <tensor name> [expected_rows]\n",
                argv[0]);
        return 2;
    }

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

    qw36_lazy_w lw = {
        .data = t.data,
        .dtype = t.dtype,
        .ggml_type = t.ggml_type,
        .rows = t.n_dims >= 2 ? t.dims[1] : 1,
        .cols = t.n_dims >= 1 ? t.dims[0] : 0,
        .n_extra = t.n_dims >= 3 ? t.dims[2] : 0,
        .gpu_buf = NULL,
    };

    size_t rows = 0, cols = 0, numel = 0;
    if (qw36__lazy_w_shape(&lw, &rows, &cols, &numel)) {
        fprintf(stderr, "invalid tensor shape\n");
        qw36_gguf_close(f);
        return 1;
    }
    if (argc >= 4) {
        size_t expected = strtoull(argv[3], NULL, 10);
        if (rows != expected) {
            fprintf(stderr, "row count mismatch: got %zu expected %zu\n",
                    rows, expected);
            qw36_gguf_close(f);
            return 1;
        }
    }

    printf("tensor: %s rows=%zu cols=%zu f16_bytes=%zu\n",
           argv[2], rows, cols, numel * sizeof(uint16_t));
    uint16_t *f16 = qw36__materialize_f16_rows(&lw);
    if (!f16) {
        fprintf(stderr, "f16 materialize failed\n");
        qw36_gguf_close(f);
        return 1;
    }

    int rc = 0;
    if (check_row(&lw, f16, 0, cols)) rc = 1;
    if (rows > 2 && check_row(&lw, f16, rows / 2, cols)) rc = 1;
    if (rows > 1 && check_row(&lw, f16, rows - 1, cols)) rc = 1;

    free(f16);
    qw36_gguf_close(f);
    return rc;
}
