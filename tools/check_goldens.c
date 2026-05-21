/* tools/check_goldens.c — load a golden bin, rerun the kernel on its
 * inputs, diff against the stored outputs. Print first 8 mismatches
 * with rtol/atol.
 *
 * Exits 0 on pass, 1 on mismatch beyond tolerance, 2 on malformed file.
 *
 * v0 scope: same CPU primitives as gen_goldens. Cross-backend checking
 * (Metal/CUDA running the same kernels and producing the same outputs)
 * lands on top of this — a future check_goldens_metal binary will
 * import the same fixtures but route through the GPU vtable. */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QW36_GOLDEN_MAGIC   0x51444C47u
#define QW36_GOLDEN_VERSION 1u

typedef enum {
    QW36_GOLDEN_RMSNORM = 1,
    QW36_GOLDEN_SWIGLU  = 2,
    QW36_GOLDEN_SILU    = 3,
} qw36_golden_kernel;

static float silu_f32(float x) { return x / (1.0f + expf(-x)); }

static void rmsnorm_f32(float *out, const float *x, const float *w,
                        size_t n, float eps) {
    double ss = 0.0;
    for (size_t i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float scale = (float)(1.0 / sqrt(ss / (double)n + (double)eps));
    for (size_t i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}

static int diff_floats(const float *a, const float *b, size_t n,
                       float rtol, float atol, const char *label) {
    size_t shown = 0, total_bad = 0;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        float tol = atol + rtol * fmaxf(fabsf(a[i]), fabsf(b[i]));
        if (d > tol) {
            total_bad++;
            if (shown < 8) {
                fprintf(stderr, "  [%s] mismatch at %zu: got=%.8g expected=%.8g delta=%.3e tol=%.3e\n",
                        label, i, a[i], b[i], d, tol);
                shown++;
            }
        }
    }
    if (total_bad) {
        fprintf(stderr, "  [%s] %zu / %zu elements outside tol (rtol=%.2e atol=%.2e)\n",
                label, total_bad, n, rtol, atol);
        return 1;
    }
    fprintf(stderr, "  [%s] %zu elements match (rtol=%.2e atol=%.2e)\n",
            label, n, rtol, atol);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <fixture.bin> [rtol] [atol]\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    float rtol = argc > 2 ? (float)atof(argv[2]) : 1e-5f;
    float atol = argc > 3 ? (float)atof(argv[3]) : 1e-6f;

    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 2; }

    uint32_t hdr[5];
    if (fread(hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return 2; }
    if (hdr[0] != QW36_GOLDEN_MAGIC || hdr[1] != QW36_GOLDEN_VERSION) {
        fprintf(stderr, "bad magic/version\n"); fclose(f); return 2;
    }
    qw36_golden_kernel kernel = (qw36_golden_kernel)hdr[2];
    uint32_t n_inputs  = hdr[3];
    uint32_t n_outputs = hdr[4];

    uint32_t *in_dims  = (uint32_t *)malloc(n_inputs  * sizeof(uint32_t));
    uint32_t *out_dims = (uint32_t *)malloc(n_outputs * sizeof(uint32_t));
    if (!in_dims || !out_dims) { fclose(f); return 2; }
    fread(in_dims,  sizeof(uint32_t), n_inputs,  f);
    fread(out_dims, sizeof(uint32_t), n_outputs, f);
    uint64_t seed; fread(&seed, sizeof(seed), 1, f);

    float **inputs  = (float **)calloc(n_inputs,  sizeof(float *));
    float **outputs = (float **)calloc(n_outputs, sizeof(float *));
    for (uint32_t i = 0; i < n_inputs; i++) {
        inputs[i] = (float *)malloc(in_dims[i] * sizeof(float));
        fread(inputs[i], sizeof(float), in_dims[i], f);
    }
    for (uint32_t i = 0; i < n_outputs; i++) {
        outputs[i] = (float *)malloc(out_dims[i] * sizeof(float));
        fread(outputs[i], sizeof(float), out_dims[i], f);
    }
    fclose(f);

    /* Rerun the kernel on `inputs` and compare to `outputs`. */
    int rc = 0;
    switch (kernel) {
    case QW36_GOLDEN_RMSNORM: {
        if (n_inputs != 2 || n_outputs != 1 ||
            in_dims[0] != in_dims[1] || in_dims[0] != out_dims[0]) {
            fprintf(stderr, "rmsnorm fixture shape mismatch\n"); rc = 2; break;
        }
        float *y = (float *)malloc(out_dims[0] * sizeof(float));
        rmsnorm_f32(y, inputs[0], inputs[1], out_dims[0], 1e-6f);
        rc = diff_floats(y, outputs[0], out_dims[0], rtol, atol, "rmsnorm");
        free(y);
        break;
    }
    case QW36_GOLDEN_SILU: {
        if (n_inputs != 1 || n_outputs != 1 || in_dims[0] != out_dims[0]) {
            fprintf(stderr, "silu fixture shape mismatch\n"); rc = 2; break;
        }
        float *y = (float *)malloc(out_dims[0] * sizeof(float));
        for (uint32_t i = 0; i < out_dims[0]; i++)
            y[i] = silu_f32(inputs[0][i]);
        rc = diff_floats(y, outputs[0], out_dims[0], rtol, atol, "silu");
        free(y);
        break;
    }
    case QW36_GOLDEN_SWIGLU: {
        if (n_inputs != 2 || n_outputs != 1 ||
            in_dims[0] != in_dims[1] || in_dims[0] != out_dims[0]) {
            fprintf(stderr, "swiglu fixture shape mismatch\n"); rc = 2; break;
        }
        float *y = (float *)malloc(out_dims[0] * sizeof(float));
        for (uint32_t i = 0; i < out_dims[0]; i++)
            y[i] = silu_f32(inputs[0][i]) * inputs[1][i];
        rc = diff_floats(y, outputs[0], out_dims[0], rtol, atol, "swiglu");
        free(y);
        break;
    }
    default:
        fprintf(stderr, "unknown kernel id %u\n", (unsigned)kernel);
        rc = 2;
    }

    for (uint32_t i = 0; i < n_inputs;  i++) free(inputs[i]);
    for (uint32_t i = 0; i < n_outputs; i++) free(outputs[i]);
    free(inputs); free(outputs); free(in_dims); free(out_dims);
    (void)seed;
    return rc;
}
