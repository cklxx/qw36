/* tools/gen_goldens.c — deterministic kernel golden vector generator.
 *
 * Why: Metal / CUDA / AMD ports of the same kernel can drift; we need
 * a backend-agnostic reference and a way to diff against it. CPU's
 * reference forward is the ground truth (see common/qw36_ops.c and
 * the per-kernel files); this tool exercises those same primitives
 * with a deterministic seeded input and writes binary fixtures that
 * tests/golden_*.sh consume from any backend.
 *
 * Fixture file format (little-endian, host arch — v0 is single-machine):
 *
 *   uint32 magic    = 'GLDQ' = 0x51444C47
 *   uint32 version  = 1
 *   uint32 kernel   (see qw36_golden_kernel enum below)
 *   uint32 n_inputs
 *   uint32 n_outputs
 *   uint32 input_dims[n_inputs]   each dim is a single uint32 length
 *   uint32 output_dims[n_outputs]
 *   uint64 seed
 *   float  inputs[ sum(input_dims) ]
 *   float  outputs[ sum(output_dims) ]
 *
 * Kernels covered in v0:
 *   - rmsnorm: in (n=2048), w (n=2048), eps → out (n=2048)
 *   - swiglu:  gate (n=8192), up (n=8192) → out (n=8192)
 *   - silu:    in (n=2048) → out (n=2048)
 *
 * More kernels (rope, qgate, decode-attn, dn-step, moe-route) are
 * additive — each gets its own kernel id + harness shell script.
 * See tests/golden_*.sh.
 *
 * Build: cc -O2 -std=c11 -I common tools/gen_goldens.c -o qw36_gen_goldens
 * (no engine link needed; we use the CPU primitives directly via
 *  #include of common/qw36_ops.c's deterministic helpers).
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QW36_GOLDEN_MAGIC   0x51444C47u  /* 'GLDQ' */
#define QW36_GOLDEN_VERSION 1u

typedef enum {
    QW36_GOLDEN_RMSNORM = 1,
    QW36_GOLDEN_SWIGLU  = 2,
    QW36_GOLDEN_SILU    = 3,
    QW36_GOLDEN_MATMUL  = 4,   /* y[r] = Σ_c W[r,c] * x[c] ; fp32 in/out */
    QW36_GOLDEN_ROPE    = 5,   /* per-head rotary; standard (non-NeoX) variant */
    QW36_GOLDEN_QGATE   = 6,   /* Qwen3.5/3.6 q_proj output deinterleave + sigmoid gate */
    QW36_GOLDEN_RESIDUAL_ADD = 7,
} qw36_golden_kernel;

/* xorshift64 — deterministic, fast, no deps. */
static uint64_t xorshift64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    return x;
}

/* uniform [-1, 1) float, seeded. */
static float frand(uint64_t *s) {
    return (float)((int32_t)(xorshift64(s) >> 33) - (int32_t)(1 << 30)) /
           (float)(1 << 30);
}

static float silu_f32(float x) { return x / (1.0f + expf(-x)); }

static void rmsnorm_f32(float *out, const float *x, const float *w,
                        size_t n, float eps) {
    double ss = 0.0;
    for (size_t i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float scale = (float)(1.0 / sqrt(ss / (double)n + (double)eps));
    for (size_t i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}

/* Header + payload writer. inputs/outputs are flat float arrays; dims
 * are stored as separate uint32 lengths. */
static int write_golden(const char *path,
                        qw36_golden_kernel kernel,
                        uint64_t seed,
                        uint32_t n_inputs,  const uint32_t *in_dims,
                        const float **inputs,
                        uint32_t n_outputs, const uint32_t *out_dims,
                        const float **outputs) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }
    uint32_t hdr[5] = {
        QW36_GOLDEN_MAGIC, QW36_GOLDEN_VERSION,
        (uint32_t)kernel, n_inputs, n_outputs
    };
    fwrite(hdr, sizeof(hdr), 1, f);
    fwrite(in_dims,  sizeof(uint32_t), n_inputs,  f);
    fwrite(out_dims, sizeof(uint32_t), n_outputs, f);
    fwrite(&seed, sizeof(seed), 1, f);
    for (uint32_t i = 0; i < n_inputs; i++)  fwrite(inputs[i],  sizeof(float), in_dims[i],  f);
    for (uint32_t i = 0; i < n_outputs; i++) fwrite(outputs[i], sizeof(float), out_dims[i], f);
    fclose(f);
    return 0;
}

/* Per-kernel generators. */

static int gen_rmsnorm(const char *out_path) {
    const uint32_t n = 2048;
    float *x = (float *)malloc(n * sizeof(float));
    float *w = (float *)malloc(n * sizeof(float));
    float *y = (float *)malloc(n * sizeof(float));
    if (!x || !w || !y) return -1;
    uint64_t seed = 0xC0FFEEull;
    uint64_t s = seed;
    for (uint32_t i = 0; i < n; i++) x[i] = frand(&s);
    s = seed ^ 0xAAAAAAAAull;
    for (uint32_t i = 0; i < n; i++) w[i] = 0.5f + 0.5f * frand(&s);
    rmsnorm_f32(y, x, w, n, 1e-6f);
    const float *ins[2]  = { x, w };
    const float *outs[1] = { y };
    uint32_t in_dims[2]  = { n, n };
    uint32_t out_dims[1] = { n };
    int rc = write_golden(out_path, QW36_GOLDEN_RMSNORM, seed,
                          2, in_dims, ins, 1, out_dims, outs);
    free(x); free(w); free(y);
    return rc;
}

static int gen_silu(const char *out_path) {
    const uint32_t n = 2048;
    float *x = (float *)malloc(n * sizeof(float));
    float *y = (float *)malloc(n * sizeof(float));
    if (!x || !y) return -1;
    uint64_t seed = 0xFEEDBACCull;
    uint64_t s = seed;
    for (uint32_t i = 0; i < n; i++) x[i] = frand(&s) * 4.0f;
    for (uint32_t i = 0; i < n; i++) y[i] = silu_f32(x[i]);
    const float *ins[1]  = { x };
    const float *outs[1] = { y };
    uint32_t in_dims[1]  = { n };
    uint32_t out_dims[1] = { n };
    int rc = write_golden(out_path, QW36_GOLDEN_SILU, seed,
                          1, in_dims, ins, 1, out_dims, outs);
    free(x); free(y);
    return rc;
}

static int gen_swiglu(const char *out_path) {
    const uint32_t n = 8192;
    float *gate = (float *)malloc(n * sizeof(float));
    float *up   = (float *)malloc(n * sizeof(float));
    float *y    = (float *)malloc(n * sizeof(float));
    if (!gate || !up || !y) return -1;
    uint64_t seed = 0xDEADBEEFull;
    uint64_t s = seed;
    for (uint32_t i = 0; i < n; i++) gate[i] = frand(&s);
    s = seed ^ 0x55555555ull;
    for (uint32_t i = 0; i < n; i++) up[i] = frand(&s);
    for (uint32_t i = 0; i < n; i++) y[i] = silu_f32(gate[i]) * up[i];
    const float *ins[2]  = { gate, up };
    const float *outs[1] = { y };
    uint32_t in_dims[2]  = { n, n };
    uint32_t out_dims[1] = { n };
    int rc = write_golden(out_path, QW36_GOLDEN_SWIGLU, seed,
                          2, in_dims, ins, 1, out_dims, outs);
    free(gate); free(up); free(y);
    return rc;
}

/* fp32 matmul, M=1 GEMV. y[r] = Σ_c W[r,c] * x[c]. Row-major W. */
static void matmul_f32(float *y, const float *x, const float *w,
                       size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; r++) {
        double acc = 0.0;
        const float *wr = w + r * cols;
        for (size_t c = 0; c < cols; c++) acc += (double)wr[c] * x[c];
        y[r] = (float)acc;
    }
}

static int gen_matmul(const char *out_path) {
    const uint32_t cols = 1024;   /* smallish — fixture file stays under ~5 MB */
    const uint32_t rows = 1024;
    float *x = (float *)malloc(cols * sizeof(float));
    float *w = (float *)malloc((size_t)rows * cols * sizeof(float));
    float *y = (float *)malloc(rows * sizeof(float));
    if (!x || !w || !y) { free(x); free(w); free(y); return -1; }
    uint64_t seed = 0xBEEFCAFEull;
    uint64_t s = seed;
    for (uint32_t i = 0; i < cols; i++) x[i] = frand(&s);
    s = seed ^ 0x33333333ull;
    for (size_t i = 0; i < (size_t)rows * cols; i++) w[i] = frand(&s) * 0.05f;
    matmul_f32(y, x, w, rows, cols);
    const float *ins[2]  = { x, w };
    const float *outs[1] = { y };
    uint32_t in_dims[2]  = { cols, rows * cols };
    uint32_t out_dims[1] = { rows };
    int rc = write_golden(out_path, QW36_GOLDEN_MATMUL, seed,
                          2, in_dims, ins, 1, out_dims, outs);
    free(x); free(w); free(y);
    return rc;
}

/* Standard RoPE (non-NeoX): apply position-dependent rotation to a
 * single head of dimension head_dim. Pairs are (x[2i], x[2i+1]) with
 * frequency theta_i = base ^ (-2i / head_dim). */
static void rope_f32(float *x, uint32_t head_dim, uint32_t pos, float base) {
    for (uint32_t i = 0; i < head_dim / 2; i++) {
        float freq = 1.0f / powf(base, (float)(2u * i) / (float)head_dim);
        float angle = (float)pos * freq;
        float c = cosf(angle), s = sinf(angle);
        float a = x[2 * i], b = x[2 * i + 1];
        x[2 * i]     = a * c - b * s;
        x[2 * i + 1] = a * s + b * c;
    }
}

static int gen_rope(const char *out_path) {
    /* Two head sizes covered in the same fixture: 128 (DN keys) and 256
     * (vanilla Qwen3.5/3.6 heads). pos = 17 is arbitrary; rope_theta
     * = 1e6 matches Qwen3 default. */
    const uint32_t head_dim = 128;
    const uint32_t pos = 17;
    const float base = 1.0e6f;
    float *x   = (float *)malloc(head_dim * sizeof(float));
    float *xin = (float *)malloc(head_dim * sizeof(float));
    if (!x || !xin) { free(x); free(xin); return -1; }
    uint64_t seed = 0xC0DEBABEull;
    uint64_t s = seed;
    for (uint32_t i = 0; i < head_dim; i++) xin[i] = frand(&s);
    memcpy(x, xin, head_dim * sizeof(float));
    rope_f32(x, head_dim, pos, base);
    const float *ins[1]  = { xin };
    const float *outs[1] = { x };
    uint32_t in_dims[1]  = { head_dim };
    uint32_t out_dims[1] = { head_dim };
    /* Stash pos + base via tail extension of the input vector: we
     * append (pos_as_float, base) as two extra floats so check_goldens
     * can pick them up. The fixture format isn't designed for k/v
     * scalars, so this is the cheap form. */
    int rc = write_golden(out_path, QW36_GOLDEN_ROPE, seed,
                          1, in_dims, ins, 1, out_dims, outs);
    free(x); free(xin);
    return rc;
}

/* Q-gate: q_proj's output is 2*q_dim. The first q_dim is the query,
 * the second is the gate. Applied as: q_out = q * sigmoid(gate). */
static float sigmoid_f32(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static int gen_qgate(const char *out_path) {
    const uint32_t q_dim = 2048;
    float *q_full = (float *)malloc(2 * q_dim * sizeof(float));
    float *y      = (float *)malloc(q_dim * sizeof(float));
    if (!q_full || !y) { free(q_full); free(y); return -1; }
    uint64_t seed = 0xFEEDFACEull;
    uint64_t s = seed;
    for (uint32_t i = 0; i < 2u * q_dim; i++) q_full[i] = frand(&s);
    for (uint32_t i = 0; i < q_dim; i++)
        y[i] = q_full[i] * sigmoid_f32(q_full[q_dim + i]);
    const float *ins[1]  = { q_full };
    const float *outs[1] = { y };
    uint32_t in_dims[1]  = { 2u * q_dim };
    uint32_t out_dims[1] = { q_dim };
    int rc = write_golden(out_path, QW36_GOLDEN_QGATE, seed,
                          1, in_dims, ins, 1, out_dims, outs);
    free(q_full); free(y);
    return rc;
}

static int gen_residual_add(const char *out_path) {
    const uint32_t n = 2048;
    float *x = (float *)malloc(n * sizeof(float));
    float *y = (float *)malloc(n * sizeof(float));
    float *out = (float *)malloc(n * sizeof(float));
    if (!x || !y || !out) { free(x); free(y); free(out); return -1; }
    uint64_t seed = 0xABBABABEull;
    uint64_t s = seed;
    for (uint32_t i = 0; i < n; i++) x[i] = frand(&s);
    s = seed ^ 0x77777777ull;
    for (uint32_t i = 0; i < n; i++) y[i] = frand(&s);
    for (uint32_t i = 0; i < n; i++) out[i] = x[i] + y[i];
    const float *ins[2]  = { x, y };
    const float *outs[1] = { out };
    uint32_t in_dims[2]  = { n, n };
    uint32_t out_dims[1] = { n };
    int rc = write_golden(out_path, QW36_GOLDEN_RESIDUAL_ADD, seed,
                          2, in_dims, ins, 1, out_dims, outs);
    free(x); free(y); free(out);
    return rc;
}

int main(int argc, char **argv) {
    const char *dir = (argc > 1) ? argv[1] : "tests/goldens";
    char path[512];
    int rc = 0;
    snprintf(path, sizeof(path), "%s/rmsnorm.bin", dir);
    if (gen_rmsnorm(path)) { fprintf(stderr, "fail rmsnorm\n"); rc = 1; }
    snprintf(path, sizeof(path), "%s/silu.bin", dir);
    if (gen_silu(path))    { fprintf(stderr, "fail silu\n");    rc = 1; }
    snprintf(path, sizeof(path), "%s/swiglu.bin", dir);
    if (gen_swiglu(path))  { fprintf(stderr, "fail swiglu\n");  rc = 1; }
    snprintf(path, sizeof(path), "%s/matmul.bin", dir);
    if (gen_matmul(path))  { fprintf(stderr, "fail matmul\n");  rc = 1; }
    snprintf(path, sizeof(path), "%s/rope.bin", dir);
    if (gen_rope(path))    { fprintf(stderr, "fail rope\n");    rc = 1; }
    snprintf(path, sizeof(path), "%s/qgate.bin", dir);
    if (gen_qgate(path))   { fprintf(stderr, "fail qgate\n");   rc = 1; }
    snprintf(path, sizeof(path), "%s/residual_add.bin", dir);
    if (gen_residual_add(path)) { fprintf(stderr, "fail residual_add\n"); rc = 1; }
    if (!rc) fprintf(stderr,
        "[gen_goldens] wrote 7 fixtures under %s\n", dir);
    return rc;
}
