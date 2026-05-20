/* qw36_gguf.h — minimal GGUF v3 reader.
 *
 * We do not support every GGUF variant. Just enough to load Qwen 3.6
 * dense + MoE checkpoints in F32 / F16 / BF16 / Q8_0 / Q4_K / Q2_K.
 */

#ifndef QW36_GGUF_H
#define QW36_GGUF_H

#include <stddef.h>
#include <stdint.h>

#include "qw36.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qw36_gguf_file qw36_gguf_file;

typedef struct {
    const char *name;
    qw36_dtype  dtype;
    uint32_t    ggml_type;  /* raw GGML_TYPE_* — useful for diagnostics */
    uint32_t    n_dims;
    uint64_t    dims[4];
    const void *data;     /* mmap-backed; lifetime = file */
    size_t      nbytes;
} qw36_gguf_tensor;

qw36_gguf_file *qw36_gguf_open(const char *path, char *err, size_t err_cap);
void            qw36_gguf_close(qw36_gguf_file *f);

/* Metadata accessors. Return 0 on success, -1 if key is absent. */
int qw36_gguf_get_u32(const qw36_gguf_file *f, const char *key, uint32_t *out);
int qw36_gguf_get_f32(const qw36_gguf_file *f, const char *key, float *out);
int qw36_gguf_get_str(const qw36_gguf_file *f, const char *key,
                      const char **out);
/* Read an array of int32 / uint32 values into a user-provided buffer.
 * On success returns the number of elements actually written (≤ cap).
 * On absence or type mismatch returns -1. */
int qw36_gguf_get_u32_array(const qw36_gguf_file *f, const char *key,
                            uint32_t *out, uint32_t cap);

/* Tensor lookup. Returned struct points into the file (no copy). */
int qw36_gguf_get_tensor(const qw36_gguf_file *f, const char *name,
                         qw36_gguf_tensor *out);

/* Enumeration. Returns total tensor count; fill (*out) at index i. */
size_t qw36_gguf_tensor_count(const qw36_gguf_file *f);
int    qw36_gguf_get_tensor_by_index(const qw36_gguf_file *f, size_t i,
                                     qw36_gguf_tensor *out);

#ifdef __cplusplus
}
#endif

#endif /* QW36_GGUF_H */
