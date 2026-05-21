/* qw36_model.h - model-source abstraction used by the engine loader.
 *
 * GGUF remains the only inference-backed source today, but the engine now
 * binds tensors and metadata through this layer so MLX safetensors can be
 * added without threading a second file format through the forward code.
 */

#ifndef QW36_MODEL_H
#define QW36_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include "qw36.h"
#include "qw36_gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qw36_model_file qw36_model_file;

typedef enum {
    QW36_MODEL_FORMAT_GGUF = 1,
    QW36_MODEL_FORMAT_MLX  = 2
} qw36_model_format;

typedef struct {
    const char *name;
    qw36_dtype  dtype;
    uint32_t    storage_type;
    uint32_t    n_dims;
    uint64_t    dims[4];
    const void *data;
    size_t      nbytes;
} qw36_tensor_view;

qw36_model_file *qw36_model_open(const char *path, char *err, size_t err_cap);
void             qw36_model_close(qw36_model_file *m);

qw36_model_format qw36_model_format_of(const qw36_model_file *m);
const char       *qw36_model_path(const qw36_model_file *m);

int qw36_model_get_u32(const qw36_model_file *m, const char *key,
                       uint32_t *out);
int qw36_model_get_f32(const qw36_model_file *m, const char *key,
                       float *out);
int qw36_model_get_str(const qw36_model_file *m, const char *key,
                       const char **out);
int qw36_model_get_u32_array(const qw36_model_file *m, const char *key,
                             uint32_t *out, uint32_t cap);
int qw36_model_get_tensor(const qw36_model_file *m, const char *name,
                          qw36_tensor_view *out);

/* Legacy escape hatch for tokenizer/public APIs that still expose GGUF. */
qw36_gguf_file *qw36_model_as_gguf(const qw36_model_file *m);

#ifdef __cplusplus
}
#endif

#endif /* QW36_MODEL_H */
