/* qw36_safetensors.h - minimal safetensors mmap reader.
 *
 * This reader intentionally covers only the model-weight subset we need:
 * top-level tensor entries with dtype, shape, and data_offsets. It keeps the
 * file mmapped and returns tensor data pointers into that mapping.
 */

#ifndef QW36_SAFETENSORS_H
#define QW36_SAFETENSORS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qw36_safetensors_file qw36_safetensors_file;

typedef enum {
    QW36_ST_DTYPE_UNKNOWN = 0,
    QW36_ST_DTYPE_BOOL,
    QW36_ST_DTYPE_U8,
    QW36_ST_DTYPE_U16,
    QW36_ST_DTYPE_U32,
    QW36_ST_DTYPE_U64,
    QW36_ST_DTYPE_I8,
    QW36_ST_DTYPE_I16,
    QW36_ST_DTYPE_I32,
    QW36_ST_DTYPE_I64,
    QW36_ST_DTYPE_F8_E4M3,
    QW36_ST_DTYPE_F8_E5M2,
    QW36_ST_DTYPE_F16,
    QW36_ST_DTYPE_BF16,
    QW36_ST_DTYPE_F32,
    QW36_ST_DTYPE_F64
} qw36_safetensors_dtype;

typedef struct {
    const char *name;       /* owned by file, null-terminated */
    const char *dtype_name; /* owned by file, original dtype string */
    qw36_safetensors_dtype dtype;

    uint32_t n_dims;
    uint64_t dims[8];

    uint64_t data_offsets[2]; /* relative to safetensors data section */
    const void *data;         /* mmap-backed; lifetime = file */
    size_t nbytes;
} qw36_safetensors_tensor;

qw36_safetensors_file *qw36_safetensors_open(const char *path,
                                             char *err, size_t err_cap);
void qw36_safetensors_close(qw36_safetensors_file *f);

size_t qw36_safetensors_tensor_count(const qw36_safetensors_file *f);
int qw36_safetensors_get_tensor_by_index(const qw36_safetensors_file *f,
                                         size_t i,
                                         qw36_safetensors_tensor *out);
int qw36_safetensors_get_tensor(const qw36_safetensors_file *f,
                                const char *name,
                                qw36_safetensors_tensor *out);

size_t qw36_safetensors_header_len(const qw36_safetensors_file *f);
const void *qw36_safetensors_data_base(const qw36_safetensors_file *f);
const char *qw36_safetensors_dtype_name(qw36_safetensors_dtype dtype);

#ifdef __cplusplus
}
#endif

#endif /* QW36_SAFETENSORS_H */
