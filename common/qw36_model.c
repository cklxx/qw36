/* qw36_model.c - model-source abstraction.
 *
 * The implementation intentionally wraps GGUF first. MLX safetensors needs
 * multi-tensor affine weights (weight/scales/biases), so it should enter here
 * as a source-side logical tensor conversion instead of leaking into forward.
 */

#include "qw36_model.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct qw36_model_file {
    qw36_model_format format;
    char             *path;
    qw36_gguf_file   *gguf;
};

static char *qw36__strdup(const char *s)
{
    if (!s) return NULL;
    size_t n = strlen(s) + 1u;
    char *p = (char *)malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

qw36_model_file *qw36_model_open(const char *path, char *err, size_t err_cap)
{
    if (!path) {
        if (err && err_cap) snprintf(err, err_cap, "missing model path");
        return NULL;
    }

    qw36_model_file *m = (qw36_model_file *)calloc(1, sizeof(*m));
    if (!m) {
        if (err && err_cap) snprintf(err, err_cap, "oom (model source)");
        return NULL;
    }
    m->path = qw36__strdup(path);
    if (!m->path) {
        free(m);
        if (err && err_cap) snprintf(err, err_cap, "oom (model path)");
        return NULL;
    }

    m->gguf = qw36_gguf_open(path, err, err_cap);
    if (!m->gguf) {
        free(m->path);
        free(m);
        return NULL;
    }
    m->format = QW36_MODEL_FORMAT_GGUF;
    return m;
}

void qw36_model_close(qw36_model_file *m)
{
    if (!m) return;
    if (m->gguf) qw36_gguf_close(m->gguf);
    free(m->path);
    free(m);
}

qw36_model_format qw36_model_format_of(const qw36_model_file *m)
{
    return m ? m->format : 0;
}

const char *qw36_model_path(const qw36_model_file *m)
{
    return m ? m->path : NULL;
}

int qw36_model_get_u32(const qw36_model_file *m, const char *key,
                       uint32_t *out)
{
    if (!m || !m->gguf) return -1;
    return qw36_gguf_get_u32(m->gguf, key, out);
}

int qw36_model_get_f32(const qw36_model_file *m, const char *key,
                       float *out)
{
    if (!m || !m->gguf) return -1;
    return qw36_gguf_get_f32(m->gguf, key, out);
}

int qw36_model_get_str(const qw36_model_file *m, const char *key,
                       const char **out)
{
    if (!m || !m->gguf) return -1;
    return qw36_gguf_get_str(m->gguf, key, out);
}

int qw36_model_get_u32_array(const qw36_model_file *m, const char *key,
                             uint32_t *out, uint32_t cap)
{
    if (!m || !m->gguf) return -1;
    return qw36_gguf_get_u32_array(m->gguf, key, out, cap);
}

int qw36_model_get_tensor(const qw36_model_file *m, const char *name,
                          qw36_tensor_view *out)
{
    if (!m || !m->gguf || !out) return -1;
    qw36_gguf_tensor t;
    if (qw36_gguf_get_tensor(m->gguf, name, &t)) return -1;
    out->name         = t.name;
    out->dtype        = t.dtype;
    out->storage_type = t.ggml_type;
    out->n_dims       = t.n_dims;
    memcpy(out->dims, t.dims, sizeof(out->dims));
    out->data         = t.data;
    out->nbytes       = t.nbytes;
    return 0;
}

qw36_gguf_file *qw36_model_as_gguf(const qw36_model_file *m)
{
    return m ? m->gguf : NULL;
}
