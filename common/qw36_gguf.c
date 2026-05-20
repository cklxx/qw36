/* qw36_gguf.c — GGUF v3 loader (stub).
 *
 * Owner: Claude. Implementation follows ggerganov/ggml gguf.c, slimmed.
 */

#include "qw36_gguf.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct qw36_gguf_file {
    int   fd;
    void *map;
    size_t map_len;
    /* TODO(Claude): kv table, tensor table. */
};

qw36_gguf_file *qw36_gguf_open(const char *path, char *err, size_t err_cap)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        int e = errno;
        if (err && err_cap) snprintf(err, err_cap, "open(%s): %s",
                                     path, strerror(e));
        return NULL;
    }
    /* TODO(Claude): mmap, parse magic+version, read KV section, read
     * tensor infos, expose accessors. */
    close(fd);
    if (err && err_cap) snprintf(err, err_cap, "qw36_gguf_open: TODO");
    return NULL;
}

void qw36_gguf_close(qw36_gguf_file *f)
{
    if (!f) return;
    if (f->map && f->map != MAP_FAILED) munmap(f->map, f->map_len);
    if (f->fd >= 0) close(f->fd);
    free(f);
}

int qw36_gguf_get_u32(const qw36_gguf_file *f, const char *key, uint32_t *out)
{ (void)f; (void)key; (void)out; return -1; /* TODO(Claude) */ }

int qw36_gguf_get_f32(const qw36_gguf_file *f, const char *key, float *out)
{ (void)f; (void)key; (void)out; return -1; /* TODO(Claude) */ }

int qw36_gguf_get_str(const qw36_gguf_file *f, const char *key, const char **out)
{ (void)f; (void)key; (void)out; return -1; /* TODO(Claude) */ }

int qw36_gguf_get_tensor(const qw36_gguf_file *f, const char *name,
                         qw36_gguf_tensor *out)
{ (void)f; (void)name; (void)out; return -1; /* TODO(Claude) */ }
