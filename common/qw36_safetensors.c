/* qw36_safetensors.c - minimal safetensors mmap reader.
 *
 * Format:
 *   u64le header_len
 *   JSON header, padded with spaces
 *   raw tensor bytes
 *
 * Tensor data_offsets are relative to the first byte after the JSON header.
 * This module parses enough JSON to enumerate tensor objects and skips
 * __metadata__ without materializing it.
 */

#include "qw36_safetensors.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    char *name;
    char *dtype_name;
    qw36_safetensors_dtype dtype;
    uint32_t n_dims;
    uint64_t dims[8];
    uint64_t data_offsets[2];
    const void *data;
    size_t nbytes;
} st_tensor_info;

struct qw36_safetensors_file {
    int fd;
    const uint8_t *map;
    size_t map_len;

    const uint8_t *header;
    size_t header_len;
    const uint8_t *data_start;

    st_tensor_info *tensors;
    size_t n_tensors;
    size_t cap_tensors;
};

typedef struct {
    const uint8_t *p;
    const uint8_t *start;
    const uint8_t *end;
    int error;
    const char *why;
} json_cur;

static void json_fail(json_cur *c, const char *why) {
    if (!c->error) {
        c->error = 1;
        c->why = why;
    }
}

static void skip_ws(json_cur *c) {
    while (c->p < c->end) {
        uint8_t ch = *c->p;
        if (ch != ' ' && ch != '\n' && ch != '\r' && ch != '\t') break;
        c->p++;
    }
}

static int consume(json_cur *c, char want) {
    skip_ws(c);
    if (c->p >= c->end || *c->p != (uint8_t)want) {
        json_fail(c, "unexpected token");
        return -1;
    }
    c->p++;
    return 0;
}

static int append_byte(char **buf, size_t *len, size_t *cap, char ch) {
    if (*len + 1 >= *cap) {
        size_t ncap = *cap ? *cap * 2 : 32;
        char *nbuf = (char *)realloc(*buf, ncap);
        if (!nbuf) return -1;
        *buf = nbuf;
        *cap = ncap;
    }
    (*buf)[(*len)++] = ch;
    return 0;
}

static int hex_val(uint8_t ch) {
    if (ch >= '0' && ch <= '9') return (int)(ch - '0');
    if (ch >= 'a' && ch <= 'f') return (int)(ch - 'a') + 10;
    if (ch >= 'A' && ch <= 'F') return (int)(ch - 'A') + 10;
    return -1;
}

static int parse_hex4(json_cur *c, unsigned *out) {
    unsigned v = 0;
    for (int i = 0; i < 4; i++) {
        if (c->p >= c->end) {
            json_fail(c, "truncated unicode escape");
            return -1;
        }
        int h = hex_val(*c->p++);
        if (h < 0) {
            json_fail(c, "bad unicode escape");
            return -1;
        }
        v = (v << 4) | (unsigned)h;
    }
    *out = v;
    return 0;
}

static char *parse_string(json_cur *c) {
    skip_ws(c);
    if (c->p >= c->end || *c->p != '"') {
        json_fail(c, "expected string");
        return NULL;
    }
    c->p++;

    char *buf = NULL;
    size_t len = 0;
    size_t cap = 0;
    while (c->p < c->end) {
        uint8_t ch = *c->p++;
        if (ch == '"') {
            if (append_byte(&buf, &len, &cap, '\0')) {
                free(buf);
                json_fail(c, "oom");
                return NULL;
            }
            return buf;
        }
        if (ch < 0x20) {
            free(buf);
            json_fail(c, "control character in string");
            return NULL;
        }
        if (ch == '\\') {
            if (c->p >= c->end) {
                free(buf);
                json_fail(c, "truncated escape");
                return NULL;
            }
            ch = *c->p++;
            switch (ch) {
                case '"':  ch = '"';  break;
                case '\\': ch = '\\'; break;
                case '/':  ch = '/';  break;
                case 'b':  ch = '\b'; break;
                case 'f':  ch = '\f'; break;
                case 'n':  ch = '\n'; break;
                case 'r':  ch = '\r'; break;
                case 't':  ch = '\t'; break;
                case 'u': {
                    unsigned u = 0;
                    if (parse_hex4(c, &u)) { free(buf); return NULL; }
                    if (u > 0x7f) {
                        free(buf);
                        json_fail(c, "non-ascii string escape");
                        return NULL;
                    }
                    ch = (uint8_t)u;
                    break;
                }
                default:
                    free(buf);
                    json_fail(c, "bad string escape");
                    return NULL;
            }
        }
        if (append_byte(&buf, &len, &cap, (char)ch)) {
            free(buf);
            json_fail(c, "oom");
            return NULL;
        }
    }

    free(buf);
    json_fail(c, "unterminated string");
    return NULL;
}

static int skip_string(json_cur *c) {
    skip_ws(c);
    if (c->p >= c->end || *c->p != '"') {
        json_fail(c, "expected string");
        return -1;
    }
    c->p++;
    while (c->p < c->end) {
        uint8_t ch = *c->p++;
        if (ch == '"') return 0;
        if (ch < 0x20) {
            json_fail(c, "control character in string");
            return -1;
        }
        if (ch == '\\') {
            if (c->p >= c->end) {
                json_fail(c, "truncated escape");
                return -1;
            }
            ch = *c->p++;
            if (ch == 'u') {
                unsigned ignored = 0;
                if (parse_hex4(c, &ignored)) return -1;
            } else if (ch != '"' && ch != '\\' && ch != '/' &&
                       ch != 'b' && ch != 'f' && ch != 'n' &&
                       ch != 'r' && ch != 't') {
                json_fail(c, "bad string escape");
                return -1;
            }
        }
    }
    json_fail(c, "unterminated string");
    return -1;
}

static int skip_number(json_cur *c) {
    skip_ws(c);
    if (c->p >= c->end) {
        json_fail(c, "expected number");
        return -1;
    }
    if (*c->p == '-') c->p++;
    if (c->p >= c->end || *c->p < '0' || *c->p > '9') {
        json_fail(c, "expected number");
        return -1;
    }
    if (*c->p == '0') {
        c->p++;
    } else {
        while (c->p < c->end && *c->p >= '0' && *c->p <= '9') c->p++;
    }
    if (c->p < c->end && *c->p == '.') {
        c->p++;
        if (c->p >= c->end || *c->p < '0' || *c->p > '9') {
            json_fail(c, "bad number");
            return -1;
        }
        while (c->p < c->end && *c->p >= '0' && *c->p <= '9') c->p++;
    }
    if (c->p < c->end && (*c->p == 'e' || *c->p == 'E')) {
        c->p++;
        if (c->p < c->end && (*c->p == '+' || *c->p == '-')) c->p++;
        if (c->p >= c->end || *c->p < '0' || *c->p > '9') {
            json_fail(c, "bad number");
            return -1;
        }
        while (c->p < c->end && *c->p >= '0' && *c->p <= '9') c->p++;
    }
    return 0;
}

static int match_lit(json_cur *c, const char *lit) {
    size_t n = strlen(lit);
    skip_ws(c);
    if ((size_t)(c->end - c->p) < n || memcmp(c->p, lit, n) != 0) {
        json_fail(c, "unexpected literal");
        return -1;
    }
    c->p += n;
    return 0;
}

static int skip_value(json_cur *c);

static int skip_array(json_cur *c) {
    if (consume(c, '[')) return -1;
    skip_ws(c);
    if (c->p < c->end && *c->p == ']') { c->p++; return 0; }
    for (;;) {
        if (skip_value(c)) return -1;
        skip_ws(c);
        if (c->p < c->end && *c->p == ',') { c->p++; continue; }
        if (c->p < c->end && *c->p == ']') { c->p++; return 0; }
        json_fail(c, "expected array delimiter");
        return -1;
    }
}

static int skip_object(json_cur *c) {
    if (consume(c, '{')) return -1;
    skip_ws(c);
    if (c->p < c->end && *c->p == '}') { c->p++; return 0; }
    for (;;) {
        if (skip_string(c)) return -1;
        if (consume(c, ':')) return -1;
        if (skip_value(c)) return -1;
        skip_ws(c);
        if (c->p < c->end && *c->p == ',') { c->p++; continue; }
        if (c->p < c->end && *c->p == '}') { c->p++; return 0; }
        json_fail(c, "expected object delimiter");
        return -1;
    }
}

static int skip_value(json_cur *c) {
    skip_ws(c);
    if (c->p >= c->end) {
        json_fail(c, "expected value");
        return -1;
    }
    switch (*c->p) {
        case '"': return skip_string(c);
        case '{': return skip_object(c);
        case '[': return skip_array(c);
        case 't': return match_lit(c, "true");
        case 'f': return match_lit(c, "false");
        case 'n': return match_lit(c, "null");
        default:
            if (*c->p == '-' || (*c->p >= '0' && *c->p <= '9'))
                return skip_number(c);
            json_fail(c, "expected value");
            return -1;
    }
}

static int parse_u64(json_cur *c, uint64_t *out) {
    uint64_t v = 0;
    int saw_digit = 0;
    skip_ws(c);
    while (c->p < c->end && *c->p >= '0' && *c->p <= '9') {
        unsigned digit = (unsigned)(*c->p - '0');
        if (v > (UINT64_MAX - digit) / 10u) {
            json_fail(c, "integer overflow");
            return -1;
        }
        v = v * 10u + digit;
        c->p++;
        saw_digit = 1;
    }
    if (!saw_digit) {
        json_fail(c, "expected unsigned integer");
        return -1;
    }
    *out = v;
    return 0;
}

static int parse_shape(json_cur *c, uint32_t *n_dims, uint64_t dims[4]) {
    uint32_t n = 0;
    if (consume(c, '[')) return -1;
    skip_ws(c);
    if (c->p < c->end && *c->p == ']') {
        c->p++;
        *n_dims = 0;
        return 0;
    }
    for (;;) {
        if (n >= 8) {
            json_fail(c, "shape rank exceeds 8");
            return -1;
        }
        if (parse_u64(c, &dims[n])) return -1;
        n++;
        skip_ws(c);
        if (c->p < c->end && *c->p == ',') { c->p++; continue; }
        if (c->p < c->end && *c->p == ']') {
            c->p++;
            *n_dims = n;
            return 0;
        }
        json_fail(c, "expected shape delimiter");
        return -1;
    }
}

static int parse_offsets(json_cur *c, uint64_t offsets[2]) {
    if (consume(c, '[')) return -1;
    if (parse_u64(c, &offsets[0])) return -1;
    if (consume(c, ',')) return -1;
    if (parse_u64(c, &offsets[1])) return -1;
    if (consume(c, ']')) return -1;
    return 0;
}

static qw36_safetensors_dtype dtype_from_name(const char *s) {
    if (!strcmp(s, "BOOL")) return QW36_ST_DTYPE_BOOL;
    if (!strcmp(s, "U8")) return QW36_ST_DTYPE_U8;
    if (!strcmp(s, "U16")) return QW36_ST_DTYPE_U16;
    if (!strcmp(s, "U32")) return QW36_ST_DTYPE_U32;
    if (!strcmp(s, "U64")) return QW36_ST_DTYPE_U64;
    if (!strcmp(s, "I8")) return QW36_ST_DTYPE_I8;
    if (!strcmp(s, "I16")) return QW36_ST_DTYPE_I16;
    if (!strcmp(s, "I32")) return QW36_ST_DTYPE_I32;
    if (!strcmp(s, "I64")) return QW36_ST_DTYPE_I64;
    if (!strcmp(s, "F8_E4M3")) return QW36_ST_DTYPE_F8_E4M3;
    if (!strcmp(s, "F8_E5M2")) return QW36_ST_DTYPE_F8_E5M2;
    if (!strcmp(s, "F16")) return QW36_ST_DTYPE_F16;
    if (!strcmp(s, "BF16")) return QW36_ST_DTYPE_BF16;
    if (!strcmp(s, "F32")) return QW36_ST_DTYPE_F32;
    if (!strcmp(s, "F64")) return QW36_ST_DTYPE_F64;
    return QW36_ST_DTYPE_UNKNOWN;
}

const char *qw36_safetensors_dtype_name(qw36_safetensors_dtype dtype) {
    switch (dtype) {
        case QW36_ST_DTYPE_BOOL: return "BOOL";
        case QW36_ST_DTYPE_U8: return "U8";
        case QW36_ST_DTYPE_U16: return "U16";
        case QW36_ST_DTYPE_U32: return "U32";
        case QW36_ST_DTYPE_U64: return "U64";
        case QW36_ST_DTYPE_I8: return "I8";
        case QW36_ST_DTYPE_I16: return "I16";
        case QW36_ST_DTYPE_I32: return "I32";
        case QW36_ST_DTYPE_I64: return "I64";
        case QW36_ST_DTYPE_F8_E4M3: return "F8_E4M3";
        case QW36_ST_DTYPE_F8_E5M2: return "F8_E5M2";
        case QW36_ST_DTYPE_F16: return "F16";
        case QW36_ST_DTYPE_BF16: return "BF16";
        case QW36_ST_DTYPE_F32: return "F32";
        case QW36_ST_DTYPE_F64: return "F64";
        default: return "UNKNOWN";
    }
}

static void tensor_info_free(st_tensor_info *t) {
    if (!t) return;
    free(t->name);
    free(t->dtype_name);
    memset(t, 0, sizeof(*t));
}

static void file_destroy(qw36_safetensors_file *f) {
    if (!f) return;
    if (f->tensors) {
        for (size_t i = 0; i < f->n_tensors; i++)
            tensor_info_free(&f->tensors[i]);
        free(f->tensors);
    }
    if (f->map && f->map != MAP_FAILED) munmap((void *)f->map, f->map_len);
    if (f->fd >= 0) close(f->fd);
    free(f);
}

static int append_tensor(qw36_safetensors_file *f, st_tensor_info *t) {
    if (f->n_tensors == f->cap_tensors) {
        size_t ncap = f->cap_tensors ? f->cap_tensors * 2 : 16;
        st_tensor_info *nt = (st_tensor_info *)realloc(f->tensors,
            ncap * sizeof(*nt));
        if (!nt) return -1;
        f->tensors = nt;
        f->cap_tensors = ncap;
    }
    f->tensors[f->n_tensors++] = *t;
    memset(t, 0, sizeof(*t));
    return 0;
}

static int parse_tensor_object(json_cur *c, qw36_safetensors_file *f,
                               char *name) {
    st_tensor_info t;
    int have_dtype = 0;
    int have_shape = 0;
    int have_offsets = 0;

    memset(&t, 0, sizeof(t));
    t.name = name;

    if (consume(c, '{')) goto fail;
    skip_ws(c);
    if (c->p < c->end && *c->p == '}') {
        json_fail(c, "empty tensor object");
        goto fail;
    }

    for (;;) {
        char *field = parse_string(c);
        if (!field) goto fail;
        if (consume(c, ':')) { free(field); goto fail; }

        if (!strcmp(field, "dtype")) {
            free(t.dtype_name);
            t.dtype_name = parse_string(c);
            if (!t.dtype_name) { free(field); goto fail; }
            t.dtype = dtype_from_name(t.dtype_name);
            have_dtype = 1;
        } else if (!strcmp(field, "shape")) {
            if (parse_shape(c, &t.n_dims, t.dims)) { free(field); goto fail; }
            have_shape = 1;
        } else if (!strcmp(field, "data_offsets")) {
            if (parse_offsets(c, t.data_offsets)) { free(field); goto fail; }
            have_offsets = 1;
        } else {
            if (skip_value(c)) { free(field); goto fail; }
        }
        free(field);

        skip_ws(c);
        if (c->p < c->end && *c->p == ',') { c->p++; continue; }
        if (c->p < c->end && *c->p == '}') { c->p++; break; }
        json_fail(c, "expected tensor object delimiter");
        goto fail;
    }

    if (!have_dtype || !have_shape || !have_offsets) {
        json_fail(c, "missing tensor field");
        goto fail;
    }
    if (t.data_offsets[1] < t.data_offsets[0]) {
        json_fail(c, "invalid data_offsets");
        goto fail;
    }

    {
        uint64_t data_len = (uint64_t)(f->map_len -
            (size_t)(f->data_start - f->map));
        if (t.data_offsets[1] > data_len) {
            json_fail(c, "tensor data out of bounds");
            goto fail;
        }
        t.nbytes = (size_t)(t.data_offsets[1] - t.data_offsets[0]);
        t.data = (const void *)(f->data_start + t.data_offsets[0]);
    }

    if (append_tensor(f, &t)) {
        json_fail(c, "oom");
        goto fail;
    }
    return 0;

fail:
    tensor_info_free(&t);
    return -1;
}

static int parse_header(qw36_safetensors_file *f, char *err, size_t err_cap) {
    json_cur c;
    memset(&c, 0, sizeof(c));
    c.p = f->header;
    c.start = f->header;
    c.end = f->header + f->header_len;

    if (consume(&c, '{')) goto fail;
    skip_ws(&c);
    if (c.p < c.end && *c.p == '}') { c.p++; goto done; }

    for (;;) {
        char *key = parse_string(&c);
        if (!key) goto fail;
        if (consume(&c, ':')) { free(key); goto fail; }

        if (!strcmp(key, "__metadata__")) {
            free(key);
            if (skip_value(&c)) goto fail;
        } else {
            if (parse_tensor_object(&c, f, key)) goto fail;
        }

        skip_ws(&c);
        if (c.p < c.end && *c.p == ',') { c.p++; continue; }
        if (c.p < c.end && *c.p == '}') { c.p++; break; }
        json_fail(&c, "expected top-level delimiter");
        goto fail;
    }

done:
    skip_ws(&c);
    if (!c.error && c.p != c.end) {
        json_fail(&c, "trailing non-whitespace in header");
    }
    if (!c.error) return 0;

fail:
    if (err && err_cap) {
        size_t off = c.p >= c.start ? (size_t)(c.p - c.start) : 0;
        snprintf(err, err_cap, "invalid safetensors header at byte %zu: %s",
                 off, c.why ? c.why : "parse error");
    }
    return -1;
}

static uint64_t read_u64le(const uint8_t p[8]) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) {
        v = (v << 8) | (uint64_t)p[i];
    }
    return v;
}

qw36_safetensors_file *qw36_safetensors_open(const char *path,
                                             char *err, size_t err_cap) {
    qw36_safetensors_file *f =
        (qw36_safetensors_file *)calloc(1, sizeof(*f));
    if (!f) {
        if (err && err_cap) snprintf(err, err_cap, "oom");
        return NULL;
    }
    f->fd = -1;

    f->fd = open(path, O_RDONLY);
    if (f->fd < 0) {
        int e = errno;
        if (err && err_cap) snprintf(err, err_cap, "open(%s): %s",
                                     path, strerror(e));
        free(f);
        return NULL;
    }

    {
        struct stat st;
        if (fstat(f->fd, &st) < 0) {
            if (err && err_cap)
                snprintf(err, err_cap, "fstat: %s", strerror(errno));
            file_destroy(f);
            return NULL;
        }
        if (st.st_size < 8) {
            if (err && err_cap) snprintf(err, err_cap, "file too small");
            file_destroy(f);
            return NULL;
        }
        f->map_len = (size_t)st.st_size;
    }

    f->map = (const uint8_t *)mmap(NULL, f->map_len, PROT_READ, MAP_PRIVATE,
                                   f->fd, 0);
    if (f->map == MAP_FAILED) {
        if (err && err_cap) snprintf(err, err_cap, "mmap: %s", strerror(errno));
        file_destroy(f);
        return NULL;
    }

    {
        uint64_t header_len64 = read_u64le(f->map);
        if (header_len64 > (uint64_t)(f->map_len - 8u)) {
            if (err && err_cap)
                snprintf(err, err_cap, "header length past EOF");
            file_destroy(f);
            return NULL;
        }
        if (header_len64 > (uint64_t)SIZE_MAX) {
            if (err && err_cap) snprintf(err, err_cap, "header too large");
            file_destroy(f);
            return NULL;
        }
        f->header_len = (size_t)header_len64;
        f->header = f->map + 8u;
        f->data_start = f->header + f->header_len;
    }

    if (parse_header(f, err, err_cap)) {
        file_destroy(f);
        return NULL;
    }

    return f;
}

void qw36_safetensors_close(qw36_safetensors_file *f) {
    file_destroy(f);
}

size_t qw36_safetensors_tensor_count(const qw36_safetensors_file *f) {
    return f ? f->n_tensors : 0;
}

static void tensor_to_public(const st_tensor_info *t,
                             qw36_safetensors_tensor *out) {
    out->name = t->name;
    out->dtype_name = t->dtype_name;
    out->dtype = t->dtype;
    out->n_dims = t->n_dims;
    memcpy(out->dims, t->dims, sizeof(out->dims));
    out->data_offsets[0] = t->data_offsets[0];
    out->data_offsets[1] = t->data_offsets[1];
    out->data = t->data;
    out->nbytes = t->nbytes;
}

int qw36_safetensors_get_tensor_by_index(const qw36_safetensors_file *f,
                                         size_t i,
                                         qw36_safetensors_tensor *out) {
    if (!f || !out || i >= f->n_tensors) return -1;
    tensor_to_public(&f->tensors[i], out);
    return 0;
}

int qw36_safetensors_get_tensor(const qw36_safetensors_file *f,
                                const char *name,
                                qw36_safetensors_tensor *out) {
    if (!f || !name || !out) return -1;
    for (size_t i = 0; i < f->n_tensors; i++) {
        if (f->tensors[i].name && !strcmp(f->tensors[i].name, name)) {
            tensor_to_public(&f->tensors[i], out);
            return 0;
        }
    }
    return -1;
}

size_t qw36_safetensors_header_len(const qw36_safetensors_file *f) {
    return f ? f->header_len : 0;
}

const void *qw36_safetensors_data_base(const qw36_safetensors_file *f) {
    return f ? (const void *)f->data_start : NULL;
}
