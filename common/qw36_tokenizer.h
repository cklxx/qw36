/* qw36_tokenizer.h — BPE tokenizer for Qwen 3.6.
 *
 * Qwen 3.x uses a BBPE (byte-level BPE) tokenizer with the cl100k-style
 * pre-tokenization regex. We load the vocab + merges from the GGUF file.
 */

#ifndef QW36_TOKENIZER_H
#define QW36_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#include "qw36_gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qw36_tokenizer qw36_tokenizer;

qw36_tokenizer *qw36_tokenizer_new(const qw36_gguf_file *f,
                                   char *err, size_t err_cap);
void            qw36_tokenizer_free(qw36_tokenizer *t);

/* Encode UTF-8 text. *out_n is in/out (capacity → actual). Returns 0 on
 * success, -1 on overflow (in which case *out_n holds required capacity). */
int qw36_tokenizer_encode(const qw36_tokenizer *t,
                          const char *text,
                          uint32_t *out_ids, size_t *out_n);

/* Decode one id to its UTF-8 bytes. Returns a pointer into the tokenizer's
 * owned storage; lifetime = tokenizer. *len_out gets the byte length. */
const char *qw36_tokenizer_decode_one(const qw36_tokenizer *t,
                                      uint32_t id, size_t *len_out);

/* Special tokens (Qwen3 chat template). */
uint32_t qw36_tokenizer_bos(const qw36_tokenizer *t);
uint32_t qw36_tokenizer_eos(const qw36_tokenizer *t);
uint32_t qw36_tokenizer_im_start(const qw36_tokenizer *t);
uint32_t qw36_tokenizer_im_end(const qw36_tokenizer *t);

#ifdef __cplusplus
}
#endif

#endif /* QW36_TOKENIZER_H */
