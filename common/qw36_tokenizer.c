/* qw36_tokenizer.c — BBPE tokenizer (stub).
 *
 * Owner: Claude. Reference: tiktoken / HF tokenizers. We avoid the regex
 * dependency by hand-rolling the GPT-style pre-tokenization splitter.
 */

#include "qw36_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct qw36_tokenizer {
    /* TODO(Claude):
     *   - vocab: id -> (ptr, len)
     *   - merges: pair -> rank
     *   - special_tokens: name -> id
     */
    uint32_t bos, eos, im_start, im_end;
};

qw36_tokenizer *qw36_tokenizer_new(const qw36_gguf_file *f,
                                   char *err, size_t err_cap)
{
    (void)f;
    if (err && err_cap) snprintf(err, err_cap, "qw36_tokenizer_new: TODO");
    return NULL;
}

void qw36_tokenizer_free(qw36_tokenizer *t) { free(t); }

int qw36_tokenizer_encode(const qw36_tokenizer *t,
                          const char *text,
                          uint32_t *out_ids, size_t *out_n)
{
    (void)t; (void)text; (void)out_ids;
    if (out_n) *out_n = 0;
    return -1; /* TODO(Claude) */
}

const char *qw36_tokenizer_decode_one(const qw36_tokenizer *t,
                                      uint32_t id, size_t *len_out)
{
    (void)t; (void)id;
    if (len_out) *len_out = 0;
    return NULL; /* TODO(Claude) */
}

uint32_t qw36_tokenizer_bos(const qw36_tokenizer *t)      { return t ? t->bos : 0; }
uint32_t qw36_tokenizer_eos(const qw36_tokenizer *t)      { return t ? t->eos : 0; }
uint32_t qw36_tokenizer_im_start(const qw36_tokenizer *t) { return t ? t->im_start : 0; }
uint32_t qw36_tokenizer_im_end(const qw36_tokenizer *t)   { return t ? t->im_end : 0; }
