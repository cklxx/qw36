/* qw36_internal.h - private interfaces shared by common implementation files.
 *
 * This header is not part of the public API. It exposes the engine internals,
 * lazy weight descriptor, forward-context bookkeeping, and scalar/GPU helper
 * routines used by the split forward implementation. All names use the
 * qw36__ prefix to mark package-private linkage by convention.
 */

#ifndef QW36_INTERNAL_H
#define QW36_INTERNAL_H

#include "qw36.h"
#include "qw36_gpu.h"
#include "qw36_gguf.h"

#include <stddef.h>
#include <stdint.h>

typedef struct qw36_lazy_w {
    const void   *data;
    qw36_dtype    dtype;
    uint32_t      ggml_type;
    uint64_t      rows;
    uint64_t      cols;
    uint64_t      n_extra;
    qw36_gpu_buf *gpu_buf;
} qw36_lazy_w;

typedef struct qw36_gpu_cache_entry {
    const void   *host;
    size_t        bytes;
    qw36_dtype    dtype;
    qw36_gpu_buf *gpu_buf;
} qw36_gpu_cache_entry;

struct qw36_engine {
    qw36_config       cfg;
    qw36_weights      weights;
    qw36_gpu_backend *backend;
    qw36_gpu_ctx     *ctx;
    qw36_gguf_file   *gguf;
    void             **owned;
    size_t             owned_n;
    size_t             owned_cap;
    qw36_gpu_cache_entry *gpu_cache;
    size_t                gpu_cache_n;
    size_t                gpu_cache_cap;
    char               arch[32];
};

typedef struct qw36_forward_ctx {
    qw36_engine *eng;
    qw36_state  *st;
    const qw36_config *cfg;
    const qw36_weights *weights;
    size_t hidden;
    size_t inter;
    float *row_scratch;
    int gpu_state;
    int *x_dev_valid;
    int *x_host_valid;
    int debug_layer;
    /* When non-zero, the attn sub-block already wrote x_rms_dev =
     * rmsnorm(x_dev + attn_out) via the fused residual+rmsnorm kernel,
     * and qw36__mlp_forward must skip its own post_attn_layernorm
     * dispatch. Reset to 0 at the start of every layer in qw36_forward. */
    int post_attn_rmsnorm_done;
    /* Same idea between layers: qw36__mlp_forward may fold its closing
     * residual_add with the next layer's input_layernorm. When set,
     * qw36__attn_vanilla / qw36__attn_deltanet must skip their own
     * input_layernorm dispatch. Reset to 0 per layer in qw36_forward. */
    int input_rmsnorm_done;
} qw36_forward_ctx;

extern qw36_engine *qw36__active_engine;
extern int qw36__skip_logits_this_forward;

void *qw36__eng_own(qw36_engine *eng, void *p);
size_t qw36__dtype_nbytes(qw36_dtype dtype);
size_t qw36__tensor_bytes(qw36_dtype dtype, size_t numel);
int qw36__dtype_is_native_gpu_quant(qw36_dtype dt);
int qw36__dtype_block_geom(qw36_dtype dt, size_t *qk,
                           size_t *bytes_per_block);
int qw36__dequant_row(const qw36_lazy_w *w, size_t row_idx, float *out);
float *qw36__materialize_f32(const void *data, qw36_dtype dt, size_t n);
uint16_t qw36__f32_to_f16(float f);
float    qw36__f16_to_f32(uint16_t h);
int qw36__active_backend(qw36_gpu_backend **be_out, qw36_gpu_ctx **ctx_out);

qw36_gpu_buf *qw36__gpu_cached_upload(qw36_engine *eng, const void *host,
                                      size_t bytes, qw36_dtype dtype);
void qw36__gpu_cache_free(qw36_engine *eng);
int qw36__state_backend(qw36_state *st, qw36_gpu_backend **be_out,
                        qw36_gpu_ctx **ctx_out);
int qw36__state_copy_from_host(qw36_state *st, void *dst_dev,
                               const void *src, size_t bytes);
int qw36__state_download_to_host(qw36_state *st, void *src_dev,
                                 void *dst, size_t bytes);

int qw36__ensure_x_host(qw36_forward_ctx *fc);
int qw36__ensure_x_dev(qw36_forward_ctx *fc);

int qw36__matmul_lazy(float *y, const float *x, const qw36_lazy_w *w,
                      float *row_scratch);
int qw36__matmul_lazy_dev(qw36_gpu_buf *y, qw36_gpu_buf *x,
                          const qw36_lazy_w *w);
int qw36__matmul_lazy_slice(float *y, const float *x,
                            const qw36_lazy_w *w, size_t slice_idx,
                            float *row_scratch);
int qw36__embed_lookup_lazy(const qw36_lazy_w *w, uint32_t token, float *out);
int qw36__embed_lookup_lazy_dev(qw36_gpu_buf *out, const qw36_lazy_w *w,
                                uint32_t token);

float qw36__silu(float x);
void qw36__rmsnorm_dispatch(float *out, const float *x, const float *w,
                            size_t n, float eps);
int qw36__rmsnorm_dispatch_dev(qw36_gpu_buf *out, qw36_gpu_buf *x,
                               const float *w, size_t n, float eps);
int qw36__residual_rmsnorm_dispatch_dev(qw36_gpu_buf *x, qw36_gpu_buf *y,
                                         qw36_gpu_buf *out, const float *w,
                                         size_t n, float eps);
void qw36__residual_add_dispatch(float *x, const float *y, size_t n);
int qw36__residual_add_dispatch_dev(qw36_gpu_buf *x, qw36_gpu_buf *y,
                                    size_t n);
int qw36__swiglu_dispatch(float *y, const float *x,
                          const qw36_lazy_w *w_gate,
                          const qw36_lazy_w *w_up,
                          const qw36_lazy_w *w_down,
                          uint32_t hidden, uint32_t inter,
                          float *scratch_gate, float *scratch_up,
                          float *row_scratch);
int qw36__swiglu_dispatch_dev(qw36_gpu_buf *y, qw36_gpu_buf *x,
                              const qw36_lazy_w *w_gate,
                              const qw36_lazy_w *w_up,
                              const qw36_lazy_w *w_down,
                              uint32_t hidden, uint32_t inter);
int qw36__moe_forward_f32(float *y, const float *x,
                          const qw36_lazy_w *router,
                          const qw36_lazy_w *gate_exps,
                          const qw36_lazy_w *up_exps,
                          const qw36_lazy_w *down_exps,
                          const qw36_lazy_w *shared_gate,
                          const qw36_lazy_w *shared_up,
                          const qw36_lazy_w *shared_down,
                          uint32_t hidden, uint32_t moe_inter,
                          uint32_t n_experts, int top_k,
                          uint8_t norm_topk,
                          float *scratch, float *row_scratch);
void qw36__rope_head(float *x, size_t pos, size_t rot_dim, float theta_base,
                     const uint32_t *sections, uint32_t n_sections);

int qw36__attention_dispatch(float *y, const float *x,
                             const qw36_layer_weights *L,
                             float *k_cache, float *v_cache,
                             const qw36_config *c,
                             uint32_t seq_pos, uint32_t seq_capacity);
int qw36__attention_dispatch_dev(qw36_gpu_buf *y, qw36_gpu_buf *x,
                                 const qw36_layer_weights *L,
                                 qw36_gpu_buf *k_cache,
                                 qw36_gpu_buf *v_cache,
                                 const qw36_config *c,
                                 uint32_t seq_pos,
                                 uint32_t seq_capacity);
int qw36__deltanet_dispatch_dev(qw36_state *st,
                                const qw36_layer_weights *L,
                                const qw36_config *c,
                                uint32_t layer_idx,
                                int skip_input_rmsnorm);
void qw36__conv1d_silu_decode(const float *x, const float *conv_w,
                              float *conv_state, float *y,
                              uint32_t channels, uint32_t k);
void qw36__gated_delta_decode(const float *qkv,
                              const float *b_proj_raw,
                              const float *a_proj_raw,
                              const float *dt_bias,
                              const float *a_log,
                              float *state, float *out,
                              uint32_t n_key, uint32_t n_value,
                              uint32_t key_dim, uint32_t val_dim,
                              float *qbuf, float *kbuf);

int qw36__attn_vanilla_forward(qw36_forward_ctx *fc,
                               const qw36_layer_weights *L,
                               uint32_t layer_idx);
int qw36__attn_deltanet_forward(qw36_forward_ctx *fc,
                                const qw36_layer_weights *L,
                                uint32_t layer_idx);
int qw36__mlp_forward(qw36_forward_ctx *fc,
                      const qw36_layer_weights *L,
                      uint32_t layer_idx);

#endif
