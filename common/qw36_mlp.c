/* qw36_mlp.c - post-attention feed-forward dispatch.
 *
 * This module runs the second residual block of each layer after vanilla
 * attention or DeltaNet has updated the residual stream. It supports dense
 * SwiGLU layers and Qwen-MoE router/top-k/shared-expert layers.
 *
 * GPU-resident paths are attempted first when backend kernels and uploaded
 * weights are available. CPU fallback keeps the reference path authoritative
 * and updates the forward-context host/device validity bits for the caller.
 */

#include "qw36_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int has_dense_swiglu_weights(const qw36_layer_weights *L, uint32_t inter)
{
    if (!L || L->moe_router || !L->gate_proj || !L->down_proj) return 0;
    if (L->up_proj) return 1;
    const qw36_lazy_w *gate = (const qw36_lazy_w *)L->gate_proj;
    return gate && gate->rows == (uint64_t)inter * 2;
}

int qw36__mlp_forward(qw36_forward_ctx *fc,
                      const qw36_layer_weights *L,
                      uint32_t layer_idx)
{
    (void)layer_idx;
    if (!fc || !L || !fc->eng || !fc->st || !fc->cfg) return -1;

    qw36_engine *eng = fc->eng;
    qw36_state *st = fc->st;
    const qw36_config *c = fc->cfg;
    const size_t hidden = fc->hidden;
    const size_t inter = fc->inter;
    float *x = st->x;

    int mlp_gpu_done = 0;
    if (fc->gpu_state && has_dense_swiglu_weights(L, (uint32_t)inter) &&
        eng->backend && eng->backend->rmsnorm &&
        eng->backend->swiglu_mlp && eng->backend->residual_add) {
        int erc = qw36__ensure_x_dev(fc);
        if (erc) return erc;
        int grc = 0;
        grc |= qw36__rmsnorm_dispatch_dev((qw36_gpu_buf *)st->x_rms_dev,
                                          (qw36_gpu_buf *)st->x_dev,
                                          (const float *)L->post_attn_layernorm,
                                          hidden, c->rms_norm_eps);
        grc |= qw36__swiglu_dispatch_dev((qw36_gpu_buf *)st->x_rms_dev,
                                         (qw36_gpu_buf *)st->x_rms_dev,
                                         (const qw36_lazy_w *)L->gate_proj,
                                         (const qw36_lazy_w *)L->up_proj,
                                         (const qw36_lazy_w *)L->down_proj,
                                         (uint32_t)hidden, (uint32_t)inter);
        grc |= qw36__residual_add_dispatch_dev((qw36_gpu_buf *)st->x_dev,
                                               (qw36_gpu_buf *)st->x_rms_dev,
                                               hidden);
        if (grc == 0) {
            *fc->x_dev_valid = 1;
            *fc->x_host_valid = 0;
            mlp_gpu_done = 1;
        } else {
            return -9;
        }
    }

    if (fc->gpu_state && !mlp_gpu_done && L->moe_router &&
        L->moe_expert_gate && L->moe_expert_up && L->moe_expert_down &&
        eng->backend && eng->backend->rmsnorm &&
        eng->backend->moe_forward && eng->backend->residual_add) {
        const qw36_lazy_w *router = (const qw36_lazy_w *)L->moe_router;
        const qw36_lazy_w *eg = (const qw36_lazy_w *)L->moe_expert_gate;
        const qw36_lazy_w *eu = (const qw36_lazy_w *)L->moe_expert_up;
        const qw36_lazy_w *ed = (const qw36_lazy_w *)L->moe_expert_down;
        const qw36_lazy_w *sg = (const qw36_lazy_w *)L->moe_shared_gate;
        const qw36_lazy_w *su = (const qw36_lazy_w *)L->moe_shared_up;
        const qw36_lazy_w *sd = (const qw36_lazy_w *)L->moe_shared_down;
        const uint32_t mi = c->moe_intermediate_size
                          ? c->moe_intermediate_size
                          : (uint32_t)eg->rows;
        if (router->gpu_buf && eg->gpu_buf && eu->gpu_buf && ed->gpu_buf &&
            (!sg || (sg->gpu_buf && su && su->gpu_buf && sd && sd->gpu_buf))) {
            int erc = qw36__ensure_x_dev(fc);
            if (erc) return erc;
            int grc = 0;
            grc |= qw36__rmsnorm_dispatch_dev((qw36_gpu_buf *)st->x_rms_dev,
                                              (qw36_gpu_buf *)st->x_dev,
                                              (const float *)L->post_attn_layernorm,
                                              hidden, c->rms_norm_eps);
            if (grc == 0) {
                eng->backend->moe_forward(eng->ctx,
                    (qw36_gpu_buf *)st->x_rms_dev,
                    (qw36_gpu_buf *)st->x_rms_dev,
                    router->gpu_buf, eg->gpu_buf, eu->gpu_buf, ed->gpu_buf,
                    sg ? sg->gpu_buf : NULL,
                    su ? su->gpu_buf : NULL,
                    sd ? sd->gpu_buf : NULL,
                    (uint32_t)hidden, mi, c->moe_num_experts,
                    c->moe_experts_per_tok, c->moe_norm_topk_prob);
                grc |= qw36__residual_add_dispatch_dev((qw36_gpu_buf *)st->x_dev,
                                                       (qw36_gpu_buf *)st->x_rms_dev,
                                                       hidden);
            }
            if (grc == 0) {
                *fc->x_dev_valid = 1;
                *fc->x_host_valid = 0;
                mlp_gpu_done = 1;
            } else {
                return -10;
            }
        }
    }

    if (mlp_gpu_done) return 0;

    int erc = qw36__ensure_x_host(fc);
    if (erc) return erc;
    qw36__rmsnorm_dispatch(st->x_rms, x, (const float *)L->post_attn_layernorm,
                           hidden, c->rms_norm_eps);

    if (L->moe_router) {
        const uint32_t mi = c->moe_intermediate_size
                          ? c->moe_intermediate_size
                          : ((const qw36_lazy_w *)L->moe_expert_gate)->rows;
        const size_t need = (size_t)c->moe_num_experts
                          + (size_t)c->moe_experts_per_tok * 2
                          + (size_t)mi * 2
                          + (size_t)hidden;
        float *moe_scratch = (float *)calloc(need, sizeof(float));
        if (!moe_scratch) return -3;
        int mrc = qw36__moe_forward_f32(st->x_rms, st->x_rms,
            (const qw36_lazy_w *)L->moe_router,
            (const qw36_lazy_w *)L->moe_expert_gate,
            (const qw36_lazy_w *)L->moe_expert_up,
            (const qw36_lazy_w *)L->moe_expert_down,
            (const qw36_lazy_w *)L->moe_shared_gate,
            (const qw36_lazy_w *)L->moe_shared_up,
            (const qw36_lazy_w *)L->moe_shared_down,
            (uint32_t)hidden, mi, c->moe_num_experts,
            (int)c->moe_experts_per_tok, c->moe_norm_topk_prob,
            moe_scratch, fc->row_scratch);
        free(moe_scratch);
        if (mrc) return -11;
        qw36__residual_add_dispatch(x, st->x_rms, hidden);
    } else if (has_dense_swiglu_weights(L, (uint32_t)inter)) {
        if (qw36__swiglu_dispatch(st->x_rms, st->x_rms,
                                  (const qw36_lazy_w *)L->gate_proj,
                                  (const qw36_lazy_w *)L->up_proj,
                                  (const qw36_lazy_w *)L->down_proj,
                                  (uint32_t)hidden, (uint32_t)inter,
                                  st->gate, st->up,
                                  fc->row_scratch))
            return -12;
        qw36__residual_add_dispatch(x, st->x_rms, hidden);
    }

    *fc->x_host_valid = 1;
    *fc->x_dev_valid = 0;
    return 0;
}
