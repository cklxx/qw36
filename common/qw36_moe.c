/* qw36_moe.c - Mixture-of-Experts reference implementation boundary.
 *
 * MoE routing is invoked by qw36_mlp.c after post-attention RMSNorm. The
 * exported qw36__moe_forward_f32 implementation stays CPU-reference
 * simple: router matmul, top-k insertion sort, softmax over selected experts,
 * weighted expert accumulation, and optional shared expert contribution.
 *
 * Future GPU slice/view contracts and router diagnostics should land here
 * instead of growing qw36_forward again.
 */

#include "qw36_internal.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

/* Top-k selection (small k, unsorted by index). On return out_idx/out_val
 * hold the k largest values of `logits`, sorted descending by value. */
static void top_k_select(const float *logits, uint32_t n, int k,
                         uint32_t *out_idx, float *out_val)
{
    /* Initialise the heap with -inf. */
    for (int i = 0; i < k; i++) { out_idx[i] = 0; out_val[i] = -INFINITY; }
    for (uint32_t i = 0; i < n; i++) {
        if (logits[i] <= out_val[k - 1]) continue;
        /* Insertion sort: find position, shift. */
        int j = k - 1;
        while (j > 0 && logits[i] > out_val[j - 1]) {
            out_val[j] = out_val[j - 1];
            out_idx[j] = out_idx[j - 1];
            j--;
        }
        out_val[j] = logits[i];
        out_idx[j] = i;
    }
}

/* Stable softmax over the first k entries of vals (in-place). */
static void softmax_k(float *vals, int k) {
    float mx = vals[0];
    for (int i = 1; i < k; i++) if (vals[i] > mx) mx = vals[i];
    double sum = 0.0;
    for (int i = 0; i < k; i++) { vals[i] = expf(vals[i] - mx); sum += vals[i]; }
    float inv = (float)(1.0 / sum);
    for (int i = 0; i < k; i++) vals[i] *= inv;
}

/* MoE forward (per token):
 *
 *   r = router @ x                       [n_experts]
 *   pick top_k experts by r
 *   probs = softmax(r[top_k]); (optionally renormalized)
 *   y = Σ_{e ∈ top_k} probs[e] * expert_e(x)
 *     + shared_expert(x)                 (if shared weights present)
 *
 *   expert_e(x) = down_exps[e] @ (qw36__silu(gate_exps[e] @ x) * (up_exps[e] @ x))
 *
 * scratch needs: n_experts + 2*k + 2*max(moe_inter, shared_inter)
 *              + 2*hidden + 1 floats. */
int qw36__moe_forward_f32(float *y, const float *x,
                           const qw36_lazy_w *router,
                           const qw36_lazy_w *gate_exps,
                           const qw36_lazy_w *up_exps,
                           const qw36_lazy_w *down_exps,
                           const qw36_lazy_w *shared_gate,
                           const qw36_lazy_w *shared_up,
                           const qw36_lazy_w *shared_down,
                           const qw36_lazy_w *shared_gate_inp,
                           uint32_t hidden, uint32_t moe_inter,
                           uint32_t shared_inter,
                           uint32_t n_experts, int top_k,
                           uint8_t norm_topk,
                           float *scratch, float *row_scratch)
{
    float    *r_logits = scratch;                          /* n_experts */
    float    *tk_vals  = r_logits + n_experts;             /* top_k    */
    uint32_t *tk_idx   = (uint32_t *)(tk_vals + top_k);    /* top_k    */
    const uint32_t act_inter = moe_inter > shared_inter ? moe_inter : shared_inter;
    float    *tmp_gate = (float *)(tk_idx + top_k);        /* act_inter*/
    float    *tmp_up   = tmp_gate + act_inter;             /* act_inter*/
    float    *tmp_y    = tmp_up   + act_inter;             /* hidden   */
    float    *shared_gate_scalar = tmp_y + hidden;         /* 1        */
    float    *tmp_x    = shared_gate_scalar + 1;           /* hidden   */
    const float *xin = x;

    if (y == x) {
        memcpy(tmp_x, x, (size_t)hidden * sizeof(float));
        xin = tmp_x;
    }

    /* 1. router */
    if (qw36__matmul_lazy(r_logits, xin, router, row_scratch)) return -1;

    /* 2. top-k */
    top_k_select(r_logits, n_experts, top_k, tk_idx, tk_vals);

    /* 3. softmax (and optional renormalize — softmax_k already sums to 1) */
    softmax_k(tk_vals, top_k);
    (void)norm_topk; /* always renormalized — Qwen3 sets norm_topk_prob=true */

    /* 4. accumulate top-k experts */
    memset(y, 0, hidden * sizeof(float));
    for (int t = 0; t < top_k; t++) {
        const uint32_t e = tk_idx[t];
        const float    p = tk_vals[t];
        if (qw36__matmul_lazy_slice(tmp_gate, xin, gate_exps, e, row_scratch)) return -1;
        if (qw36__matmul_lazy_slice(tmp_up,   xin, up_exps,   e, row_scratch)) return -1;
        for (uint32_t i = 0; i < moe_inter; i++)
            tmp_gate[i] = qw36__silu(tmp_gate[i]) * tmp_up[i];
        if (qw36__matmul_lazy_slice(tmp_y, tmp_gate, down_exps, e, row_scratch)) return -1;
        for (uint32_t i = 0; i < hidden; i++) y[i] += p * tmp_y[i];
    }

    /* 5. shared expert (Qwen3-MoE: always-on extra path) */
    if (shared_gate && shared_up && shared_down) {
        if (qw36__matmul_lazy(tmp_gate, xin, shared_gate, row_scratch)) return -1;
        if (qw36__matmul_lazy(tmp_up,   xin, shared_up,   row_scratch)) return -1;
        for (uint32_t i = 0; i < shared_inter; i++)
            tmp_gate[i] = qw36__silu(tmp_gate[i]) * tmp_up[i];
        if (qw36__matmul_lazy(tmp_y, tmp_gate, shared_down, row_scratch)) return -1;
        float scale = 1.0f;
        if (shared_gate_inp) {
            if (qw36__matmul_lazy(shared_gate_scalar, xin, shared_gate_inp,
                                  row_scratch)) return -1;
            scale = 1.0f / (1.0f + expf(-shared_gate_scalar[0]));
        }
        for (uint32_t i = 0; i < hidden; i++) y[i] += scale * tmp_y[i];
    }
    return 0;
}
