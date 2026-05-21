/* qw36_policy.c - backend profile and feature-flag resolution.
 *
 * This module is the single place where user-facing runtime profiles and
 * legacy environment knobs are translated into package-private booleans.
 * It deliberately does not mutate process environment or inspect model
 * tensors; callers pass the resolved qw36_backend_policy into the existing
 * materialize/repack/fuse/upload preparation stages.
 *
 * Precedence:
 *   1. CLI may set QW36_PROFILE before engine_open.
 *   2. Explicit low-level env knobs override profile defaults.
 *   3. Unknown/empty profile names fall back to the conservative reference
 *      defaults so existing scripts keep their old behavior.
 */

#include "qw36_internal.h"

#include <stdlib.h>
#include <string.h>

static int env_bool(const char *name, int fallback)
{
    const char *v = getenv(name);
    return v ? (atoi(v) != 0) : fallback;
}

static const char *profile_from_env(void)
{
    const char *p = getenv("QW36_PROFILE");
    if (p && *p) return p;
    p = getenv("QW36_BACKEND_PROFILE");
    return (p && *p) ? p : "reference";
}

static int profile_is(const char *p, const char *a, const char *b)
{
    return p && (!strcmp(p, a) || (b && !strcmp(p, b)));
}

int qw36__profile_name_is_valid(const char *profile)
{
    if (!profile || !*profile) return 1;
    return profile_is(profile, "reference", "strict") ||
           profile_is(profile, "balanced", "default") ||
           profile_is(profile, "fp16", NULL) ||
           profile_is(profile, "lowmem", "quant") ||
           profile_is(profile, "fast", "serving");
}

void qw36__backend_policy_from_env(qw36_backend_policy *p,
                                   const qw36_gpu_backend *backend)
{
    memset(p, 0, sizeof(*p));
    p->profile_name = profile_from_env();
    p->is_metal = backend && backend->name &&
                  strcmp(backend->name, "metal") == 0;
    if (!p->is_metal) return;

    const int profile_fast = profile_is(p->profile_name, "fast", "serving");
    const int profile_lowmem = profile_is(p->profile_name, "lowmem", "quant");
    const int profile_fp16 = profile_is(p->profile_name, "fp16", NULL);
    const int metal_fast = env_bool("QW36_METAL_FAST", 0);
    const int fast = profile_fast || metal_fast;

    p->quant_gpu = env_bool("QW36_METAL_QUANT_GPU",
                            fast || profile_lowmem);
    p->fp16_weights = env_bool("QW36_METAL_FP16_WEIGHTS",
                               profile_fp16 && !p->quant_gpu);

    const int qk_repack_env =
        env_bool("QW36_METAL_QK_REPACK",
                 env_bool("QW36_METAL_QK_AFFINE32", 0));
    p->qk_repack = p->quant_gpu && qk_repack_env;
    p->q4k_affine32 =
        p->qk_repack ||
        (p->quant_gpu && env_bool("QW36_METAL_Q4K_AFFINE32", fast));
    p->q5k_affine32 =
        p->qk_repack ||
        (p->quant_gpu && env_bool("QW36_METAL_Q5K_AFFINE32", fast));
    p->q6k_scale16 =
        p->quant_gpu && env_bool("QW36_METAL_Q6K_SCALE16", fast);
    p->quant_lm_head =
        p->quant_gpu && env_bool("QW36_METAL_QUANT_GPU_LM_HEAD", fast);

    p->fuse_dense_gate_up = 1;
    p->fuse_vanilla_qkv =
        p->fp16_weights && !p->quant_gpu &&
        env_bool("QW36_METAL_FUSE_QKV", 1);
    p->fuse_dn_qkvzab =
        p->fp16_weights && !p->quant_gpu &&
        env_bool("QW36_METAL_FUSE_DN_QKVZAB", 1);
}
