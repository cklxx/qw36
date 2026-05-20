/* qw36_ops.c - shared forward-state helpers for common implementation files.
 *
 * This file owns the package-private glue that keeps host and device copies
 * of the residual stream coherent while the forward pass is split across
 * attention, DeltaNet, and MLP modules. It deliberately does not allocate
 * model state; callers pass a qw36_forward_ctx built by qw36_forward.
 *
 * Return convention: 0 means the requested copy is available, and negative
 * values are forwarded by qw36_forward as engine errors. The lower-level math
 * primitives live beside their dequant helpers in qw36.c for now, but are
 * exposed through qw36_internal.h with the same qw36__ package prefix.
 */

#include "qw36_internal.h"

int qw36__ensure_x_host(qw36_forward_ctx *fc)
{
    if (!fc || !fc->st || !fc->x_host_valid || !fc->x_dev_valid)
        return -8;
    if (!*fc->x_host_valid && *fc->x_dev_valid) {
        if (qw36__state_download_to_host(fc->st, fc->st->x_dev, fc->st->x,
                                         fc->hidden * sizeof(float)))
            return -8;
        *fc->x_host_valid = 1;
    }
    return 0;
}

int qw36__ensure_x_dev(qw36_forward_ctx *fc)
{
    if (!fc || !fc->st || !fc->x_host_valid || !fc->x_dev_valid)
        return -8;
    if (!*fc->x_dev_valid && *fc->x_host_valid && fc->gpu_state) {
        if (qw36__state_copy_from_host(fc->st, fc->st->x_dev, fc->st->x,
                                       fc->hidden * sizeof(float)))
            return -8;
        *fc->x_dev_valid = 1;
    }
    return 0;
}
