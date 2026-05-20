/* qw36_cpu_stub.c — CPU "backend": no-op.
 *
 * Owner: Claude. Returning NULL tells the CLI to use the pure-C reference
 * forward pass implemented in common/qw36.c. Nothing more.
 */

#include "qw36_gpu.h"

qw36_gpu_backend *qw36_backend_create(void) { return NULL; }
