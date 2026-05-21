# cpu/Makefile — CPU-only build, no GPU toolchain required.
#
# Owner: Claude. Used to smoke-test the engine end-to-end while the GPU
# backends are still being filled in. The binary uses the pure-C reference
# forward pass from common/qw36.c.

ROOT     ?= $(abspath ..)
COMMON    = $(ROOT)/common
CC       ?= cc
CFLAGS   ?= -O3 -std=c11 -Wall -Wextra -Wno-unused-function
INCLUDES := -I$(COMMON)
LDLIBS   := -lm

COMMON_SRC := \
    $(COMMON)/qw36.c \
    $(COMMON)/qw36_dequant.c \
    $(COMMON)/qw36_policy.c \
    $(COMMON)/qw36_ops.c \
    $(COMMON)/qw36_attn_vanilla.c \
    $(COMMON)/qw36_attn_deltanet.c \
    $(COMMON)/qw36_mlp.c \
    $(COMMON)/qw36_moe.c \
    $(COMMON)/qw36_kvcache.c \
    $(COMMON)/qw36_gguf.c \
    $(COMMON)/qw36_model.c \
    $(COMMON)/qw36_safetensors.c \
    $(COMMON)/qw36_tokenizer.c \
    $(COMMON)/qw36_cli.c

BACKEND_SRC := qw36_cpu_stub.c

OUT ?= $(ROOT)/qw36_cpu

.PHONY: all clean cpu_with_extra

all: $(OUT)

$(OUT): $(COMMON_SRC) $(BACKEND_SRC) $(COMMON)/qw36.h $(COMMON)/qw36_gpu.h $(COMMON)/qw36_internal.h
	$(CC) $(CFLAGS) $(INCLUDES) $(COMMON_SRC) $(BACKEND_SRC) -o $@ $(LDLIBS)

# Target used by top-level `make asan` / `make ubsan`. Caller sets
# CFLAGS_EXTRA, LDFLAGS_EXTRA, and OUT to override the release config.
cpu_with_extra: $(COMMON_SRC) $(BACKEND_SRC) $(COMMON)/qw36.h $(COMMON)/qw36_gpu.h $(COMMON)/qw36_internal.h
	$(CC) $(CFLAGS) $(CFLAGS_EXTRA) $(INCLUDES) $(COMMON_SRC) $(BACKEND_SRC) \
	    $(LDFLAGS_EXTRA) -o $(OUT) $(LDLIBS)

clean:
	rm -f $(OUT)
