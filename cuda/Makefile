# cuda/Makefile — owner: codex.

ROOT     ?= $(abspath ..)
COMMON    = $(ROOT)/common
NVCC     ?= nvcc
NVCCFLAGS?= -O3 -std=c++17 -Xcompiler "-Wall -Wextra" -arch=native
INCLUDES := -I$(COMMON)
LDLIBS   ?= -lcublas

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

BACKEND_SRC := qw36_cuda.cu

OUT := $(ROOT)/qw36_cuda

.PHONY: all clean

all: $(OUT)

$(OUT): $(BACKEND_SRC) $(COMMON_SRC) $(COMMON)/qw36.h $(COMMON)/qw36_gpu.h $(COMMON)/qw36_internal.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -x cu $(BACKEND_SRC) \
	    -x c $(COMMON_SRC) \
	    $(LDLIBS) -o $@

clean:
	rm -f $(OUT)
