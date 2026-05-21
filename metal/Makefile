# metal/Makefile — owner: codex.
# Builds default.metallib from qw36_metal.metal and links the C/Obj-C host.

ROOT     ?= $(abspath ..)
COMMON    = $(ROOT)/common
CC       ?= clang
CFLAGS   ?= -O3 -std=c11 -Wall -Wextra -fobjc-arc
INCLUDES := -I$(COMMON)
FRAMEWORKS := -framework Foundation -framework Metal -framework MetalKit \
    -framework MetalPerformanceShaders

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
    $(COMMON)/qw36_tokenizer.c \
    $(COMMON)/qw36_cli.c

BACKEND_SRC := qw36_metal.m
METAL_SRC   := qw36_metal.metal

OUT     := $(ROOT)/qw36_metal
METALLIB := default.metallib

.PHONY: all clean

all: $(OUT) $(METALLIB)

$(METALLIB): $(METAL_SRC)
	xcrun -sdk macosx metal    -c $(METAL_SRC) -o $(METAL_SRC:.metal=.air)
	xcrun -sdk macosx metallib $(METAL_SRC:.metal=.air) -o $@

$(OUT): $(BACKEND_SRC) $(COMMON_SRC) $(COMMON)/qw36.h $(COMMON)/qw36_gpu.h $(COMMON)/qw36_internal.h
	$(CC) $(CFLAGS) $(INCLUDES) $(BACKEND_SRC) $(COMMON_SRC) \
	    $(FRAMEWORKS) -o $@

clean:
	rm -f $(OUT) $(METALLIB) qw36_metal.air
