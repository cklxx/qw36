# amd/Makefile — owner: codex.

ROOT      ?= $(abspath ..)
COMMON     = $(ROOT)/common
HIPCC     ?= hipcc
CXXFLAGS  ?= -O3 -std=c++17 -Wall -Wextra
CFLAGS    ?= -O3 -std=c11   -Wall -Wextra
INCLUDES  := -I$(COMMON)
LDLIBS    ?= -lrocblas

COMMON_SRC := \
    $(COMMON)/qw36.c \
    $(COMMON)/qw36_dequant.c \
    $(COMMON)/qw36_ops.c \
    $(COMMON)/qw36_attn_vanilla.c \
    $(COMMON)/qw36_attn_deltanet.c \
    $(COMMON)/qw36_mlp.c \
    $(COMMON)/qw36_moe.c \
    $(COMMON)/qw36_gguf.c \
    $(COMMON)/qw36_tokenizer.c \
    $(COMMON)/qw36_cli.c

BACKEND_SRC := qw36_amd.cpp

OUT := $(ROOT)/qw36_amd

.PHONY: all clean

all: $(OUT)

$(OUT): $(BACKEND_SRC) $(COMMON_SRC) $(COMMON)/qw36.h $(COMMON)/qw36_gpu.h $(COMMON)/qw36_internal.h
	$(HIPCC) $(CXXFLAGS) $(INCLUDES) -x hip $(BACKEND_SRC) \
	    -x c $(COMMON_SRC) \
	    $(LDLIBS) -o $@

clean:
	rm -f $(OUT)
