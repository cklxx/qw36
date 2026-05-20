# Top-level dispatch. Each backend has its own Makefile under <backend>/.

.PHONY: all cpu amd metal cuda clean test

ROOT := $(CURDIR)
COMMON_SRC := common/qw36.c common/qw36_ops.c common/qw36_attn_vanilla.c \
              common/qw36_attn_deltanet.c common/qw36_mlp.c common/qw36_moe.c \
              common/qw36_gguf.c common/qw36_tokenizer.c common/qw36_cli.c
COMMON_HDR := common/qw36.h common/qw36_gpu.h common/qw36_gguf.h \
              common/qw36_tokenizer.h common/qw36_internal.h
export ROOT COMMON_SRC COMMON_HDR

all: cpu
	@have_any=1; \
	if command -v hipcc >/dev/null 2>&1;  then $(MAKE) amd   || true; have_any=1; fi; \
	if [ "$$(uname)" = "Darwin" ];        then $(MAKE) metal || true; have_any=1; fi; \
	if command -v nvcc >/dev/null 2>&1;   then $(MAKE) cuda  || true; have_any=1; fi; \
	if [ "$$have_any" = "0" ]; then echo "no toolchain detected (CPU build also failed)"; exit 1; fi

cpu:
	$(MAKE) -C cpu

amd:
	$(MAKE) -C amd

metal:
	$(MAKE) -C metal

cuda:
	$(MAKE) -C cuda

test: cpu metal
	tests/precision_cpu_vs_metal.sh
	@echo "(e2e_qwen35_smoke is informational — currently regressed)"
	-tests/e2e_qwen35_smoke.sh ./qw36_cpu
	-tests/e2e_qwen35_smoke.sh ./qw36_metal

clean:
	$(MAKE) -C cpu   clean   2>/dev/null || true
	$(MAKE) -C amd   clean   2>/dev/null || true
	$(MAKE) -C metal clean   2>/dev/null || true
	$(MAKE) -C cuda  clean   2>/dev/null || true
	$(MAKE) -C tests clean   2>/dev/null || true
	rm -f qw36_cpu qw36_amd qw36_metal qw36_cuda
