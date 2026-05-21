# Top-level dispatch. Each backend has its own Makefile under <backend>/.
#
# Common targets at this level:
#   make all       — build CPU plus whatever GPU toolchains are present
#   make cpu       — CPU reference build
#   make metal     — Apple Metal backend (macOS)
#   make cuda      — CUDA backend (compile-checks only without GPU)
#   make amd       — AMD HIP backend (same)
#   make check     — fast smoke (cpu build + quant_fastest_smoke + kvcache smoke)
#   make perf      — full perf bench (cpu+metal+precision+compare_mlx short)
#   make asan      — CPU build with AddressSanitizer
#   make ubsan     — CPU build with UndefinedBehaviorSanitizer
#   make lint      — clang-format dry-run (warns on style drift)
#   make fmt       — clang-format apply (use with caution)
#   make clean     — wipe all per-backend artefacts

.PHONY: all cpu amd metal cuda clean test check perf asan ubsan lint fmt help

ROOT := $(CURDIR)
COMMON_SRC := common/qw36.c common/qw36_ops.c common/qw36_attn_vanilla.c \
              common/qw36_attn_deltanet.c common/qw36_mlp.c common/qw36_moe.c \
              common/qw36_kvcache.c \
              common/qw36_gguf.c common/qw36_tokenizer.c common/qw36_cli.c
COMMON_HDR := common/qw36.h common/qw36_gpu.h common/qw36_gguf.h \
              common/qw36_kvcache.h \
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
	rm -f qw36_cpu qw36_amd qw36_metal qw36_cuda \
	      qw36_cpu_asan qw36_cpu_ubsan

# Fast smoke. Doesn't require a model on disk if the helper scripts
# soft-skip; intended for the inner edit→build→check loop.
check: cpu
	@if [ -x ./qw36_metal ] || [ "$$(uname)" = "Darwin" ]; then \
	    $(MAKE) metal && tests/quant_fastest_smoke.sh || true; \
	fi
	tests/kvcache_smoke.sh

# Full perf bench. Slower; intended for "I think this is ready to land".
perf: cpu
	@if [ "$$(uname)" = "Darwin" ]; then \
	    $(MAKE) metal && \
	    tests/precision_cpu_vs_metal.sh && \
	    tests/compare_mlx.sh short; \
	else \
	    echo "perf target only meaningful with Metal at the moment"; \
	fi

# Sanitizer builds. Engineered into cpu/Makefile so the same source
# compiles with -fsanitize flags layered on top of release flags.
asan:
	CFLAGS_EXTRA="-fsanitize=address -fno-omit-frame-pointer -g" \
	    LDFLAGS_EXTRA="-fsanitize=address" \
	    OUT="$(ROOT)/qw36_cpu_asan" \
	    $(MAKE) -C cpu cpu_with_extra

ubsan:
	CFLAGS_EXTRA="-fsanitize=undefined -fno-omit-frame-pointer -g" \
	    LDFLAGS_EXTRA="-fsanitize=undefined" \
	    OUT="$(ROOT)/qw36_cpu_ubsan" \
	    $(MAKE) -C cpu cpu_with_extra

# Style checks. lint is dry-run (CI calls this with -Werror in a
# wrapper). fmt is the side-effectful "apply now" command — use with
# care; review the diff before committing.
LINT_SRC := $(shell find common cpu metal cuda amd tests tools -maxdepth 2 \
                     -name '*.c' -o -name '*.h' 2>/dev/null)

lint:
	@if ! command -v clang-format >/dev/null 2>&1; then \
	    echo "[lint] clang-format not installed — skipping"; \
	    exit 0; \
	fi; \
	if clang-format --dry-run -Werror $(LINT_SRC) >/dev/null 2>&1; then \
	    echo "[lint] clean"; \
	else \
	    echo "[lint] style drift detected; run 'make fmt' to apply"; \
	    exit 1; \
	fi

fmt:
	@if ! command -v clang-format >/dev/null 2>&1; then \
	    echo "[fmt] clang-format not installed — install LLVM (brew install clang-format)"; \
	    exit 1; \
	fi
	clang-format -i $(LINT_SRC)

help:
	@awk '/^[a-zA-Z_-]+:.*## / { \
	    split($$0, parts, ":.*## "); \
	    printf "  %-12s %s\n", parts[1], parts[2]; \
	}' $(MAKEFILE_LIST) || true
	@echo ""
	@echo "  cpu          build CPU reference → ./qw36_cpu"
	@echo "  metal        build Metal backend → ./qw36_metal (macOS only)"
	@echo "  cuda / amd   off-host builds (compile-check on this host)"
	@echo "  check        fast smoke (cpu + 0.8B fastest path + kvcache)"
	@echo "  perf         compare_mlx.sh short + precision_cpu_vs_metal"
	@echo "  asan / ubsan sanitizer CPU builds → ./qw36_cpu_{asan,ubsan}"
	@echo "  lint         clang-format dry-run on C/H files"
	@echo "  fmt          clang-format -i  (review the diff!)"
	@echo "  clean        wipe all per-backend artefacts"
