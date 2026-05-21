/* qw36_cli.c — main entry. Same source compiles into all binaries.
 *
 * Owner: Claude. CPU-only path works when qw36_backend_create() returns NULL.
 */

#include "qw36_internal.h"
#include "qw36.h"
#include "qw36_gpu.h"
#include "qw36_gguf.h"
#include "qw36_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>          /* strcasecmp */
#include <time.h>

/* Provided by the backend object linked in (amd|metal|cuda|cpu). The CPU
 * build links cpu/qw36_cpu_stub.c, which just returns NULL → CLI falls
 * back to the reference forward path in qw36.c. */
qw36_gpu_backend *qw36_backend_create(void);

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s -m <model.gguf> [-p <prompt>] [-n <max_new_tokens>]\n"
        "          [-t <temperature>] [--top-p <p>] [--top-k <k>] [--seed <u64>]\n"
        "          [--seq <capacity>] [--interactive] [--no-special]\n"
        "          [--profile reference|fp16|lowmem|fast] [--fast] [--strict]\n"
        "          [--layer-trace <L> --layer-trace-out <path>] [--layer-trace-pos <pos>]\n"
        "          [--info]            print config + tokenizer summary and exit\n"
        "          [--print-config]    print effective profile + all QW36_* env knobs and exit\n"
        "          [--doctor]          run preflight checks (SDK, model, GPU, env) and exit\n"
        "\n"
        "binary: %s\n", prog, qw36_version());
}

typedef struct {
    const char *model_path;
    const char *prompt;
    int   max_new;
    float temperature;
    float top_p;
    int   top_k;
    uint64_t seed;
    int   interactive;
    int   info_only;
    int   seq_capacity;
    int   no_special;
    int   debug_top;     /* print top-K logits per step */
    int   dump_tokens;   /* tokenize the prompt and exit */
    int   layer_trace;   /* vanilla layer index to dump, -1 disables */
    int   layer_trace_pos; /* optional seq_pos filter for trace */
    const char *profile;
    int   fast_profile;
    int   strict_profile;
    const char *layer_trace_out;
    int   print_config;  /* --print-config: dump effective env + exit */
    int   doctor;        /* --doctor: preflight checks + exit */
} qw36_cli_args;

/* QW36_* env knob inventory used by --print-config and --doctor. Keep in
 * sync with docs/env_knobs.md when adding a knob. Source-of-truth is
 * the file:line cited in the docs. */
typedef struct {
    const char *name;
    const char *lifecycle; /* "stable" | "internal" | "research" | "debug" */
    const char *summary;
} qw36_env_knob_doc;

static const qw36_env_knob_doc qw36_env_knobs[] = {
    /* stable */
    { "QW36_PROFILE",                 "stable",
      "backend policy: reference|fp16|lowmem|fast" },
    { "QW36_METAL_FAST",              "stable",
      "umbrella for --fast (sets the full quant + lm_head + KV path)" },
    { "QW36_METAL_QUANT_GPU",         "stable",
      "keep K-quant blocks on GPU; per-row dequant kernels" },
    { "QW36_METAL_FP16_WEIGHTS",      "stable",
      "materialize quants to fp16 before upload (MPS path)" },
    { "QW36_METAL_FP16_KV",           "stable",
      "fp16 K/V cache" },
    /* internal (set by profile) */
    { "QW36_METAL_Q4K_AFFINE32",      "internal",
      "Q4_K → per-32 affine repack + qmv kernel" },
    { "QW36_METAL_Q5K_AFFINE32",      "internal",
      "Q5_K → per-32 affine repack + qmv kernel" },
    { "QW36_METAL_Q6K_SCALE16",       "internal",
      "Q6_K → per-16 scale repack + qmv kernel" },
    { "QW36_METAL_QUANT_GPU_LM_HEAD", "internal",
      "decouple tied lm_head and run as Q6K_SCALE16 on GPU" },
    { "QW36_METAL_Q4K_AFFINE32_MLX",  "internal",
      "MLX-style Q4K qdot (default on under quant gpu)" },
    { "QW36_METAL_Q5K_AFFINE32_MLX",  "internal",
      "MLX-style Q5K qdot (opt-in; wash on this host)" },
    { "QW36_METAL_Q6K_SCALE16_MLX",   "internal",
      "MLX-style Q6K qdot (default on under quant gpu)" },
    /* long-context */
    { "QW36_METAL_KV_TRANSPOSED",     "internal",
      "auto|0|1 — transposed K/V layout; auto flips on at seq_capacity > 512" },
    { "QW36_METAL_ATTN_X4",           "research",
      "x4 batched scoring attention variant (~0% on this host)" },
    /* DN / MoE probes */
    { "QW36_METAL_FUSE_QKV",          "internal",
      "fuse vanilla QKV concat into one matmul" },
    { "QW36_METAL_FUSE_DN_QKVZAB",    "internal",
      "fuse DN 4-projection concat into one matmul" },
    { "QW36_METAL_FUSE_DN_CONV",      "internal",
      "fused conv1d + gated_delta Metal kernel" },
    { "QW36_METAL_FUSE_DN_TAIL",      "internal",
      "fused gated_rmsnorm + dn_out tail kernel" },
    { "QW36_METAL_DN_TAIL_DIRECT",    "research",
      "dispatch DN tail without the fused kernel (debug)" },
    { "QW36_SKIP_DN",                 "debug",
      "skip all DeltaNet layers (vanilla-only forward)" },
    { "QW36_SKIP_CONV1D",             "debug",
      "skip conv1d step in DN forward" },
    { "QW36_DN_A_RAW",                "debug",
      "skip logf(|a|) transform on ssm_a weight load" },
    { "QW36_DN_KH_MOD",               "debug",
      "alternative k-head / v-head pairing for DN bisection" },
    /* tracing */
    { "QW36_DEBUG_LAYER",             "debug",
      "per-layer ||x|| trace" },
    { "QW36_MAX_LAYERS",              "debug",
      "stop forward after the first N transformer layers" },
    { "QW36_BYPASS_LAYERS",           "debug",
      "skip specific layer ids / ranges" },
    { "QW36_TRACE_LAYER",             "debug",
      "dump intermediate tensors to binary file" },
    { "QW36_TRACE_POS",               "debug",
      "filter trace by seq_pos" },
    { "QW36_TRACE_TAKE",              "debug",
      "max trace samples to keep" },
    { "QW36_TRACE_OUT",               "debug",
      "trace output file" },
    { "QW36_USE_MROPE_SECTIONS",      "debug",
      "section-based mRoPE (off by default; metadata often wrong)" },
    { "QW36_METAL_PERF",              "debug",
      "per-kernel [metal-perf] table; disables persistent encoder" },
    { "QW36_METAL_TIMING",            "debug",
      "per-forward wallclock log" },
};
#define QW36_ENV_KNOB_COUNT (sizeof(qw36_env_knobs) / sizeof(qw36_env_knobs[0]))

/* --print-config: dump the effective profile + every env knob's value.
 * Provenance for now is "set" / "unset". A future iteration could
 * distinguish default / profile / env / CLI sources. */
static void cli_print_config(const qw36_cli_args *a) {
    const char *profile = getenv("QW36_PROFILE");
    fprintf(stdout, "qw36 effective config (binary: %s)\n", qw36_version());
    fprintf(stdout, "  profile               = %s\n",
            profile && *profile ? profile : "(default reference)");
    fprintf(stdout, "  model_path            = %s\n",
            a->model_path ? a->model_path : "(unset; --doctor will FAIL)");
    fprintf(stdout, "  seq_capacity          = %d\n", a->seq_capacity);
    fprintf(stdout, "  max_new_tokens        = %d\n", a->max_new);
    fprintf(stdout, "  temperature           = %g (top_p=%g top_k=%d seed=%llu)\n",
            (double)a->temperature, (double)a->top_p, a->top_k,
            (unsigned long long)a->seed);
    fprintf(stdout, "\nQW36_* env knobs:\n");
    fprintf(stdout, "  %-32s %-9s %-5s  %s\n",
            "name", "lifecycle", "value", "summary");
    for (size_t i = 0; i < QW36_ENV_KNOB_COUNT; i++) {
        const qw36_env_knob_doc *k = &qw36_env_knobs[i];
        const char *v = getenv(k->name);
        fprintf(stdout, "  %-32s %-9s %-5s  %s\n",
                k->name, k->lifecycle, v && *v ? v : "—", k->summary);
    }
    fprintf(stdout, "\nLifecycle: stable = user API; internal = profile-driven; "
                   "research / debug = opt-in.\n");
    fprintf(stdout, "Reference: docs/env_knobs.md.\n");
}

/* --doctor: preflight checks. Returns 0 if all OK, 1 if any FAIL.
 * WARN-level findings still return 0; they're advisory.
 *
 * Checks ordered by what fails first when something is wrong on a
 * fresh machine: build artifacts → backend → model → env conflicts. */
static int cli_doctor(const qw36_cli_args *a) {
    int fail = 0;
    int warn = 0;
    #define DOCTOR_OK(label, fmt, ...) \
        fprintf(stdout, "  [OK]   %-22s " fmt "\n", label, ##__VA_ARGS__)
    #define DOCTOR_WARN(label, fmt, ...) do { \
        fprintf(stdout, "  [WARN] %-22s " fmt "\n", label, ##__VA_ARGS__); \
        warn++; \
    } while (0)
    #define DOCTOR_FAIL(label, fmt, ...) do { \
        fprintf(stdout, "  [FAIL] %-22s " fmt "\n", label, ##__VA_ARGS__); \
        fail++; \
    } while (0)

    fprintf(stdout, "qw36 --doctor (binary: %s)\n", qw36_version());

    /* Backend object linked in. NULL → CPU stub. */
    qw36_gpu_backend *be = qw36_backend_create();
    if (be && be->name) {
        DOCTOR_OK("backend", "%s available", be->name);
    } else {
        DOCTOR_OK("backend", "cpu (reference); no GPU backend linked");
    }

    /* Model path. */
    if (!a->model_path || !*a->model_path) {
        DOCTOR_WARN("model path", "no -m <model.gguf> given — "
                                  "doctor cannot validate model");
    } else {
        FILE *f = fopen(a->model_path, "rb");
        if (!f) {
            DOCTOR_FAIL("model file",
                "%s: cannot open (check path, perms)", a->model_path);
        } else {
            unsigned char magic[4] = {0};
            size_t n = fread(magic, 1, 4, f);
            fclose(f);
            if (n != 4 || memcmp(magic, "GGUF", 4) != 0) {
                DOCTOR_FAIL("model file",
                    "%s: not a GGUF file (magic mismatch)", a->model_path);
            } else {
                DOCTOR_OK("model file", "%s", a->model_path);
            }
        }
    }

    /* Profile sanity. */
    const char *profile = getenv("QW36_PROFILE");
    if (profile && *profile && !qw36__profile_name_is_valid(profile)) {
        DOCTOR_FAIL("QW36_PROFILE",
            "'%s' is not a known profile (reference|fp16|lowmem|fast)",
            profile);
    } else {
        DOCTOR_OK("QW36_PROFILE", "%s",
            profile && *profile ? profile : "unset (defaults to reference)");
    }

    /* Conflict: --fast + QW36_METAL_FP16_WEIGHTS=1 contradict. */
    const char *fast = getenv("QW36_METAL_FAST");
    const char *fp16 = getenv("QW36_METAL_FP16_WEIGHTS");
    if (fast && atoi(fast) && fp16 && atoi(fp16)) {
        DOCTOR_WARN("env conflict",
            "QW36_METAL_FAST=1 and QW36_METAL_FP16_WEIGHTS=1 both set; "
            "fast wins, fp16 setting is ignored");
    }

    /* Conflict: QW36_METAL_QUANT_GPU=0 + fast profile is incoherent. */
    const char *qgpu = getenv("QW36_METAL_QUANT_GPU");
    if (fast && atoi(fast) && qgpu && !atoi(qgpu)) {
        DOCTOR_FAIL("env conflict",
            "QW36_METAL_FAST=1 requires QW36_METAL_QUANT_GPU=1 but you set =0");
    }

    /* KV transposed tri-state validation. */
    const char *kvt = getenv("QW36_METAL_KV_TRANSPOSED");
    if (kvt && *kvt) {
        if (strcasecmp(kvt, "auto") && strcasecmp(kvt, "0") &&
            strcasecmp(kvt, "1") && strcasecmp(kvt, "on") &&
            strcasecmp(kvt, "off") && strcasecmp(kvt, "true") &&
            strcasecmp(kvt, "false")) {
            DOCTOR_WARN("QW36_METAL_KV_TRANSPOSED",
                "value '%s' is treated as 'on'; use auto|0|1 for clarity",
                kvt);
        } else {
            DOCTOR_OK("QW36_METAL_KV_TRANSPOSED", "%s", kvt);
        }
    } else {
        DOCTOR_OK("QW36_METAL_KV_TRANSPOSED", "auto (flips on at seq > 512)");
    }

    fprintf(stdout, "\nResult: %d FAIL, %d WARN. %s\n",
            fail, warn,
            fail ? "fix the FAILs above and re-run." :
                   warn ? "WARN entries are advisory." : "all clear.");
    return fail ? 1 : 0;
    #undef DOCTOR_OK
    #undef DOCTOR_WARN
    #undef DOCTOR_FAIL
}

static int parse_args(int argc, char **argv, qw36_cli_args *a) {
    a->max_new      = 128;
    a->temperature  = 0.0f;
    a->top_p        = 1.0f;
    a->top_k        = 0;
    a->seed         = 42;
    a->seq_capacity = 2048;
    a->layer_trace  = -1;
    a->layer_trace_pos = -1;
    for (int i = 1; i < argc; i++) {
        const char *s = argv[i];
        #define EAT() (i + 1 < argc ? argv[++i] : NULL)
        if      (!strcmp(s, "-m"))            a->model_path = EAT();
        else if (!strcmp(s, "-p"))            a->prompt = EAT();
        else if (!strcmp(s, "-n"))            a->max_new = atoi(EAT());
        else if (!strcmp(s, "-t"))            a->temperature = (float)atof(EAT());
        else if (!strcmp(s, "--top-p"))       a->top_p = (float)atof(EAT());
        else if (!strcmp(s, "--top-k"))       a->top_k = atoi(EAT());
        else if (!strcmp(s, "--seed"))        a->seed = strtoull(EAT(), NULL, 10);
        else if (!strcmp(s, "--seq"))         a->seq_capacity = atoi(EAT());
        else if (!strcmp(s, "--interactive")) a->interactive = 1;
        else if (!strcmp(s, "--info"))        a->info_only = 1;
        else if (!strcmp(s, "--no-special"))  a->no_special = 1;
        else if (!strcmp(s, "--profile"))     a->profile = EAT();
        else if (!strcmp(s, "--fast"))        a->fast_profile = 1;
        else if (!strcmp(s, "--strict"))      a->strict_profile = 1;
        else if (!strcmp(s, "--debug-top"))   a->debug_top = atoi(EAT());
        else if (!strcmp(s, "--dump-tokens")) a->dump_tokens = 1;
        else if (!strcmp(s, "--layer-trace")) a->layer_trace = atoi(EAT());
        else if (!strcmp(s, "--layer-trace-out")) a->layer_trace_out = EAT();
        else if (!strcmp(s, "--layer-trace-pos")) a->layer_trace_pos = atoi(EAT());
        else if (!strcmp(s, "--print-config")) a->print_config = 1;
        else if (!strcmp(s, "--doctor"))      a->doctor = 1;
        else if (!strcmp(s, "-h") || !strcmp(s, "--help")) { usage(argv[0]); return 1; }
        else { fprintf(stderr, "unknown arg: %s\n", s); usage(argv[0]); return -1; }
        #undef EAT
    }
    /* --print-config and --doctor don't require a model. */
    if (!a->model_path && !a->print_config && !a->doctor) {
        usage(argv[0]); return -1;
    }
    return 0;
}

static int apply_profile_args(const qw36_cli_args *a)
{
    const char *profile = a->profile;
    if (a->fast_profile) profile = "fast";
    if (a->strict_profile) {
        if (a->fast_profile || a->profile) {
            fprintf(stderr, "qw36: --strict cannot be combined with --fast/--profile\n");
            return -1;
        }
        profile = "reference";
    }
    if (!profile) return 0;
    if (!qw36__profile_name_is_valid(profile)) {
        fprintf(stderr,
                "qw36: unknown profile '%s' (expected reference, fp16, lowmem, fast)\n",
                profile);
        return -1;
    }
    setenv("QW36_PROFILE", profile, 1);
    if (!strcmp(profile, "fast") || !strcmp(profile, "serving")) {
        setenv("QW36_METAL_FAST", "1", 1);
        setenv("QW36_METAL_QUANT_GPU", "1", 1);
        setenv("QW36_METAL_Q4K_AFFINE32", "1", 1);
        setenv("QW36_METAL_Q5K_AFFINE32", "1", 1);
        setenv("QW36_METAL_Q6K_SCALE16", "1", 1);
        setenv("QW36_METAL_QUANT_GPU_LM_HEAD", "1", 1);
    } else if (!strcmp(profile, "lowmem") || !strcmp(profile, "quant")) {
        setenv("QW36_METAL_FAST", "0", 1);
        setenv("QW36_METAL_QUANT_GPU", "1", 1);
        setenv("QW36_METAL_FP16_WEIGHTS", "0", 1);
    } else if (!strcmp(profile, "fp16")) {
        setenv("QW36_METAL_FAST", "0", 1);
        setenv("QW36_METAL_QUANT_GPU", "0", 1);
        setenv("QW36_METAL_FP16_WEIGHTS", "1", 1);
    } else if (!strcmp(profile, "reference") || !strcmp(profile, "strict") ||
               !strcmp(profile, "balanced") || !strcmp(profile, "default")) {
        setenv("QW36_METAL_FAST", "0", 1);
        setenv("QW36_METAL_QUANT_GPU", "0", 1);
        setenv("QW36_METAL_FP16_WEIGHTS", "0", 1);
        setenv("QW36_METAL_Q4K_AFFINE32", "0", 1);
        setenv("QW36_METAL_Q5K_AFFINE32", "0", 1);
        setenv("QW36_METAL_Q6K_SCALE16", "0", 1);
        setenv("QW36_METAL_QUANT_GPU_LM_HEAD", "0", 1);
    }
    return 0;
}

static double mono_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void print_info(qw36_engine *eng, qw36_tokenizer *tok) {
    const qw36_config *c = qw36_engine_config(eng);
    fprintf(stderr,
        "qw36 info:\n"
        "  hidden=%u  intermediate=%u  layers=%u\n"
        "  heads=%u  kv_heads=%u  head_dim=%u\n"
        "  vocab=%u  context=%u\n"
        "  rms_eps=%g  rope_theta=%g  partial_rotary=%g\n"
        "  moe_experts=%u  moe_per_tok=%u  moe_inter=%u\n"
        "  bos=%u eos=%u im_start=%u im_end=%u\n",
        c->hidden_size, c->intermediate_size, c->num_hidden_layers,
        c->num_attention_heads, c->num_key_value_heads, c->head_dim,
        c->vocab_size, c->max_position_embeddings,
        c->rms_norm_eps, c->rope_theta, c->partial_rotary_factor,
        c->moe_num_experts, c->moe_experts_per_tok, c->moe_intermediate_size,
        qw36_tokenizer_bos(tok), qw36_tokenizer_eos(tok),
        qw36_tokenizer_im_start(tok), qw36_tokenizer_im_end(tok));
}

/* Build the Qwen3 chat-template wrapped prompt:
 *   <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
 * If --no-special is set, just use the raw prompt. */
static int build_prompt_ids(const qw36_tokenizer *tok, const char *prompt,
                            int wrap_chat,
                            uint32_t **out_ids, size_t *out_n)
{
    const char *full = prompt;
    char *scratch = NULL;
    if (wrap_chat) {
        size_t plen = strlen(prompt);
        scratch = (char *)malloc(plen + 64);
        if (!scratch) return -1;
        /* We rely on the tokenizer's special-token vocab containing
         * "<|im_start|>" / "<|im_end|>" — load_specials wires those up. */
        snprintf(scratch, plen + 64,
                 "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
                 prompt);
        full = scratch;
    }

    size_t cap = strlen(full) * 4 + 16;
    uint32_t *ids = (uint32_t *)malloc(cap * sizeof(uint32_t));
    if (!ids) { free(scratch); return -1; }
    size_t n = cap;
    int rc = qw36_tokenizer_encode(tok, full, ids, &n);
    free(scratch);
    if (rc) { free(ids); return -1; }
    *out_ids = ids;
    *out_n   = n;
    return 0;
}

int main(int argc, char **argv) {
    qw36_cli_args a = {0};
    a.layer_trace = -1;  /* reinforce parse_args default */
    int rc = parse_args(argc, argv, &a);
    if (rc) return rc < 0 ? 1 : 0;
    if (apply_profile_args(&a)) return 1;

    /* --print-config / --doctor short-circuit before opening the model;
     * they need to be usable on a broken/missing-model host. */
    if (a.print_config) { cli_print_config(&a); return 0; }
    if (a.doctor)       { return cli_doctor(&a); }

    /* Backend: NULL ⇒ CPU reference path. */
    qw36_gpu_backend *be = qw36_backend_create();
    fprintf(stderr, "qw36: backend = %s\n", be ? be->name : "cpu (reference)");

    char err[256] = {0};
    qw36_engine *eng = qw36_engine_open(a.model_path, be, err, sizeof(err));
    if (!eng) { fprintf(stderr, "qw36: %s\n", err); return 3; }

    qw36_tokenizer *tok = qw36_tokenizer_new(qw36_engine_gguf(eng),
                                             err, sizeof(err));
    if (!tok) { fprintf(stderr, "qw36: tokenizer: %s\n", err); qw36_engine_close(eng); return 4; }

    if (a.info_only) {
        /* Layer-type detection: vanilla Qwen3 uses blk.X.attn_q.weight per
         * layer; Qwen3.5/3.6 uses fused attn_qkv + ssm_* (Gated DeltaNet). */
        const struct qw36_gguf_file *gf = qw36_engine_gguf(eng);
        int n_layers = (int)qw36_engine_config(eng)->num_hidden_layers;
        int has_q = 0, has_fused = 0, has_ssm = 0;
        for (int l = 0; l < n_layers; l++) {
            char buf[128]; qw36_gguf_tensor t;
            snprintf(buf, sizeof(buf), "blk.%d.attn_q.weight", l);
            if (qw36_gguf_get_tensor(gf, buf, &t) == 0) has_q++;
            snprintf(buf, sizeof(buf), "blk.%d.attn_qkv.weight", l);
            if (qw36_gguf_get_tensor(gf, buf, &t) == 0) has_fused++;
            snprintf(buf, sizeof(buf), "blk.%d.ssm_conv1d.weight", l);
            if (qw36_gguf_get_tensor(gf, buf, &t) == 0) has_ssm++;
        }
        fprintf(stderr, "qw36 layer survey: vanilla_q=%d/%d  fused_qkv=%d/%d  ssm=%d/%d\n",
                has_q, n_layers, has_fused, n_layers, has_ssm, n_layers);
        print_info(eng, tok);
        qw36_tokenizer_free(tok);
        qw36_engine_close(eng);
        return 0;
    }

    if (!a.prompt) {
        fprintf(stderr, "qw36: -p <prompt> required (or use --info)\n");
        qw36_tokenizer_free(tok); qw36_engine_close(eng); return 5;
    }

    if (a.dump_tokens) {
        uint32_t *pids = NULL; size_t pn = 0;
        if (build_prompt_ids(tok, a.prompt, !a.no_special, &pids, &pn)) {
            fprintf(stderr, "qw36: tokenize failed\n");
            qw36_tokenizer_free(tok); qw36_engine_close(eng); return 7;
        }
        fprintf(stderr, "tokens (%zu):\n", pn);
        for (size_t i = 0; i < pn; i++) {
            size_t blen = 0;
            const char *s = qw36_tokenizer_decode_one(tok, pids[i], &blen);
            fprintf(stderr, "  [%zu] id=%u  ", i, pids[i]);
            if (s) { fwrite(s, 1, blen, stderr); }
            fputc('\n', stderr);
        }
        free(pids); qw36_tokenizer_free(tok); qw36_engine_close(eng); return 0;
    }

    qw36_state *st = qw36_state_new(eng, (uint32_t)a.seq_capacity);
    if (!st) { fprintf(stderr, "qw36: state alloc failed\n");
               qw36_tokenizer_free(tok); qw36_engine_close(eng); return 6; }

    uint32_t *pids = NULL; size_t pn = 0;
    if (build_prompt_ids(tok, a.prompt, !a.no_special, &pids, &pn)) {
        fprintf(stderr, "qw36: tokenize failed\n");
        qw36_state_free(st); qw36_tokenizer_free(tok); qw36_engine_close(eng); return 7;
    }
    if (a.layer_trace >= 0) {
        if (!a.layer_trace_out || !a.layer_trace_out[0]) {
            fprintf(stderr, "qw36: --layer-trace requires --layer-trace-out <path>\n");
            free(pids); qw36_state_free(st); qw36_tokenizer_free(tok);
            qw36_engine_close(eng); return 7;
        }
        char trace_buf[32];
        snprintf(trace_buf, sizeof(trace_buf), "%d", a.layer_trace);
        setenv("QW36_TRACE_LAYER", trace_buf, 1);
        setenv("QW36_TRACE_OUT", a.layer_trace_out, 1);
        if (a.layer_trace_pos >= 0) {
            snprintf(trace_buf, sizeof(trace_buf), "%d", a.layer_trace_pos);
            setenv("QW36_TRACE_POS", trace_buf, 1);
        } else {
            unsetenv("QW36_TRACE_POS");
        }
    }
    fprintf(stderr, "qw36: prompt tokens = %zu\n", pn);

    double t0 = mono_seconds();
    if (qw36_prefill(eng, st, pids, pn) != 0) {
        fprintf(stderr, "qw36: prefill failed\n");
        free(pids); qw36_state_free(st); qw36_tokenizer_free(tok);
        qw36_engine_close(eng); return 8;
    }
    double t1 = mono_seconds();
    fprintf(stderr, "qw36: prefill %zu tokens in %.3fs (%.1f tok/s)\n",
            pn, t1 - t0, (double)pn / (t1 - t0));

    qw36_sampler sp = {
        .temperature = a.temperature,
        .top_p       = a.top_p,
        .top_k       = a.top_k,
        .rng_seed    = a.seed,
    };

    /* Generate. After prefill, st->logits is the distribution at the last
     * prompt token. Sample, print, feed back. */
    const uint32_t eos = qw36_tokenizer_eos(tok);
    const uint32_t imend = qw36_tokenizer_im_end(tok);
    int generated = 0;
    double tg0 = mono_seconds();
    const uint32_t vocab = qw36_engine_config(eng)->vocab_size;
    for (int i = 0; i < a.max_new; i++) {
        if (a.debug_top > 0) {
            /* Print top-K logits for the current step (before sampling). */
            int K = a.debug_top;
            float *tk_val = (float *)alloca(K * sizeof(float));
            uint32_t *tk_idx = (uint32_t *)alloca(K * sizeof(uint32_t));
            for (int j = 0; j < K; j++) { tk_val[j] = -1e30f; tk_idx[j] = 0; }
            for (uint32_t v = 0; v < vocab; v++) {
                if (st->logits[v] <= tk_val[K-1]) continue;
                int j = K - 1;
                while (j > 0 && st->logits[v] > tk_val[j-1]) {
                    tk_val[j] = tk_val[j-1]; tk_idx[j] = tk_idx[j-1]; j--;
                }
                tk_val[j] = st->logits[v]; tk_idx[j] = v;
            }
            fprintf(stderr, "[step %d] top-%d:\n", i, K);
            for (int j = 0; j < K; j++) {
                size_t blen = 0;
                const char *s = qw36_tokenizer_decode_one(tok, tk_idx[j], &blen);
                fprintf(stderr, "  %.4f  id=%u  '", tk_val[j], tk_idx[j]);
                if (s) fwrite(s, 1, blen, stderr);
                fprintf(stderr, "'\n");
            }
        }
        uint32_t next = qw36_sample(st->logits, vocab, &sp);
        if (next == eos || (imend && next == imend)) break;
        size_t blen = 0;
        const char *s = qw36_tokenizer_decode_one(tok, next, &blen);
        if (s && blen) {
            fwrite(s, 1, blen, stdout);
            fflush(stdout);
        }
        if (qw36_forward(eng, st, next) != 0) break;
        generated++;
    }
    double tg1 = mono_seconds();
    fputc('\n', stdout);
    fprintf(stderr, "qw36: generated %d tokens in %.3fs (%.1f tok/s)\n",
            generated, tg1 - tg0,
            generated > 0 ? (double)generated / (tg1 - tg0) : 0.0);

    free(pids);
    qw36_state_free(st);
    qw36_tokenizer_free(tok);
    qw36_engine_close(eng);
    return 0;
}
