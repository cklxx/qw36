/* qw36_cli.c — main entry. Links against exactly one backend.
 *
 * Owner: Claude. Same source compiles into all three binaries.
 */

#include "qw36.h"
#include "qw36_gpu.h"
#include "qw36_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s -m <model.gguf> [-p <prompt>] [-n <max_new_tokens>]\n"
        "          [-t <temperature>] [--top-p <p>] [--top-k <k>] [--seed <u64>]\n"
        "          [--interactive]\n"
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
} qw36_cli_args;

static int parse_args(int argc, char **argv, qw36_cli_args *a) {
    a->max_new = 128;
    a->temperature = 0.0f;
    a->top_p = 1.0f;
    a->top_k = 0;
    a->seed = 42;
    for (int i = 1; i < argc; i++) {
        const char *s = argv[i];
        #define EAT() (i + 1 < argc ? argv[++i] : NULL)
        if (!strcmp(s, "-m")) a->model_path = EAT();
        else if (!strcmp(s, "-p")) a->prompt = EAT();
        else if (!strcmp(s, "-n")) a->max_new = atoi(EAT());
        else if (!strcmp(s, "-t")) a->temperature = (float)atof(EAT());
        else if (!strcmp(s, "--top-p")) a->top_p = (float)atof(EAT());
        else if (!strcmp(s, "--top-k")) a->top_k = atoi(EAT());
        else if (!strcmp(s, "--seed")) a->seed = strtoull(EAT(), NULL, 10);
        else if (!strcmp(s, "--interactive")) a->interactive = 1;
        else if (!strcmp(s, "-h") || !strcmp(s, "--help")) { usage(argv[0]); return 1; }
        else { fprintf(stderr, "unknown arg: %s\n", s); usage(argv[0]); return -1; }
        #undef EAT
    }
    if (!a->model_path) { usage(argv[0]); return -1; }
    return 0;
}

int main(int argc, char **argv) {
    qw36_cli_args a = {0};
    int rc = parse_args(argc, argv, &a);
    if (rc) return rc < 0 ? 1 : 0;

    qw36_gpu_backend *be = qw36_backend_create();
    if (!be) {
        fprintf(stderr, "qw36: backend init failed\n");
        return 2;
    }

    char err[256] = {0};
    qw36_engine *eng = qw36_engine_open(a.model_path, be, err, sizeof(err));
    if (!eng) {
        fprintf(stderr, "qw36: %s\n", err);
        return 3;
    }

    /* TODO(Claude): tokenize prompt, prefill, sample loop, decode tokens.
     * For now just print the config we loaded. */
    (void)a;
    qw36_engine_close(eng);
    return 0;
}
