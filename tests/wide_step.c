/* Drive one decode step against a container whose shared expert is wider
 * than both the dense and routed widths, the shape Qwen2-57B-A14B ships.
 * The shared expert batches gate+up into m->ff, so if that buffer is sized
 * from the dense and routed widths alone the step writes past it. Token ids
 * go in directly, so no tokenizer is needed. */
#include <stdio.h>
#include "waste.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s CONTAINER\n", argv[0]); return 2; }
    waste_cfg cfg;
    waste_cfg_init(&cfg);
    waste_ctx *ctx = NULL;
    waste_status st = waste_open(argv[1], &cfg, &ctx);
    if (st != WASTE_OK) { fprintf(stderr, "open failed: %d\n", (int)st); return 1; }
    const int32_t toks[2] = {1, 2};
    const float *logits = NULL;
    size_t vocab = 0;
    st = waste_eval(ctx, toks, 2, &logits, &vocab);
    if (st != WASTE_OK) { fprintf(stderr, "eval failed: %d\n", (int)st); waste_close(ctx); return 1; }
    printf("step ok, vocab %zu, logit0 %.4f\n", vocab, logits ? logits[0] : 0.0f);
    waste_close(ctx);
    return 0;
}
