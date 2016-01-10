#include "seq2seq.h"
#include "util.h"
#include "matrix.h"

#include <string.h>
#include <stdio.h>
#include <assert.h>

EMBEDDING* embedding_init(int input_size, int output_size, int max_sequence, float init_scale) {
    EMBEDDING* embedding = malloc(sizeof(EMBEDDING));

    embedding->input_size = input_size;
    embedding->output_size = output_size;
    embedding->max_sequence_length = max_sequence;
    embedding->W = init_array(input_size * output_size, init_scale);
    embedding->dW = malloc(sizeof(float) * input_size * output_size);
    embedding->x = malloc(sizeof(int) * max_sequence);

    PARAM params[] = {
        {"W", embedding->W, embedding->dW, input_size * output_size},
    };

    embedding->num_params = sizeof(params) / sizeof(PARAM);
    embedding->params = malloc(sizeof(params));
    memcpy(embedding->params, params, sizeof(params));

    return embedding;
}

void embedding_init_sequence(EMBEDDING* embedding) {
    embedding->t = 0;

    memset(embedding->dW, 0, sizeof(float) * embedding->input_size * embedding->output_size);
    memset(embedding->x, 0, sizeof(int) * embedding->max_sequence_length);
}

float* embedding_forward(EMBEDDING* embedding, int x_t) {
    assert(x_t < embedding->input_size && x_t >= 0);

    int t = embedding->t++;
    embedding->x[t] = x_t;

    return embedding->W + embedding->output_size * x_t;
}

void embedding_backward(EMBEDDING* embedding, float* delta) {
    int t = --embedding->t;
    int x = embedding->x[t];

    assert(t >= 0 && x >= 0 && x < embedding->input_size);

    float *dW = embedding->dW + embedding->output_size * x;

    for(int i = 0; i < embedding->output_size; i++) {
        dW[i] += delta[i];
    }
}

void embedding_free(EMBEDDING* embedding) {
    free(embedding->W);
    free(embedding->dW);
    free(embedding->x);
    free(embedding->params);
    free(embedding);
}
