#include "seq2seq.h"
#include "util.h"
#include "matrix.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

SOFTMAX* softmax_init(int size, int max_sequence, float init_scale) {
    SOFTMAX* softmax = malloc(sizeof(SOFTMAX));

    softmax->size = size;
    softmax->targets = malloc(sizeof(int) * max_sequence);
    softmax->pred = malloc(sizeof(float) * size * max_sequence);
    softmax->delta = malloc(sizeof(float) * size * max_sequence);

    return softmax;
}

void softmax_init_sequence(SOFTMAX* softmax) {
    softmax->t = 0;
    softmax->sequence_length = 0;
}

float* softmax_forward(SOFTMAX* softmax, float* x_t) {
    float *y = softmax->pred + softmax->t * softmax->size;

    memcpy(y, x_t, sizeof(float) * softmax->size);

    float max_val = y[0];
    for(int i = 1; i < softmax->size; i++)
        if(max_val < y[i])
            max_val = y[i];

    float total = 0.0f;
    for(int i = 0; i < softmax->size; i++) {
        y[i] = exp(y[i] - max_val);
        total += y[i];
    }

    for(int i = 0; i < softmax->size; i++)
        y[i] /= total;

    softmax->t++;
    softmax->sequence_length++;

    return y;
}

float* softmax_backward(SOFTMAX* softmax, int target) {
    int t = --softmax->t;
    float* pred = softmax->pred + t * softmax->size;
    float* delta = softmax->delta + t * softmax->size;

    softmax->targets[t] = target;

    memcpy(delta, pred, sizeof(float) * softmax->size);
    delta[target] -= 1.0f;

    return delta;
}

float softmax_cost(SOFTMAX* softmax) {
    float cost = 0.0f;

    for(int t = 0; t < softmax->sequence_length; t++) {
        int target = softmax->targets[t];
        cost += -log(softmax->pred[target + t * softmax->size]);
    }

    return cost;
}

void softmax_free(SOFTMAX* softmax) {
    free(softmax->targets);
    free(softmax->pred);
    free(softmax->delta);
    free(softmax);
}
