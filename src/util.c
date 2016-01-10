#include <math.h>
#include "util.h"

float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

float sigmoid_grad(float x) {
    return x * (1.0f - x);
}

float tanh_grad(float x) {
    return 1 - x*x;
}

void a_sigmoid(float* X, int n) {
    for(int i = 0; i < n; i++) {
        X[i] = sigmoid(X[i]);
    }
}

void a_tanh(float* X, int n) {
    for(int i = 0; i < n; i++) {
        X[i] = tanh(X[i]);
    }
}

float rand_range(float min, float max) {
    float val = (float)rand() / RAND_MAX;
    return (val * (max - min)) + min;
}

float* init_array(size_t n, float range) {
    float *W = malloc(sizeof(float) * n);

    for(int i = 0; i < n; i++) {
        float val = rand_range(-range, range);
        W[i] = val;
    }

    return W;
}
