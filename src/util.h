#include <stdlib.h>

float sigmoid(float x);
float sigmoid_grad(float x);
float tanh_grad(float x);
void a_sigmoid(float* X, int n);
void a_tanh(float* X, int n);
float rand_range(float min, float max);
float* init_array(size_t n, float range);
