#include "seq2seq.h"
#include "util.h"
#include "matrix.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

DENSE* dense_init(int input_size, int output_size, int max_sequence, float init_scale)
{
    DENSE* dense = malloc(sizeof(DENSE));

    dense->input_size = input_size;
    dense->output_size = output_size;

    dense->W = init_array(output_size * input_size, init_scale);
    dense->dW = malloc(sizeof(float) * output_size * input_size);
    dense->x = malloc(sizeof(float) * input_size * max_sequence);
    dense->y = malloc(sizeof(float) * output_size * max_sequence);
    dense->dh = malloc(sizeof(float) * input_size);

    PARAM params[] = {
        {"W", dense->W, dense->dW, output_size * input_size},
    };

    dense->num_params = sizeof(params) / sizeof(PARAM);
    dense->params = malloc(sizeof(params));
    memcpy(dense->params, params, sizeof(params));

    return dense;
}

void dense_init_sequence(DENSE* dense)
{
    dense->t = 0;
    dense->sequence_length = 0;

    memset(dense->dW, 0, sizeof(float) * dense->output_size * dense->input_size);
}

float* dense_forward(DENSE* dense, float* xt)
{
    float* y = dense->y + dense->t * dense->output_size;

    dot(y, dense->output_size, dense->W, dense->output_size, dense->input_size, xt, dense->input_size);
    memcpy(dense->x + dense->t * dense->input_size, xt, sizeof(float) * dense->input_size);

    dense->t++;
    dense->sequence_length++;

    return y;
}

float* dense_backward(DENSE* dense, float* din)
{
    int t = --dense->t;
    float* xt = dense->x + t * dense->input_size;

    outer_add(dense->dW, dense->output_size, dense->input_size, din, dense->output_size, xt, dense->input_size);
    dot_trans(dense->dh, dense->input_size, dense->W, dense->output_size, dense->input_size, din, dense->output_size);

    return dense->dh;
}

void dense_free(DENSE* dense)
{
    free(dense->W);
    free(dense->dW);
    free(dense->x);
    free(dense->y);
    free(dense->dh);
    free(dense->params);
    free(dense);
}
