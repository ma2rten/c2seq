#include "seq2seq.h"
#include "util.h"
#include "matrix.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>


LSTM* lstm_init(int input_size, int hidden_size, int max_sequence_length, LSTM* previous, float val) {
    LSTM* lstm = malloc(sizeof(LSTM));

    lstm->input_size = input_size;
    lstm->hidden_size = hidden_size;
    lstm->max_sequence_length = max_sequence_length;

    lstm->W_hi = init_array(hidden_size * hidden_size, val);
    lstm->W_hf = init_array(hidden_size * hidden_size, val);
    lstm->W_ho = init_array(hidden_size * hidden_size, val);
    lstm->W_hj = init_array(hidden_size * hidden_size, val);

    lstm->W_xi = init_array(input_size * hidden_size, val);
    lstm->W_xf = init_array(input_size * hidden_size, val);
    lstm->W_xo = init_array(input_size * hidden_size, val);
    lstm->W_xj = init_array(input_size * hidden_size, val);

    lstm->b_i = init_array(hidden_size, val);
    lstm->b_f = init_array(hidden_size, val);
    lstm->b_o = init_array(hidden_size, val);
    lstm->b_j = init_array(hidden_size, val);

    lstm->dW_hi = malloc(sizeof(float) * hidden_size * hidden_size);
    lstm->dW_hf = malloc(sizeof(float) * hidden_size * hidden_size);
    lstm->dW_ho = malloc(sizeof(float) * hidden_size * hidden_size);
    lstm->dW_hj = malloc(sizeof(float) * hidden_size * hidden_size);

    lstm->dW_xi = malloc(sizeof(float) * input_size * hidden_size);
    lstm->dW_xf = malloc(sizeof(float) * input_size * hidden_size);
    lstm->dW_xo = malloc(sizeof(float) * input_size * hidden_size);
    lstm->dW_xj = malloc(sizeof(float) * input_size * hidden_size);

    lstm->db_i = malloc(sizeof(float) * hidden_size);
    lstm->db_f = malloc(sizeof(float) * hidden_size);
    lstm->db_o = malloc(sizeof(float) * hidden_size);
    lstm->db_j = malloc(sizeof(float) * hidden_size);

    lstm->c0 = malloc(sizeof(float) * hidden_size);
    lstm->h0 = malloc(sizeof(float) * hidden_size + 10);

    lstm->dh_prev = malloc(sizeof(float) * hidden_size);
    lstm->dc_prev = malloc(sizeof(float) * hidden_size);

    lstm->dx = malloc(sizeof(float) * input_size * max_sequence_length);
    lstm->x = malloc(sizeof(float) * input_size * max_sequence_length);
    lstm->h = malloc(sizeof(float) * hidden_size * max_sequence_length);
    lstm->c = malloc(sizeof(float) * hidden_size * max_sequence_length);
    lstm->ct = malloc(sizeof(float) * hidden_size * max_sequence_length);

    lstm->input_gate = malloc(sizeof(float) * hidden_size * max_sequence_length);
    lstm->forget_gate = malloc(sizeof(float) * hidden_size * max_sequence_length);
    lstm->output_gate = malloc(sizeof(float) * hidden_size * max_sequence_length);
    lstm->cell_update = malloc(sizeof(float) * hidden_size * max_sequence_length);

    lstm->next = NULL;
    lstm->previous = previous;

    if(previous != NULL)
        previous->next = lstm;

    PARAM params[] = {
        {"W_hi", lstm->W_hi, lstm->dW_hi, hidden_size * hidden_size},
        {"W_hf", lstm->W_hf, lstm->dW_hf, hidden_size * hidden_size},
        {"W_ho", lstm->W_ho, lstm->dW_ho, hidden_size * hidden_size},
        {"W_hj", lstm->W_hj, lstm->dW_hj, hidden_size * hidden_size},
        {"W_xj", lstm->W_xi, lstm->dW_xi, hidden_size * input_size},
        {"W_xf", lstm->W_xf, lstm->dW_xf, hidden_size * input_size},
        {"W_xo", lstm->W_xo, lstm->dW_xo, hidden_size * input_size},
        {"W_xj", lstm->W_xj, lstm->dW_xj, hidden_size * input_size},
        {"b_i", lstm->b_i, lstm->db_i, hidden_size},
        {"b_f", lstm->b_f, lstm->db_f, hidden_size},
        {"b_o", lstm->b_o, lstm->db_o, hidden_size},
        {"b_j", lstm->b_j, lstm->db_j, hidden_size},
    };

    lstm->num_params = sizeof(params) / sizeof(PARAM);
    lstm->params = malloc(sizeof(params));
    memcpy(lstm->params, params, sizeof(params));


    return lstm;
}

void lstm_init_sequence(LSTM* lstm) {
    int i;

    int n = lstm->max_sequence_length;
    int h = lstm->hidden_size;
    int in = lstm->input_size;

    lstm->t = 0;
    lstm->sequence_length = 0;

    memset(lstm->x, 0, sizeof(float) * in * n);
    memset(lstm->h, 0, sizeof(float) * h * n);
    memset(lstm->c, 0, sizeof(float) * h * n);
    memset(lstm->ct, 0, sizeof(float) * h * n);

    memset(lstm->input_gate, 0, sizeof(float) * h * n);
    memset(lstm->forget_gate, 0, sizeof(float) * h * n);
    memset(lstm->output_gate, 0, sizeof(float) * h * n);
    memset(lstm->cell_update, 0, sizeof(float) * h * n);

    memset(lstm->dW_hi, 0, sizeof(float) * h * h);
    memset(lstm->dW_hf, 0, sizeof(float) * h * h);
    memset(lstm->dW_ho, 0, sizeof(float) * h * h);
    memset(lstm->dW_hj, 0, sizeof(float) * h * h);
    memset(lstm->dW_xi, 0, sizeof(float) * in * h);
    memset(lstm->dW_xf, 0, sizeof(float) * in * h);
    memset(lstm->dW_xo, 0, sizeof(float) * in * h);
    memset(lstm->dW_xj, 0, sizeof(float) * in * h);

    memset(lstm->db_i, 0, sizeof(float) * h);
    memset(lstm->db_f, 0, sizeof(float) * h);
    memset(lstm->db_o, 0, sizeof(float) * h);
    memset(lstm->db_j, 0, sizeof(float) * h);

    if(!lstm->previous) {
        memset(lstm->c0, 0, sizeof(float) * h);
        memset(lstm->h0, 0, sizeof(float) * h);
    } else {
        LSTM* prev = lstm->previous;
        memcpy(lstm->c0, prev->c + (prev->t-1) * prev->hidden_size, sizeof(float) * h);
        memcpy(lstm->h0, prev->h + (prev->t-1) * prev->hidden_size, sizeof(float) * h);
    }

    if(!lstm->next) {
        memset(lstm->dh_prev, 0, sizeof(float) * h);
        memset(lstm->dc_prev, 0, sizeof(float) * h);
    } else {
        memcpy(lstm->dh_prev, lstm->next->dh_prev, sizeof(float) * h);
        memcpy(lstm->dc_prev, lstm->next->dc_prev, sizeof(float) * h);
    }
}

float* lstm_forward(LSTM* lstm, float* x_t) {
    float* h_prev;
    float* c_prev;
    int t = lstm->t;

    float* input_gate = lstm->input_gate + lstm->t * lstm->hidden_size;
    float* forget_gate = lstm->forget_gate + lstm->t * lstm->hidden_size;
    float* output_gate = lstm->output_gate + lstm->t * lstm->hidden_size;
    float* cell_update = lstm->cell_update + lstm->t * lstm->hidden_size;
    float* h_t = lstm->h + lstm->t * lstm->hidden_size;
    float* c_t = lstm->c + lstm->t * lstm->hidden_size;
    float* ct_t = lstm->ct + lstm->t * lstm->hidden_size;

    if(lstm->t > 0) {
        c_prev = lstm->c + (lstm->t-1) * lstm->hidden_size;
        h_prev = lstm->h + (lstm->t-1) * lstm->hidden_size;
    } else {
        c_prev = lstm->c0;
        h_prev = lstm->h0;
    }

    memcpy(input_gate, lstm->b_i, sizeof(float) * lstm->hidden_size);
    dot_add(input_gate, lstm->hidden_size, lstm->W_hi, lstm->hidden_size, lstm->hidden_size, h_prev, lstm->hidden_size);
    dot_add(input_gate, lstm->hidden_size, lstm->W_xi, lstm->hidden_size, lstm->input_size, x_t, lstm->input_size);
    a_sigmoid(input_gate, lstm->hidden_size);

    memcpy(forget_gate, lstm->b_f, sizeof(float) * lstm->hidden_size);
    dot_add(forget_gate, lstm->hidden_size, lstm->W_hf, lstm->hidden_size, lstm->hidden_size, h_prev, lstm->hidden_size);
    dot_add(forget_gate, lstm->hidden_size, lstm->W_xf, lstm->hidden_size, lstm->input_size, x_t, lstm->input_size);
    a_sigmoid(forget_gate, lstm->hidden_size);

    memcpy(output_gate, lstm->b_o, sizeof(float) * lstm->hidden_size);
    dot_add(output_gate, lstm->hidden_size, lstm->W_ho, lstm->hidden_size, lstm->hidden_size, h_prev, lstm->hidden_size);
    dot_add(output_gate, lstm->hidden_size, lstm->W_xo, lstm->hidden_size, lstm->input_size, x_t, lstm->input_size);
    a_sigmoid(output_gate, lstm->hidden_size);

    memcpy(cell_update, lstm->b_j, sizeof(float) * lstm->hidden_size);
    dot_add(cell_update, lstm->hidden_size, lstm->W_hj, lstm->hidden_size, lstm->hidden_size, h_prev, lstm->hidden_size);
    dot_add(cell_update, lstm->hidden_size, lstm->W_xj, lstm->hidden_size, lstm->input_size, x_t, lstm->input_size);
    a_tanh(cell_update, lstm->hidden_size);


    for(int i = 0; i < lstm->hidden_size; i++) {
        c_t[i] = input_gate[i] * cell_update[i] + forget_gate[i] * c_prev[i];
        ct_t[i] = tanh(c_t[i]);
        h_t[i] = output_gate[i] * ct_t[i];
    }

    memcpy(lstm->x + lstm->t * lstm->input_size, x_t, sizeof(float) * lstm->input_size);
    lstm->t++;
    lstm->sequence_length++;

    return h_t;
}

float* lstm_backward(LSTM* lstm, float* dh_in) {
    int t = --lstm->t;

    float* h_prev;
    float* c_prev;

    float* dX = lstm->dx + lstm->t * lstm->input_size;
    float* x_t = lstm->x + lstm->t * lstm->input_size;
    float* h_t = lstm->h + lstm->t * lstm->hidden_size;
    float* c_t = lstm->c + lstm->t * lstm->hidden_size;
    float* ct_t = lstm->ct + lstm->t * lstm->hidden_size;
    float* input_gate = lstm->input_gate + lstm->t * lstm->hidden_size;
    float* forget_gate = lstm->forget_gate + lstm->t * lstm->hidden_size;
    float* output_gate = lstm->output_gate + lstm->t * lstm->hidden_size;
    float* cell_update = lstm->cell_update + lstm->t * lstm->hidden_size;

    float dC[lstm->hidden_size];
    float dH[lstm->hidden_size];
    float d_input[lstm->hidden_size];
    float d_forget[lstm->hidden_size];
    float d_output[lstm->hidden_size];
    float d_update[lstm->hidden_size];

    if(lstm->t > 0) {
        c_prev = lstm->c + (lstm->t-1) * lstm->hidden_size;
        h_prev = lstm->h + (lstm->t-1) * lstm->hidden_size;
    } else {
        c_prev = lstm->c0;
        h_prev = lstm->h0;
    }

    if(lstm->t == lstm->sequence_length-1 && lstm->next != NULL) {
        memcpy(lstm->dh_prev, lstm->next->dh_prev, sizeof(float) * lstm->hidden_size);
        memcpy(lstm->dc_prev, lstm->next->dc_prev, sizeof(float) * lstm->hidden_size);
    }

    for(int i = 0; i < lstm->hidden_size; i++) {
        dH[i] = lstm->dh_prev[i];
        if(dh_in != NULL)
            dH[i] += dh_in[i];

        dC[i] = tanh_grad(ct_t[i]) * output_gate[i] * dH[i] + lstm->dc_prev[i];

        d_input[i] = sigmoid_grad(input_gate[i]) * cell_update[i] * dC[i];
        d_forget[i] = sigmoid_grad(forget_gate[i]) * c_prev[i] * dC[i];
        d_output[i] = sigmoid_grad(output_gate[i]) * ct_t[i] * dH[i];
        d_update[i] = tanh_grad(cell_update[i]) * input_gate[i] * dC[i];

        lstm->dc_prev[i] = forget_gate[i] * dC[i];

        lstm->db_i[i] += d_input[i];
        lstm->db_f[i] += d_forget[i];
        lstm->db_o[i] += d_output[i];
        lstm->db_j[i] += d_update[i];
    }

    outer_add(lstm->dW_xi, lstm->hidden_size, lstm->input_size, d_input, lstm->hidden_size, x_t, lstm->input_size);
    outer_add(lstm->dW_xf, lstm->hidden_size, lstm->input_size, d_forget, lstm->hidden_size, x_t, lstm->input_size);
    outer_add(lstm->dW_xo, lstm->hidden_size, lstm->input_size, d_output, lstm->hidden_size, x_t, lstm->input_size);
    outer_add(lstm->dW_xj, lstm->hidden_size, lstm->input_size, d_update, lstm->hidden_size, x_t, lstm->input_size);

    outer_add(lstm->dW_hi, lstm->hidden_size, lstm->hidden_size, d_input, lstm->hidden_size, h_prev, lstm->hidden_size);
    outer_add(lstm->dW_hf, lstm->hidden_size, lstm->hidden_size, d_forget, lstm->hidden_size, h_prev, lstm->hidden_size);
    outer_add(lstm->dW_ho, lstm->hidden_size, lstm->hidden_size, d_output, lstm->hidden_size, h_prev, lstm->hidden_size);
    outer_add(lstm->dW_hj, lstm->hidden_size, lstm->hidden_size, d_update, lstm->hidden_size, h_prev, lstm->hidden_size);

    dot_trans(lstm->dh_prev, lstm->hidden_size, lstm->W_hi, lstm->hidden_size, lstm->hidden_size, d_input, lstm->hidden_size);
    dot_trans_add(lstm->dh_prev, lstm->hidden_size, lstm->W_hf, lstm->hidden_size, lstm->hidden_size, d_forget, lstm->hidden_size);
    dot_trans_add(lstm->dh_prev, lstm->hidden_size, lstm->W_ho, lstm->hidden_size, lstm->hidden_size, d_output, lstm->hidden_size);
    dot_trans_add(lstm->dh_prev, lstm->hidden_size, lstm->W_hj, lstm->hidden_size, lstm->hidden_size, d_update, lstm->hidden_size);

    dot_trans(dX, lstm->input_size, lstm->W_xi, lstm->hidden_size, lstm->input_size, d_input, lstm->hidden_size);
    dot_trans_add(dX, lstm->input_size, lstm->W_xf, lstm->hidden_size, lstm->input_size, d_forget, lstm->hidden_size);
    dot_trans_add(dX, lstm->input_size, lstm->W_xo, lstm->hidden_size, lstm->input_size, d_output, lstm->hidden_size);
    dot_trans_add(dX, lstm->input_size, lstm->W_xj, lstm->hidden_size, lstm->input_size, d_update, lstm->hidden_size);

    return dX;
}

void lstm_free(LSTM* lstm) {
    free(lstm->W_hi); free(lstm->W_hf); free(lstm->W_ho); free(lstm->W_hj);
    free(lstm->W_xi); free(lstm->W_xf); free(lstm->W_xo); free(lstm->W_xj); 
    free(lstm->b_i); free(lstm->b_f); free(lstm->b_o); free(lstm->b_j); 

    free(lstm->dW_hi); free(lstm->dW_hf); free(lstm->dW_ho); free(lstm->dW_hj); 
    free(lstm->dW_xi); free(lstm->dW_xf); free(lstm->dW_xo); free(lstm->dW_xj);
    free(lstm->db_i); free(lstm->db_f); free(lstm->db_o); free(lstm->db_j); 

    free(lstm->c0); free(lstm->h0);
    free(lstm->dh_prev); free(lstm->dc_prev);

    free(lstm->dx); free(lstm->x); free(lstm->h); free(lstm->c); free(lstm->ct);
    free(lstm->input_gate); free(lstm->forget_gate); free(lstm->output_gate); free(lstm->cell_update);
    free(lstm->params);

    free(lstm);
}
