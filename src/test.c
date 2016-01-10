#include <stdio.h>

#include "softmax.h"
#include "lstm.h"
#include "matrix.h"
#include "util.h"

void init_values(SOFTMAX* softmax, LSTM* lstm);

int main() {
    const int input_size = 3;
    const int hidden_size = 2;
    const int output_size = 4;

    LSTM* lstm = lstm_init(input_size, hidden_size, 100, NULL, 0.1);
    SOFTMAX* softmax = softmax_init(hidden_size, output_size, 100, 0.1);

    init_values(softmax, lstm);
    lstm_init_sequence(lstm);
    softmax_init_sequence(softmax);

    float X1 [] = {0.120158952482, 0.617203109707, 0.300170319956};
    float X2 [] = {-0.352249846494, -1.14251819802, -0.349342722413};

    float* h = lstm_forward(lstm, X1);
    softmax_forward(softmax, h);

    h = lstm_forward(lstm, X2);
    softmax_forward(softmax, h);

    float* d = softmax_backward(softmax, 1);
    lstm_backward(lstm, d);

    d = softmax_backward(softmax, 2);
    lstm_backward(lstm, d);

    printf("cost: %f\n", softmax_cost(softmax));

    return 0;
}

void init_values(SOFTMAX* softmax, LSTM* lstm) {
    lstm->W_hi[0] = 1.148586;
    lstm->W_hi[1] = -0.432577;
    lstm->W_hi[2] = -0.373474;
    lstm->W_hi[3] = -0.758703;
    lstm->W_hf[0] = 0.611936;
    lstm->W_hf[1] = -1.627434;
    lstm->W_hf[2] = 1.233768;
    lstm->W_hf[3] = -0.538255;
    lstm->W_ho[0] = 0.225595;
    lstm->W_ho[1] = -0.176331;
    lstm->W_ho[2] = 1.033866;
    lstm->W_ho[3] = -1.456739;
    lstm->W_hj[0] = -0.227983;
    lstm->W_hj[1] = -0.271567;
    lstm->W_hj[2] = 0.801696;
    lstm->W_hj[3] = -0.777741;
    lstm->W_xi[0] = -0.099551;
    lstm->W_xi[1] = -0.506832;
    lstm->W_xi[2] = 0.024372;
    lstm->W_xi[3] = 0.336489;
    lstm->W_xi[4] = -0.635443;
    lstm->W_xi[5] = 0.660907;
    lstm->W_xf[0] = 0.520534;
    lstm->W_xf[1] = 0.290115;
    lstm->W_xf[2] = 0.520109;
    lstm->W_xf[3] = -0.394750;
    lstm->W_xf[4] = -0.070951;
    lstm->W_xf[5] = -0.540267;
    lstm->W_xo[0] = -0.154665;
    lstm->W_xo[1] = 0.306201;
    lstm->W_xo[2] = -0.399331;
    lstm->W_xo[3] = -0.229066;
    lstm->W_xo[4] = -0.396739;
    lstm->W_xo[5] = -0.487980;
    lstm->W_xj[0] = -0.387544;
    lstm->W_xj[1] = -0.007312;
    lstm->W_xj[2] = -0.645079;
    lstm->W_xj[3] = 0.135340;
    lstm->W_xj[4] = 0.958287;
    lstm->W_xj[5] = 0.428419;
    lstm->b_i[0] = 0.000000;
    lstm->b_i[1] = 0.000000;
    lstm->b_f[0] = 3.000000;
    lstm->b_f[1] = 3.000000;
    lstm->b_o[0] = 0.000000;
    lstm->b_o[1] = 0.000000;
    lstm->b_j[0] = 0.000000;
    lstm->b_j[1] = 0.000000;
    softmax->W[0] = -0.135648;
    softmax->W[1] = -0.627648;
    softmax->W[2] = -0.528321;
    softmax->W[3] = 1.196746;
    softmax->W[4] = 0.035927;
    softmax->W[5] = -0.450424;
    softmax->W[6] = 0.134998;
    softmax->W[7] = 1.485105;
}
