#include "seq2seq.h"
#include "util.h"
#include "matrix.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <stdbool.h>


int main()
{
    const float DELTA = 1e-5f;
    const float THRESHOLD = 1e-2;

    bool passed = true;

    srand(time(NULL));

    CONFIG config = {0};

    config.input_size = 4;
    config.embedding_size = 3;
    config.hidden_size = 7;
    config.output_size = 5;
    config.num_hidden_layers = 2;
    config.max_sequence = 100;
    config.init_range = 0.8f;

    SEQ2SEQ* seq2seq = seq2seq_init(&config);

    int X[] = {1,2,3};
    int Y[] = {3,1};

    INPUT input = {X, Y, sizeof(X) / sizeof(int), sizeof(Y) / sizeof(int)};

    for(int i = 0; i < seq2seq->num_params; i++) {
        for(int j = 0; j < seq2seq->params[i].size; j++) {
            float cost1, cost2;
            float val = seq2seq->params[i].param[j];

            seq2seq->params[i].param[j] = val + DELTA;
            cost1 = seq2seq_train(seq2seq, &input);

            seq2seq->params[i].param[j] = val - DELTA;
            cost2 = seq2seq_train(seq2seq, &input);

            seq2seq->params[i].param[j] = val;
            seq2seq_train(seq2seq, &input);

            float grad_num = (cost1 - cost2) / (2 * DELTA);
            float grad_analytic = seq2seq->params[i].grad[j];
            float error = fabs(grad_num - grad_analytic);

            if (error > THRESHOLD) {
                printf("ERROR %s[%d] %f %f\n", seq2seq->params[i].name, j, grad_num, grad_analytic);
                passed = false;
                break;
            } else {
                //printf("PASS %s[%d] %f %f\n", params[i].name, j, grad_num, grad_analytic);
            }
        }
    }

    if(passed) {
        printf("ALL PASSED\n");
    }

    seq2seq_free(seq2seq);

    return 1;
}
