#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "seq2seq.h"

void show(int* in, int len_in, int* out, int len_out);

int main(int argc, char* argv[])
{
    CONFIG config;
    parse_args(&config, argc, argv);

    FILE* f = fopen(config.train_file, "r");
    if(!f) {
        fprintf(stderr, "File not found: %s\n", config.train_file);
        exit(-1);
    }

    srand(time(NULL) + getpid());

    SEQ2SEQ* seq2seq = seq2seq_init(&config);
    INPUT input;

    for(int epoch = 1; epoch <= config.num_epoch; epoch++) {
        int n = 0;
        float cost = 0.0f;

        fseek(f, 0, SEEK_SET);

        while(read_line(f, &input, config.max_sequence)) {
            int buffer[20];
            int len = seq2seq_generate(seq2seq, input.X, input.X_length, buffer, 20);

            n++;

            cost += seq2seq_train(seq2seq, &input);
            seq2seq_sgd(seq2seq, &config);

            show(input.X, input.X_length, buffer, len);

            free(input.X); free(input.Y);
        }

        printf("epoch %d: train cost %f\n", epoch, cost/n);
    }

    seq2seq_free(seq2seq);
    fclose(f);
}

void show(int* in, int len_in, int* out, int len_out) {
    for(int i = 0; i < len_in; i++) {
        printf("%d ", in[i]);
    }
    printf("-> ");
    for(int i = 0; i < len_out; i++) {
        printf("%d ", out[i]);
    }
    printf("\n");
}
