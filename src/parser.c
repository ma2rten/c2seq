#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "parser.h"

#define UNKNOWN -1
#define END_PAIR -2
#define END_SEQUENCE -3
#define END_NUM -4

/*
batch_size = 128
larning_rate = 0.7
init_scale = -0.08 ... 0.08
gradent clipped to 5
learning rate is halved every 0.5 epoch after 5 epoch
*/

int parse_char(char c) {
    switch (c) {
        case '0': return 0;
        case '1': return 1;
        case '2': return 2;
        case '3': return 3;
        case '4': return 4;
        case '5': return 5;
        case '6': return 6;
        case '7': return 7;
        case '8': return 8;
        case '9': return 9;
        case '\n': return END_PAIR;
        case EOF: return END_PAIR;
        case ';': return END_SEQUENCE;
        case ' ': return END_NUM;
    }

    return UNKNOWN;
}

int read_num(FILE* f, int* p_value) {
    int digit;
    int value = 0;
    bool is_first = true;

    while(1) {
        digit = parse_char(fgetc(f));

        if(is_first) {
            // ignore extra spaces
            if(digit == END_NUM)
                continue;

            // case of only spaces before line ending
            if(digit < 0)
                break;

            is_first = false;
        }

        // stop when we encountered a non-digit
        if(digit < 0)
            break;

        // add digit to final value
        value = 10 * value + digit;
    }

    *p_value = !is_first ? value : -1;

    return digit;
}

int read_sequence(FILE* f, int** seq, int* len, int max_sequence, int EOS) {
    int i;
    int code = END_NUM;
    int X[max_sequence];

    for(i = 0; code == END_NUM; i++) {
        code = read_num(f, X + i);

        if(i == max_sequence) {
            fprintf(stderr, "ERROR: input too long\n");
            exit(-1);
        }
    }

    if(X[i-1] == -1) {
        i--;
    }

    if(code == END_PAIR && i == 0) {
        return 1;
    }

    *len = i;
    *seq = malloc(sizeof(int) * i);
    memcpy(*seq, X, sizeof(int) * i);

    if(code != EOS) {
        fprintf(stderr, "ERROR: invalid file format\n");
        exit(-1);
    }

    return 0;
}

int read_line (FILE* f, INPUT* input, int max_sequence) {
    int code = read_sequence(f, &input->X, &input->X_length, max_sequence, END_SEQUENCE);

    if(code)
        return 0;

    read_sequence(f, &input->Y, &input->Y_length, max_sequence, END_PAIR);

    return 1;
}
