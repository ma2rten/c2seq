#include <stdio.h>

typedef struct {
    int* X;
    int* Y;
    int X_length;
    int Y_length;
} INPUT;

int read_line (FILE* f, INPUT* input, int max_sequence);
