
typedef struct {
    int t;
    int size;
    int sequence_length;
    int* targets;
    float* delta;
    float* pred;

} SOFTMAX;

SOFTMAX* softmax_init(int size, int max_sequence, float init_scale);
void softmax_init_sequence(SOFTMAX* softmax);
float* softmax_forward(SOFTMAX* softmax, float* x_t);
float* softmax_backward(SOFTMAX* softmax, int target);
float softmax_cost(SOFTMAX* softmax);
void softmax_free(SOFTMAX* softmax);
