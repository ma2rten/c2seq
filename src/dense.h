
typedef struct {
    int t;
    int input_size;
    int output_size;
    int sequence_length;

    float* x;
    float* y;
    float* dh;

    float* W;
    float* dW;

    PARAM* params;
    int num_params;

} DENSE;

DENSE* dense_init(int input_size, int output_size, int max_sequence, float init_scale);
void dense_init_sequence(DENSE* dense);
float* dense_forward(DENSE* dense, float* xt);
float* dense_backward(DENSE* dense, float* delta);
void dense_free(DENSE* dense);
