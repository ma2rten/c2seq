
typedef struct {
    int input_size;
    int output_size;
    int max_sequence_length;

    int t;
    int* x;
    float* W;
    float* dW;

    PARAM* params;
    int num_params;

} EMBEDDING;

EMBEDDING* embedding_init(int input_size, int output_size, int max_sequence, float init_scale);
void embedding_init_sequence(EMBEDDING* embedding);
float* embedding_forward(EMBEDDING* embedding, int x_t);
void embedding_backward(EMBEDDING* embedding, float* delta);
float embedding_cost(EMBEDDING* embedding);
void embedding_free(EMBEDDING* embedding);
