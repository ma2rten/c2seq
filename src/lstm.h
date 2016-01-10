typedef struct lstm LSTM;

struct lstm {
    int input_size;
    int hidden_size;
    int max_sequence_length;
    int sequence_length;

    // parameters
    float *W_hi, *W_hf, *W_ho, *W_hj;
    float *W_xi, *W_xf, *W_xo, *W_xj;
    float *b_i, *b_f, *b_o, *b_j;

    // current state
    int t;
    float* x;
    float* h;
    float* c;
    float* ct;
    float* dx;

    float* c0;
    float* h0;

    float* input_gate;
    float* forget_gate;
    float* output_gate;
    float* cell_update;

    // gradients
    float *dW_hi, *dW_hf, *dW_ho, *dW_hj;
    float *dW_xi, *dW_xf, *dW_xo, *dW_xj;
    float *db_i, *db_f, *db_o, *db_j;

    float *dh_prev;
    float *dc_prev;

    // link to left/right layer
    LSTM* previous;
    LSTM* next;

    // parameter list
    PARAM* params;
    int num_params;
};


LSTM* lstm_init(int input_size, int hidden_size, int max_sequence_length, LSTM* previous, float val);
void lstm_init_sequence(LSTM* lstm);
float* lstm_forward(LSTM* lstm, float* x_t);
float* lstm_backward(LSTM* lstm, float* dh_in);
void lstm_free(LSTM* lstm);
