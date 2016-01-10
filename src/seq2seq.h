#include "params.h"
#include "lstm.h"
#include "dense.h"
#include "softmax.h"
#include "embedding.h"
#include "parser.h"
#include "args.h"

typedef struct {
    EMBEDDING* in_embedding;
    LSTM** in_lstms;

    EMBEDDING* out_embedding;
    LSTM** out_lstms;
    DENSE* out_dense;
    SOFTMAX* out_softmax;
    CONFIG* config;

    PARAM* params;
    int num_params;

} SEQ2SEQ;


SEQ2SEQ* seq2seq_init(CONFIG* config);
int seq2seq_generate (SEQ2SEQ* seq2seq, int* in, int len_in, int* buffer, int max_len);
float seq2seq_train(SEQ2SEQ* seq2seq, INPUT* input);
void seq2seq_sgd (SEQ2SEQ* seq2seq, CONFIG* config);
void seq2seq_free (SEQ2SEQ* seq2seq);
