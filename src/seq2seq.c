#include "seq2seq.h"
#include "matrix.h"

#include <stdlib.h>
#include <string.h>


#define EOS 0

SEQ2SEQ* seq2seq_init(CONFIG* config)
{
    SEQ2SEQ* seq2seq = malloc(sizeof(SEQ2SEQ));
    seq2seq->config = config;

    seq2seq->in_lstms = malloc(sizeof(SEQ2SEQ*) * config->num_hidden_layers);
    seq2seq->out_lstms = malloc(sizeof(SEQ2SEQ*) * config->num_hidden_layers);

    for(int i = 0; i < config->num_hidden_layers; i++) {
        int input_size = i == 0 ? config->embedding_size : config->hidden_size;
        seq2seq->in_lstms[i] = lstm_init(input_size, config->hidden_size, config->max_sequence, NULL, config->init_range);
        seq2seq->out_lstms[i] = lstm_init(input_size, config->hidden_size, config->max_sequence, seq2seq->in_lstms[i], config->init_range);
    }

    seq2seq->in_embedding = embedding_init(config->input_size, config->embedding_size, config->max_sequence, config->init_range);
    seq2seq->out_embedding = embedding_init(config->output_size+1, config->embedding_size, config->max_sequence, config->init_range);
    seq2seq->out_dense = dense_init(config->hidden_size, config->output_size+1, config->max_sequence, config->init_range);
    seq2seq->out_softmax = softmax_init(config->output_size+1, config->max_sequence, config->init_range);

    seq2seq->num_params = 0;
    seq2seq->num_params += seq2seq->in_embedding->num_params;
    seq2seq->num_params += seq2seq->out_embedding->num_params;
    seq2seq->num_params += seq2seq->out_dense->num_params;
    for(int i = 0; i < config->num_hidden_layers; i++) {
        seq2seq->num_params += seq2seq->in_lstms[i]->num_params;
        seq2seq->num_params += seq2seq->out_lstms[i]->num_params;
    }

    PARAM* params = malloc(sizeof(PARAM) * seq2seq->num_params);
    seq2seq->params = params;

    #define add_params(layer) memcpy(params, layer->params, sizeof(PARAM) * layer->num_params); params += layer->num_params;
    add_params(seq2seq->in_embedding);
    add_params(seq2seq->out_embedding);
    add_params(seq2seq->out_dense);
    for(int i = 0; i < config->num_hidden_layers; i++) {
        add_params(seq2seq->in_lstms[i]);
        add_params(seq2seq->out_lstms[i]);
    }

    return seq2seq;
}

int seq2seq_generate(SEQ2SEQ* seq2seq, int* in, int len_in, int* buffer, int max_len)
{
    float *h;

    embedding_init_sequence(seq2seq->in_embedding);
    for(int i = 0; i < seq2seq->config->num_hidden_layers; i++)
        lstm_init_sequence(seq2seq->in_lstms[i]);

    for(int i = 0; i < len_in; i++) {
        h = embedding_forward(seq2seq->in_embedding, in[i]);
        for(int j = 0; j < seq2seq->config->num_hidden_layers; j++)
            h = lstm_forward(seq2seq->in_lstms[j], h);
    }

    embedding_init_sequence(seq2seq->out_embedding);
    for(int i = 0; i < seq2seq->config->num_hidden_layers; i++)
        lstm_init_sequence(seq2seq->out_lstms[i]);
    dense_init_sequence(seq2seq->out_dense);
    softmax_init_sequence(seq2seq->out_softmax);

    int len, token = EOS;

    for(len = 0; len < max_len; len++) {
        h = embedding_forward(seq2seq->out_embedding, token);
        for(int j = 0; j < seq2seq->config->num_hidden_layers; j++)
            h = lstm_forward(seq2seq->out_lstms[j], h);
        h = dense_forward(seq2seq->out_dense, h);
        h = softmax_forward(seq2seq->out_softmax, h);
        token = argmax(h, seq2seq->out_softmax->size);

        if(token == EOS)
            break;

        buffer[len] = token;
    }

    return len;
}

float seq2seq_train(SEQ2SEQ* seq2seq, INPUT* input)
{
    float* h, *dh;

    embedding_init_sequence(seq2seq->in_embedding);
    for(int i = 0; i < seq2seq->config->num_hidden_layers; i++) 
        lstm_init_sequence(seq2seq->in_lstms[i]);

    for(int i = 0; i < input->X_length; i++) {
        h = embedding_forward(seq2seq->in_embedding, input->X[i]);
        for(int j = 0; j < seq2seq->config->num_hidden_layers; j++)
            h = lstm_forward(seq2seq->in_lstms[j], h);
    }

    embedding_init_sequence(seq2seq->out_embedding);
    for(int i = 0; i < seq2seq->config->num_hidden_layers; i++)
        lstm_init_sequence(seq2seq->out_lstms[i]);
    dense_init_sequence(seq2seq->out_dense);
    softmax_init_sequence(seq2seq->out_softmax);

    for(int i = 0; i < input->Y_length + 1; i++) {
        int y;

        if(i == 0)
            y = EOS;
        else
            y = input->Y[i-1] + 1;

        h = embedding_forward(seq2seq->out_embedding, y);
        for(int j = 0; j < seq2seq->config->num_hidden_layers; j++)
            h = lstm_forward(seq2seq->out_lstms[j], h);
        h = dense_forward(seq2seq->out_dense, h);
        h = softmax_forward(seq2seq->out_softmax, h);
    }

    for(int i = input->Y_length; i >= 0; i--) {
        int y;

        if(i == input->Y_length)
            y = EOS;
        else
            y = input->Y[i];

        dh = softmax_backward(seq2seq->out_softmax, y);
        dh = dense_backward(seq2seq->out_dense, dh);
        for(int j = seq2seq->config->num_hidden_layers - 1; j >= 0; j--)
            dh = lstm_backward(seq2seq->out_lstms[j], dh);
        embedding_backward(seq2seq->out_embedding, dh);
    }

    for(int i = input->X_length - 1; i >= 0; i--) {
        dh = NULL;
        for(int j = seq2seq->config->num_hidden_layers - 1; j >= 0; j--)
            dh = lstm_backward(seq2seq->in_lstms[j], dh);
        embedding_backward(seq2seq->in_embedding, dh);
    }

    return softmax_cost(seq2seq->out_softmax);
}

void seq2seq_sgd(SEQ2SEQ* seq2seq, CONFIG* config)
{
    for(int i = 0; i < seq2seq->num_params; i++)
        for(int j = 0; j < seq2seq->params[i].size; j++)
            seq2seq->params[i].param[j] -= config->learning_rate * seq2seq->params[i].grad[j];
}

void seq2seq_free(SEQ2SEQ* seq2seq) {
    for(int i = 0; i < seq2seq->config->num_hidden_layers; i++) {
        lstm_free(seq2seq->in_lstms[i]);
        lstm_free(seq2seq->out_lstms[i]);
    }
    free(seq2seq->in_lstms);
    free(seq2seq->out_lstms);
    free(seq2seq->params);
    embedding_free(seq2seq->in_embedding);
    softmax_free(seq2seq->out_softmax);
    dense_free(seq2seq->out_dense);
    embedding_free(seq2seq->out_embedding);

    free(seq2seq);
}
