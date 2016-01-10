typedef struct {
    char* train_file;
    float init_range;
    float learning_rate;
    int num_epoch;

    int input_size;
    int embedding_size;
    int hidden_size;
    int output_size;
    int num_hidden_layers;
    int max_sequence;
}
CONFIG;

void parse_args(CONFIG* config, int argc, char* argv[]);
