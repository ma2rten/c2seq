#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>

#include "args.h"

const char usage [] = "Usage: seq2seq train_file";

void parse_args(CONFIG* config, int argc, char* argv[])
{
    // default options
    config->init_range = 0.08f; //r
    config->learning_rate = 0.7f; //l
    config->num_epoch = 3; //e
    config->input_size = 10; //i
    config->output_size = 10; //o
    config->embedding_size = 10; //b
    config->hidden_size = 10; //h
    config->num_hidden_layers = 1; //n
    config->max_sequence = 100; //,

    static struct option long_options[] = {
        {"init_range", required_argument, 0, 'r'},
        {"learning_rate", required_argument, 0, 'l'},
        {"num_epoch", required_argument, 0, 'e'},
        {"input_size", required_argument, 0, 'i'},
        {"output_size", required_argument, 0, 'o'},
        {"embedding_size", required_argument, 0, 'b'},
        {"hidden_size", required_argument, 0, 'h'},
        {"num_hidden_layers", required_argument, 0, 'n'},
        {"max_sequence", required_argument, 0, 'm'},
        {0, 0, 0, 0}
    };

    // parse commandline
    while (1) {
        int opt = getopt_long(argc, argv, "r:l:e:i:o:b:h:n:m:", long_options, NULL);

        if(opt == -1)
            break;

        switch (opt) {
            case 'r':
                config->init_range = atof(optarg);
                break;
            case 'l':
                config->learning_rate = atof(optarg);
                break;
            case 'e':
                config->num_epoch = atoi(optarg);
                break;
            case 'i':
                config->input_size = atoi(optarg);
                break;
            case 'o':
                config->output_size = atoi(optarg);
                break;
            case 'b':
                config->embedding_size = atoi(optarg);
                break;
            case 'h':
                config->hidden_size = atoi(optarg);
                break;
            case 'n':
                config->num_hidden_layers = atoi(optarg);
                break;
            case 'm':
                config->max_sequence = atoi(optarg);
                break;
            default:
                fprintf(stderr, usage);
                exit(-1);
        }
    }

    if (optind >= argc) {
        fprintf(stderr, "Invalid Commandline\n");
        fprintf(stderr, usage);
        exit(-1);
    }

    config->train_file = argv[optind];

    // check if values are valid
    if(config->init_range <= 0.0f ||
            config->init_range <= 0.0f ||
            config->num_epoch <= 0 ||
            config->input_size <= 0 ||
            config->output_size <= 0 ||
            config->embedding_size <= 0 ||
            config->hidden_size <= 0 ||
            config->num_hidden_layers <= 0 ||
            config->max_sequence <= 0)
    {
        fprintf(stderr, "Invalid Commandline Value\n");
        fprintf(stderr, usage);
        //exit(-1);
    }
}

void print_config(CONFIG* config)
{
    printf("train_file: %s\n", config->train_file);
    printf("init_range: %f\n", config->init_range);
    printf("learning_rate: %f\n", config->learning_rate);
    printf("num_epoch: %d\n", config->num_epoch);
    printf("input_size: %d\n", config->input_size);
    printf("embedding_size: %d\n", config->embedding_size);
    printf("hidden_size: %d\n", config->hidden_size);
    printf("output_size: %d\n", config->output_size);
    printf("num_hidden_layers: %d\n", config->num_hidden_layers);
    printf("max_sequence: %d\n", config->num_hidden_layers);
}
