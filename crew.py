import argparse

import explain
from my_corrclust import cc_weights

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str,
                        help='The directory with the input data.')
    parser.add_argument('model_dir', type=str,
                        help='The directory with the model')
    parser.add_argument('output_dir', type=str,
                        help='The directory in which the explanations and other additional outputs will be saved')

    parser.add_argument('--semantic', type=bool, choices=[True, False], default=True,
                        help='Include semantic relatedness')
    parser.add_argument('--importance', type=bool, choices=[True, False], default=True,
                        help='Include importance relatedness')
    parser.add_argument('--schema', type=bool, choices=[True, False], default=True,
                        help='Include schema relatedness')

    parser.add_argument("--wordpieced", type=bool, choices=[True, False], default=False,
                        help='True: tokenize the output, False: preserve original words')

    parser.add_argument("--max_seq_length", default=100, type=int, metavar='MAX',
                        help="The maximum total length of the tokenized input sequence. Sequences longer than this "
                             "will be truncated, and sequences shorter will be padded.")

    parser.add_argument('--lime_n_word_samples', type=int, default=1000, metavar='WS',
                        help='LIME: number of perturbed samples to learn a word-level explanation')
    parser.add_argument('--lime_n_group_samples', type=int, default=200, metavar='GS',
                        help='LIME: number of perturbed samples to learn a group-level explanation')
    parser.add_argument('--lime_n_word_features', type=int, default=70, metavar='WF',
                        help='LIME: maximum number of features present in explanation')

    parser.add_argument('--gpu', default=True, choices=[True, False], type=bool,
                        help='True: GPU, False: CPU')

    parser.add_argument('-seed', '--seed', type=int, default=42,
                        help='seed')

    parser.add_argument("-f", "--file", required=False)  # colab

    args = parser.parse_args()

    args.task_name = 'lrds'
    args.model_type = 'bert'
    args.model_name_or_path = 'bert-base-uncased'
    args.do_lower_case = True
    args.per_gpu_eval_batch_size = 1

    args.words_scorer = 'lime'
    args.group_scorer = 'corrclust'
    args.model_setup = f'crew{"_wordpieced" if args.wordpieced else ""}'
    args.cc_weights = cc_weights(
        1. if args.semantic else 0.,
        1. if args.importance else 0.,
        1. if args.schema else 0.,
    )
    args.cc_emb_sim_bias = 'mean'
    args.cc_alg = 'demaine'
    args.cc_schema_scores = [-0.5, 0.]
    args.device = 'cuda' if args.gpu else 'cpu'
    args.n_gpu = 1

    explain.main(args)
