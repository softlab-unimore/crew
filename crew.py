import argparse

import explain
from my_corrclust import cc_weights

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str)
    parser.add_argument('model_dir', type=str)
    parser.add_argument('output_dir', type=str)

    parser.add_argument('--emb_sim', type=bool, default=True)
    parser.add_argument('--impacts', type=bool, default=True)
    parser.add_argument('--schema', type=bool, default=True)

    parser.add_argument("--wordpieced", type=bool, default=False)

    parser.add_argument("--max_seq_length", default=100, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument('--lime_n_word_samples', type=int, default=1000)
    parser.add_argument('--lime_n_group_samples', type=int, default=200)
    parser.add_argument('--lime_n_word_features', type=int, default=70)

    parser.add_argument('--gpu', default=True, type=bool, help='True:gpu, False:cpu')

    parser.add_argument('-seed', '--seed', type=int, default=42, help='seed')

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
        1. if args.emb_sim else 0.,
        1. if args.impacts else 0.,
        1. if args.schema else 0.,
    )
    args.cc_emb_sim_bias = 'mean'
    args.cc_alg = 'demaine'
    args.cc_schema_scores = [-0.5, 0.]
    args.device = 'cuda' if args.gpu else 'cpu'
    args.n_gpu = 1

    explain.main(args)
