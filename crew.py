import argparse
import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, BertTokenizer

import expldf
from data_parser import load_and_cache_examples
from embeddings_cache import EMBS
from models import BERT4SeqClf
from phevals import PostHocAccuracy
from groups import get_text_groups
from crew_impl import CREW
from my_corrclust import cc_weights
from prefix_words import prefix_words_to_feature, get_words_attrs_mask
from utils import set_seed, uniquify, EXEC_TIME_PROFILER, bindump


def crew(args):
    if args.device == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        torch.cuda.empty_cache()

    set_seed(args)

    model_class, tokenizer_class = AutoModelForSequenceClassification, BertTokenizer
    model = model_class.from_pretrained(args.model_dir)
    tokenizer = tokenizer_class.from_pretrained(args.model_dir, do_lower_case=args.do_lower_case)
    model.to(args.device)
    model.eval()

    model = BERT4SeqClf(model, tokenizer, args.device, model.bert)
    EMBS.work_with(model)

    dataset = load_and_cache_examples(args, args.task_name, tokenizer, 'test')

    exp_dir = uniquify(f'{args.output_dir}/{args.model_setup}')
    os.makedirs(exp_dir)
    with open(f'{exp_dir}/args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    ds_sampler = SequentialSampler(dataset)
    ds_dloader = DataLoader(dataset, sampler=ds_sampler, batch_size=eval_batch_size)

    count = -1
    acc = PostHocAccuracy()

    wout = expldf.ExplDataFrame()
    gout = expldf.ExplDataFrame(True)

    EXEC_TIME_PROFILER.reset()

    lime_cache_name = f'lime_cache_{len(dataset)}_{args.lime_n_word_samples}'
    if args.lime_n_word_features > 0:
        lime_cache_name += f'_{args.lime_n_word_features}'
    lime_cache_path = f'./{args.data_dir}/{lime_cache_name}'
    if os.path.exists(lime_cache_path):
        lime_cache = pickle.load(open(lime_cache_path, 'rb'))
        lime_cached = True
    else:
        lime_cache = None
        lime_cached = False

    pred_probs_ls = []

    crew_ = CREW(args, model, tokenizer, lime_cache)
    if lime_cache is None:
        lime_cache = [[], [], []]

    for batch in tqdm(ds_dloader, desc='Evaluating'):
        EXEC_TIME_PROFILER.start()

        set_seed(args)
        count += 1

        batch = tuple(t.to(args.device) for t in batch)

        words, segments_ids, attrs_mask, prefix_words_a, prefix_words_b = get_words_attrs_mask(
            batch[0][0], batch[2][0], batch[4][0], tokenizer, args.wordpieced)

        prefix_words_ = prefix_words_a + prefix_words_b

        # BERT classification
        input_ids_ = batch[0]
        attention_mask_ = batch[1]
        segments_ids_ = batch[2]

        logits = model.predict(None, input_ids_, attention_mask_, segments_ids_, args.wordpieced)
        logits = logits.detach().cpu().numpy()[0]
        y_true = logits.argmax()

        idxs, word_scores, groups, group_scores = crew_.explain(
            prefix_words_, attrs_mask, segments_ids, count)

        if len(idxs) == len(prefix_words_):
            pred_probs = logits
        else:
            top_prefix_words = [prefix_words_[i] for i in idxs]
            f = prefix_words_to_feature(top_prefix_words, segments_ids, tokenizer, args.wordpieced, args.max_seq_length)
            input_ids_ = torch.tensor([f.input_ids], device=args.device)
            attention_mask_ = torch.tensor([f.input_mask], device=args.device)
            segment_ids_ = torch.tensor([f.segment_ids], device=args.device)
            logits_ = model.predict(None, input_ids_, attention_mask_, segment_ids_, args.wordpieced)
            pred_probs = logits_.detach().cpu().numpy()

        pred_probs = pred_probs.tolist()
        pred_probs_ls.append(pred_probs)
        y_pred = np.argmax(pred_probs)

        if not lime_cached:
            lime_cache[0].append([prefix_words_[i] for i in idxs])
            lime_cache[1].append(word_scores
                                 # .tolist()
                                 )
            lime_cache[2].append(pred_probs)

        acc.append(y_true, y_pred)

        wout.add_rows(count,
                      impact=word_scores,
                      word=[words[i] for i in idxs],
                      wid_word=[prefix_words_[i] for i in idxs],
                      column=[int(attrs_mask[i]) for i in idxs],
                      segment=[int(segments_ids[i]) for i in idxs],
                      wid=idxs
                      )

        gout.add_rows(count,
                      group=get_text_groups({i: g for i, g in enumerate(groups)}, prefix_words_),
                      impact=group_scores,
                      wids=groups,
                      )

        EXEC_TIME_PROFILER.timestep('stuff')

    bindump(wout.get_df(), f'{exp_dir}/wexpls')
    bindump(gout.get_df(), f'{exp_dir}/gexpls')
    bindump((str(os.uname()), EXEC_TIME_PROFILER.get_list()), f'{exp_dir}/exec_time_profile.pkl')
    bindump(pred_probs_ls, f'{exp_dir}/pred_probs')

    ret = {
        'Post-hoc accuracy': acc.get_score(),
    }
    for k, v in ret.items():
        print(k, v)
    bindump(ret, f'{exp_dir}/return.pkl')

    if not lime_cached:
        bindump(lime_cache, lime_cache_path)

    print('All done!')


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

    crew(args)
