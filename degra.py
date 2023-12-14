import argparse
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizer

from utils import bindump
from embeddings_cache import BERT4SeqClf, EMBS
from eval_measures import CReWDegradPredict, CReWDegradPredictGroups, DegradationScoreF1


def degra(exp_path, model_path, device='cuda', do_lower_case=True):
    with open(exp_path + '/wexpls', 'rb') as f:
        wexpls = pickle.load(f)

    with open(exp_path + '/gexpls', 'rb') as f:
        gexpls = pickle.load(f)

    model_class, tokenizer_class = AutoModelForSequenceClassification, BertTokenizer
    model = model_class.from_pretrained(model_path)
    tokenizer = tokenizer_class.from_pretrained(model_path, do_lower_case=do_lower_case)
    model.to(device)

    model.eval()
    model = BERT4SeqClf(model, tokenizer, device, model.bert)
    EMBS.work_with(model)

    w_degrad_score, g_degrad_score = DegradationScoreF1(.10), DegradationScoreF1(.10)

    for inx in tqdm(wexpls['data_inx'].unique()):
        wexpl = wexpls[wexpls['data_inx'] == inx]
        gexpl = gexpls[gexpls['data_inx'] == inx]

        l_words_dt, r_words_dt, segments_dt = dict(), dict(), dict()
        wids_desc_imp = []

        for row in wexpl.itertuples():
            row = row._asdict()
            wids_desc_imp.append((row['wid'], row['impact']))
            if row['segment'] == 0:
                l_words_dt[row['wid']] = row['word']
            else:
                r_words_dt[row['wid']] = row['word']
            segments_dt[row['wid']] = row['segment']

        newkeys = dict()
        for i, k in enumerate(sorted(l_words_dt.keys())):
            # newkeys[k] = i + 1
            newkeys[k] = i
        for i, k in enumerate(sorted(r_words_dt.keys())):
            # newkeys[k] = i + 1 + len(l_words_dt) + 1
            newkeys[k] = i + len(l_words_dt)

        wids_desc_imp = [wid for wid, _ in sorted(wids_desc_imp, key=lambda x: x[1], reverse=True)]
        wids_desc_imp = [newkeys[old] for old in wids_desc_imp]

        l_words = [l_words_dt[k] for k in sorted(l_words_dt.keys())]
        r_words = [r_words_dt[k] for k in sorted(r_words_dt.keys())]
        # words = np.array(['[CLS]'] + l_words + ['[SEP]'] + r_words + ['[SEP]'])
        # segments = np.array([0] + [0] * len(l_words) + [0] + [1] * len(r_words) + [1])
        words = np.array(l_words + r_words)
        segments = np.array([0] * len(l_words) + [1] * len(r_words))

        degrad_predict = CReWDegradPredict(model, device, wids_desc_imp, words, None, segments, False, tokenizer)

        w_degrad_score.append(degrad_predict)

        wids_desc_imp = []

        for row in gexpl.itertuples():
            row = row._asdict()
            wids_desc_imp.append((set(newkeys[old] for old in row['wids']), row['impact']))

        wids_desc_imp = np.array([gr for gr, _ in sorted(wids_desc_imp, key=lambda x: x[1], reverse=True)])

        degrad_predict = CReWDegradPredictGroups(model, device, wids_desc_imp, words, None, segments, False, tokenizer)

        g_degrad_score.append(degrad_predict)

    return w_degrad_score, g_degrad_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_path', type=str, default='./output/beer/crew_00',
                        help='The directory with the explanation.')
    parser.add_argument('--model_path', type=str, default='./../GMASK/BERT_GMASK/model/beer',
                        help='The directory with the model')

    parser.add_argument('--gpu', default=True, choices=[True, False], type=bool,
                        help='True: GPU, False: CPU')

    parser.add_argument("-f", "--file", required=False)  # colab

    args = parser.parse_args()

    args.do_lower_case = True
    args.device = 'cuda' if args.gpu else 'cpu'

    if args.device == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        torch.cuda.empty_cache()

    w_degra, g_degra = degra(args.exp_path, args.model_path, args.device, args.do_lower_case)

    res = {
        'degrad_score': g_degra.get_score(),
        'degrad_steps': g_degra.get_degrad_steps(),
        'lerf_f1': g_degra.get_lerf_f1().tolist(),
        'morf_f1': g_degra.get_morf_f1().tolist(),
    }
    print(res)
    bindump(res, f'{args.exp_path}/degrad_score.pkl')
