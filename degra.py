import argparse
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizer

from crew_degra import CReWDegradPredict, CReWDegradPredictGroups
from embeddings_cache import EMBS
from models import BERT4SeqClf
from phevals import DegradationScoreF1
from prefix_words import get_idx, get_word
from utils import bindump


def degra(exp_path: str, model_path: str, degra_step: float, device='cuda', do_lower_case=True):
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

    w_degrad_score, g_degrad_score = DegradationScoreF1(degra_step), DegradationScoreF1(degra_step)

    for inx in tqdm(wexpls['data_inx'].unique()):
        wexpl = wexpls[wexpls['data_inx'] == inx]
        gexpl = gexpls[gexpls['data_inx'] == inx]

        words, impacts, segments = dict(), dict(), dict()
        keys = list()
        for wid_word, i, s in zip(wexpl['wid_word'].values, wexpl['impact'].values, wexpl['segment'].values):
            wid, word = get_idx(wid_word), get_word(wid_word)
            words[wid] = word
            impacts[wid] = i
            segments[wid] = s
            keys.append(wid)
        keys = {old: new for new, old in enumerate(sorted(keys))}
        words = np.array([words[k] for k in keys])
        impacts = np.array([impacts[k] for k in keys])
        segments = np.array([segments[k] for k in keys])

        degrad_predict = CReWDegradPredict(words, impacts, None, segments, model, device, False, tokenizer)

        w_degrad_score.append(degrad_predict)

        groups, impacts = [], []
        for row in gexpl.itertuples():
            row = row._asdict()
            groups.append(set(keys[old] for old in row['wids']))
            impacts.append(row['impact'])
        groups, impacts = np.array(groups), np.array(impacts)

        degrad_predict = CReWDegradPredictGroups(words, groups, impacts, None, segments,
                                                 model, device, False, tokenizer)

        g_degrad_score.append(degrad_predict)

    return w_degrad_score, g_degrad_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('exp_path', type=str, default='./output/beer/crew_shap_00_avg',
                        help='The directory with the explanations.')
    parser.add_argument('model_path', type=str, default='./../GMASK/BERT_GMASK/model/beer',
                        help='The directory with the model.')
    parser.add_argument('--degra_step', type=float, default=0.1,
                        help='The percentage step between two degradation levels. 100% divided by this step should'
                             'be an integer number, that is the number of degradation levels.')
    parser.add_argument('--gpu', default=True, choices=[True, False], type=bool,
                        help='True: GPU, False: CPU.')

    parser.add_argument("-f", "--file", required=False)  # colab

    args = parser.parse_args()

    args.do_lower_case = True
    args.device = 'cuda' if args.gpu else 'cpu'

    if args.device == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        torch.cuda.empty_cache()

    w_degra, g_degra = degra(args.exp_path, args.model_path, args.degra_step, args.device, args.do_lower_case)

    res = {
        'degrad_score': g_degra.get_score(),
        'degrad_steps': g_degra.get_degrad_steps(),
        'lerf_f1': g_degra.get_lerf_f1().tolist(),
        'morf_f1': g_degra.get_morf_f1().tolist(),
    }
    print(res['degrad_score'])
    bindump(res, f'{args.exp_path}/degrad_score')
