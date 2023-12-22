import re

import numpy as np

from pyccalg import cc
from models import ModelWrapper
from groups import get_text_groups
from interpret_lime import MyLIMETextExplainer
from my_corrclust import get_cc_scores_table
from prefix_words import get_idx, get_word
from utils import EXEC_TIME_PROFILER
from wordcorr import EmbdngWordCorr, ImpactWordCorr, SchemaWordCorr


def groups_same_word_v1(words):
    word_ixs = {_w: [] for _w in words}
    for _idx, _w in enumerate(words):
        word_ixs[_w].append(_idx)
    if '[CLS]' in word_ixs.keys(): del word_ixs['[CLS]']
    if '[SEP]' in word_ixs.keys(): del word_ixs['[SEP]']
    groups = []
    for _k, _v in word_ixs.items():
        groups.append(_v)
    return groups


def groups_same_word_v2(words, mask):
    segment_word_ixs = {s: {} for s in np.unique(mask)}
    for _w, _s in zip(words, mask):
        if _w == '[SEP]':
            continue
        if _w == '[CLS]':
            continue
        segment_word_ixs[_s][_w] = []
    for _idx, (_w, _s) in enumerate(zip(words, mask)):
        if _w == '[SEP]':
            continue
        if _w == '[CLS]':
            continue
        segment_word_ixs[_s][_w].append(_idx)
    groups = []
    for _, s_dict in segment_word_ixs.items():
        for w, g in s_dict.items():
            groups.append(g)
    return groups


class CREW:

    def __init__(self, args, model: ModelWrapper, tokenizer, lime_cache=None):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.lime_words_explainer = MyLIMETextExplainer(
            model, tokenizer, args.device, args.wordpieced,
            args.lime_n_word_features, args.lime_n_word_samples,
            args.max_seq_length, args.seed)

        if lime_cache is None:
            self.lime_wexpl = [[], [], []]
            self.lime_cached = False
        else:
            self.lime_wexpl = lime_cache
            self.lime_cached = True

        self.lime_group_explainer = MyLIMETextExplainer(
            model, tokenizer, args.device, args.wordpieced,
            args.lime_n_word_features, args.lime_n_group_samples,
            args.max_seq_length, args.seed)
        self.num_features = args.lime_n_word_features

    def explain(self, prefix_words, attrs_mask, segments_ids, count=None):

        # ### v1
        # _prefix_words = get_text_groups(groups_same_word_v1(words), prefix_words_tags)
        #
        # ### v2
        # _prefix_words = get_text_groups(groups_same_word_v2(words, segments_ids.detach().cpu().numpy()), prefix_words_tags)
        #
        # ### v3
        # _prefix_words = get_text_groups(groups_same_word_v2(words, attrs_mask.detach().cpu().numpy()), prefix_words_tags)

        if not self.lime_cached:
            top_prefix_words, top_scores = self.lime_words_explainer.explain(' '.join(prefix_words), segments_ids)

            # __out_prefix_words = []
            # __out_scores = []
            # for _pw, _s in zip(top_prefix_words, top_scores):
            #     _gr = re.sub(r'^\d+__', '', _pw).split('__')
            #     for _pw in _gr:
            #         __out_prefix_words.append(_pw)
            #         __out_scores.append(_s / len(_gr))
            # __out_prefix_words = np.array(__out_prefix_words)
            # __out_scores = np.array(__out_scores)
            # __arg_sort = np.argsort(__out_scores ** 2)[-self.num_features:]
            # __out_prefix_words = __out_prefix_words[__arg_sort]
            # __out_scores = __out_scores[__arg_sort]
            # __arg_sort = np.argsort(__out_scores)[::-1]
            # top_prefix_words = __out_prefix_words[__arg_sort]
            # top_scores = __out_scores[__arg_sort]

        else:
            top_prefix_words = np.array(self.lime_wexpl[0][count])  # - 1])
            top_scores = np.array(self.lime_wexpl[1][count])  # - 1])

        EXEC_TIME_PROFILER.timestep('lime_words')

        idxs = np.array([get_idx(pw) for pw in top_prefix_words])

        word_scores = [0] * len(prefix_words)
        for ix, s in zip(idxs, top_scores):
            word_scores[ix] = s
        word_scores = np.array(word_scores)

        words = [get_word(pw) for pw in prefix_words]
        wordcorrs = {
            'emb_sim': EmbdngWordCorr(words),
            'impacts': ImpactWordCorr(word_scores),
            'schema_penalty': SchemaWordCorr(attrs_mask, self.args.cc_schema_scores),
        }
        cc_scores_table = get_cc_scores_table(idxs, segments_ids, wordcorrs, self.args.cc_weights,
                                              self.args.cc_emb_sim_bias)
        EXEC_TIME_PROFILER.timestep('graph')

        if len(np.unique(cc_scores_table[:, 2:3])) == 1:
            groups = [[get_idx(pw) for pw in prefix_words]]
        else:
            groups = cc(cc_scores_table, self.args.cc_alg)
            EXEC_TIME_PROFILER.timestep('corrclust')
        text_groups = get_text_groups(groups.tolist(), prefix_words)

        text_groups_by_score, groups_scores = self.lime_group_explainer.explain(" ".join(text_groups), segments_ids)
        if len(groups) > 1:
            group_scores_desc_ix = [0] * len(text_groups)
            for tg, s in zip(text_groups_by_score, groups_scores):
                gix = int(re.match(r'^(\d+)__', tg).group(1))
                group_scores_desc_ix[gix] = s
            group_scores_desc_ix = np.array(group_scores_desc_ix).argsort()[::-1]
            groups = groups[group_scores_desc_ix]

        EXEC_TIME_PROFILER.timestep('lime_groups')

        return idxs, top_scores, groups, groups_scores
