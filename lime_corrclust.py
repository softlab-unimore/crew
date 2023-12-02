import re

import numpy as np
import torch

from corrclust import corrclust
from embeddings_cache import ModelWrapper
from interpret_lime import MyLIMETextExplainer
from my_corrclust import get_cc_scores_table
from prefix_words import get_idx, prefix_words_to_feature
from utils import EXEC_TIME_PROFILER
from wordcorr import EmbdngWordCorr, ImpactWordCorr, SchemaWordCorr


def get_text_groups(ix_groups, data):
    text_groups = [[data[ix] for ix in g] for g in ix_groups]
    text_groups = [f'{_gr_id}__{"__".join(_pw)}' for _gr_id, _pw in enumerate(text_groups)]
    return text_groups


def groups_same_word_v1(words):
    word_ixs = {_w: [] for _w in words}
    for _idx, _w in enumerate(words):
        word_ixs[_w].append(_idx)
    del word_ixs['[CLS]']
    del word_ixs['[SEP]']
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


class LimeCorrClust:

    def __init__(self, args, model: ModelWrapper, tokenizer, lime_cache=None, do_groups=True):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.lime_words_explainer = MyLIMETextExplainer(
            model, tokenizer, args.device, args.wordpieced,
            # args.num_features, args.num_words_samples,
            args.lime_n_word_features, args.lime_n_word_samples,
            args.max_seq_length, args.seed)

        if lime_cache is None:
            self.lime_wexpl = [[], [], []]
            self.lime_cached = False
        else:
            self.lime_wexpl = lime_cache
            self.lime_cached = True
        # self.lime_wexpl = [[], [], []]
        # slh_score = []
        self.do_groups = do_groups
        if do_groups:
            # degrad_score_groups = DegradationScoreF1(.10)
            self.lime_gexpl = [[], [], []]
            self.lime_group_explainer = MyLIMETextExplainer(
                model, tokenizer, args.device, args.wordpieced,
                # args.num_features, args.num_group_samples,
                args.lime_n_word_features, args.lime_n_group_samples,
                args.max_seq_length, args.seed)
        self.num_features = args.lime_n_word_features

    def lime_corrclust(self, prefix_words_tags, words, attrs_mask, segments_ids, pred_probs, count=None):
        skip_tags = np.array([w not in ['[CLS]', '[SEP]'] for w in words])
        _prefix_words = np.array(prefix_words_tags)[skip_tags]
        # words = [w if w in ['[CLS]', '[SEP]'] else get_word(w) for w in a_no_match[_ix_words]]

        # ### v1
        # _prefix_words = get_text_groups(groups_same_word_v1(words), prefix_words_tags)
        #
        # ### v2
        # _prefix_words = get_text_groups(groups_same_word_v2(words, segments_ids.detach().cpu().numpy()), prefix_words_tags)
        #
        # ### v3
        # _prefix_words = get_text_groups(groups_same_word_v2(words, attrs_mask.detach().cpu().numpy()), prefix_words_tags)

        if not self.lime_cached:
            top_prefix_words, top_scores = self.lime_words_explainer.explain(' '.join(_prefix_words), segments_ids)

            if len(top_prefix_words) == len(_prefix_words):
                out_probs = pred_probs
            else:
                f = prefix_words_to_feature(top_prefix_words, segments_ids, self.tokenizer, self.args.wordpieced,
                                            self.args.max_seq_length)
                input_ids_ = torch.tensor([f.input_ids], device=self.args.device)
                attention_mask_ = torch.tensor([f.input_mask], device=self.args.device)
                segment_ids_ = torch.tensor([f.segment_ids], device=self.args.device)
                logits_ = self.model.predict(None, input_ids_, attention_mask_, segment_ids_, self.args.wordpieced)
                out_probs = logits_.detach().cpu().numpy()

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

            self.lime_wexpl[0].append(top_prefix_words.tolist())
            self.lime_wexpl[1].append(top_scores.tolist())
            self.lime_wexpl[2].append(out_probs[0].tolist())

        else:
            top_prefix_words = np.array(self.lime_wexpl[0][count - 1])
            top_scores = np.array(self.lime_wexpl[1][count - 1])
            out_probs = [np.array(self.lime_wexpl[2][count - 1])]

        EXEC_TIME_PROFILER.timestep('lime_words')

        w_expl_pred = out_probs[0].argmax()
        idxs = np.array([get_idx(pw) for pw in top_prefix_words])

        word_scores = [0] * len(words)
        for ix, s in zip(idxs, top_scores):
            word_scores[ix] = s
        word_scores = np.array(word_scores)

        # idxs_hnrg, _ = split_idxs_by_energy(idxs, top_scores, self.args.energy_rate)
        idxs_hnrg = idxs

        g_expl_pred, groups = None, []
        if self.do_groups:
            # cc_scores_table = get_cc_scores_table(idxs_hnrg, words, attrs_mask,
            #                                       args.cc_emb_sim_bias, word_scores, args.cc_weights)
            wordcorrs = {
                'emb_sim': EmbdngWordCorr(words),
                'impacts': ImpactWordCorr(word_scores),
                'schema_penalty': SchemaWordCorr(attrs_mask, self.args.cc_schema_scores),
            }
            cc_scores_table = get_cc_scores_table(idxs_hnrg, segments_ids, wordcorrs, self.args.cc_weights,
                                                  self.args.cc_emb_sim_bias)
            EXEC_TIME_PROFILER.timestep('graph')

            if len(np.unique(cc_scores_table[:, 2:3])) == 1:
                groups = [[get_idx(pw) for pw in _prefix_words]]
                prefix_words_groups = [_prefix_words]
            else:
                groups = corrclust(cc_scores_table, self.args.cc_alg)
                EXEC_TIME_PROFILER.timestep('corrclust')
                prefix_words_groups = [[prefix_words_tags[ix] for ix in sorted(g)] for g in groups]
                text_groups = ['__'.join(pwg) for pwg in prefix_words_groups]
                text_groups = [f'{i}__{tg}' for i, tg in enumerate(text_groups)]

            text_groups_by_score, groups_scores = self.lime_group_explainer.explain(" ".join(text_groups), segments_ids)
            if len(groups) > 1:
                group_scores_desc_ix = [0] * len(text_groups)
                for tg, s in zip(text_groups_by_score, groups_scores):
                    gix = int(re.match(r'^(\d+)__', tg).group(1))
                    group_scores_desc_ix[gix] = s
                group_scores_desc_ix = np.array(group_scores_desc_ix).argsort()[::-1]
                groups = groups[group_scores_desc_ix]
                # possibly extract groups_scores_desc_ix directly from out_text_groups
            g_expl_pred = w_expl_pred

            self.lime_gexpl[0].append(text_groups_by_score.tolist())
            self.lime_gexpl[1].append(groups_scores.tolist())
            self.lime_gexpl[2].append(out_probs[0].tolist())

            EXEC_TIME_PROFILER.timestep('lime_groups')

        #     features, labels = groups_to_grarrays(groups)
        #     if len(set(labels)) >= 2:
        #         features = np.array(features).reshape((-1, 1))  # slh_metric
        #         # features = torch.stack(
        #         #     [EMBS[i] for i in bert_tokenizer.convert_tokens_to_ids(words[features])]
        #         # ).detach().cpu().numpy()  # 'cosine'
        #
        #         rmin, rmax = cc_range(args.cc_weights, args.cc_emb_sim_bias, 0.05)
        #         cc_scores_dict = dict()
        #         for x in cc_scores_table:
        #             cc_scores_dict[f'{int(x[0])}_{int(x[1])}'] = x[2:]
        #         # slh_metric = lambda A, B: cc_scores_distance(A, B, cc_scores_dict, rmin, rmax)
        #
        #         slh_score.append(silhouette_score(features, labels, metric=cc_scores_distance, **{
        #             'cc_scores_dict': cc_scores_dict, 'range_min': rmin, 'range_max': rmax
        #         }))
        #         # slh_score.append(silhouette_score(features, labels, metric='cosine'))
        return w_expl_pred, idxs, g_expl_pred, groups
