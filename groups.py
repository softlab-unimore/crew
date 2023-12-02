from statistics import mean
from typing import Union

import numpy as np
import torch

from embeddings_cache import embs_cos_sim
from utils import entropy, normalize_preserve_zero


# def process_gmask_groups(idxs, ohe_groups, group_scores: torch.Tensor):
#     groups = [set() for _ in range(len(ohe_groups[0][0]))]
#     for x, z in zip(idxs, ohe_groups[0]):
#         for i in range(len(z)):
#             if z[i] == 1:
#                 groups[i].update({x})
#                 break
#     not_empty_groups_ix = [len(g) > 0 for g in groups]
#     groups = np.array(groups)[not_empty_groups_ix]
#     group_scores = group_scores[:, not_empty_groups_ix, :]
#     group_scores = group_scores.detach()[0, :, 0]
#     # group_scores_desc_ix = group_scores.detach()[0, :, 0].argsort().flip(0)
#     group_scores_desc_ix = group_scores.argsort().flip(0).detach().cpu().numpy()
#     return groups, group_scores_desc_ix


def groups_to_grarrays(groups: Union[list, dict]):
    features, labels = [], []
    if isinstance(groups, list):
        groups = groups_to_grdict(groups)
    # for l, g in enumerate(groups):
    for l, g in groups.items():
        for x in g:
            features.append(x)
            labels.append(l)
    return features, labels


def groups_to_grdict(groups):
    grdict = dict()
    for l, g in enumerate(groups):
        grdict[l] = g
    return grdict


def grarrays_to_groups(features, labels):
    # grdict = dict()
    # neg_label = 0  # for singletons
    # for f, l in zip(features, labels):
    #     if l < 0:
    #         neg_label += 1
    #         l = neg_label
    #     g = grdict.get(l, None)
    #     if g is None:
    #         grdict[l] = []
    #     grdict[l].append(f)
    grdict = dict()
    neg_label = 0  # for singletons
    for f, l in zip(features, labels):
        if l < 0:
            neg_label += l
            l = neg_label
        g = grdict.get(l, None)
        if g is None:
            grdict[l] = []
        grdict[l].append(f)
    max_key = max(grdict.keys())
    for k in list(grdict.keys()):
        if k < 0:
            grdict[max_key - k] = grdict[k]
            del grdict[k]
    return [grdict[k] for k in sorted(grdict.keys())]


def get_text_groups(groups: Union[list, dict], data):
    if isinstance(groups, list):
        groups = groups_to_grdict(groups)
    # text_groups = [[data[ix] for ix in g] for g in groups]
    text_groups = [f'{gr_id}__{"__".join([data[ix] for ix in gr])}' for gr_id, gr in groups.items()]
    return text_groups


def infer_group_scores(groups, word_scores):
    group_scores = []
    for g in groups:
        group_score = []
        for ix in g:
            group_score.append(word_scores[ix])
        group_scores.append(mean(group_score))
    group_scores = np.array(group_scores)
    group_scores_desc_ix = group_scores.argsort()[::-1]
    return group_scores, group_scores_desc_ix


def rate_groups(groups, words, attrs_mask: torch.Tensor, word_scores):
    attrs_mask = attrs_mask.detach().cpu().numpy()
    groups_entropy = [entropy([attrs_mask[ix] for ix in g]) for g in groups]

    groups_intra_emb_sims = []
    for ig in range(len(groups)):
        g = list(groups[ig])

        # emb_sims = []
        # for i in range(len(g) - 1):
        #     for j in range(i + 1, len(g)):
        #         emb_sims.append(float(cos_sim(
        #             EMBS[words[g[i]]], EMBS[words[g[j]]]
        #         )))

        X = []
        Y = []
        for i in range(len(g) - 1):
            for j in range(i + 1, len(g)):
                X.append(words[g[i]])
                Y.append(words[g[j]])
        if len(X) > 0:
            emb_sims = embs_cos_sim(X, Y)
        else:
            emb_sims = np.array([0])

        # groups_intra_emb_sims.append(emb_sims.mean() if len(emb_sims) > 0 else 0)
        groups_intra_emb_sims.append(emb_sims.mean())

    if word_scores is not None and len(word_scores) > 0:
        word_scores = normalize_preserve_zero(word_scores)
    groups_intra_impacts_stdev = [np.mean([word_scores[ix] for ix in g]) for g in groups]

    return groups_entropy, groups_intra_emb_sims, groups_intra_impacts_stdev
