import sys

import networkx as nx

from utils import normalize_preserve_zero, normalized_min_cut_gpu
from wordcorr import *

_epsilon = sys.float_info.epsilon


def cc_weights(emb_sim=1., impacts=1., schema_penalty=1.):
    weights = np.array([float(emb_sim), float(impacts), float(schema_penalty)], dtype=np.single)
    weights /= weights.sum()
    return {
        'emb_sim': weights[0].item(),
        'impacts': weights[1].item(),
        'schema_penalty': weights[2].item(),
    }


def cc_range(word_corrs: dict, weights: dict):
    r_min, r_max = 0, 0
    for k in word_corrs.keys():
        r = word_corrs[k].get_range() * weights[k]
        r_min += r[0]
        r_max += r[1]
    return np.array([r_min, r_max])


# def get_cc_scores_table(idxs, wordcorrs: dict, weights=None, emb_sim_bias=None, ret_range=False):
def get_cc_scores_table(idxs, segment_ids, wordcorrs: dict, weights=None, emb_sim_bias=None, ret_range=False):

    if weights is None:
        weights = cc_weights()
    idxs = np.array(idxs)[np.argsort(idxs)]

    table = list()

    for a in range(len(idxs) - 1):
        ia = idxs[a]
        for b in range(a + 1, len(idxs)):
            ib = idxs[b]
            x = [ia, ib] if ia < ib else [ib, ia]
            table.append(x)

    # idxs_l, idxs_r = [], []
    # for ix in idxs:
    #     (idxs_l if segment_ids[ix] == 0 else idxs_r).append(ix)
    # for ia in idxs_l:
    #     for ib in idxs_r:
    #         x = [ia, ib] if ia < ib else [ib, ia]
    #         table.append(x)

    for i, r in enumerate(table):
        ia = r[0]
        ib = r[1]
        scores = [wc.get_corr(ia, ib) if weights[k] else 0 for k, wc in wordcorrs.items()]
        table[i].extend(scores)

    _range = cc_range(wordcorrs, weights)

    table = np.array(table)
    if weights['emb_sim'] > 0 and emb_sim_bias:
        if emb_sim_bias == 'mean':
            shift = np.mean(table[:, 2])
        elif emb_sim_bias == 'maxdist':
            table_by_desc_score = table[table[:, 2].argsort()]
            maxdist = 0
            argmaxdist = 0
            for _i in range(0, len(table_by_desc_score) - 1):
                dist = abs(table_by_desc_score[_i, 2] - table_by_desc_score[_i + 1, 2])
                if dist > maxdist:
                    maxdist = dist
                    argmaxdist = _i
            shift = (table_by_desc_score[argmaxdist, 2] + table_by_desc_score[argmaxdist + 1, 2]) / 2
        elif emb_sim_bias == 'maxNdist':
            # centering around Normalized MinCut
            graph = nx.Graph()
            table_by_desc_score = table[table[:, 2].argsort()]
            max_dist = abs(table_by_desc_score[0, 2] - table_by_desc_score[-1, 2]) + _epsilon
            for _i in range(0, len(table_by_desc_score) - 1):
                graph.add_weighted_edges_from([(
                    _i, _i + 1,
                    max_dist - abs(table_by_desc_score[_i, 2] - table_by_desc_score[_i + 1, 2])
                )])
            parts = normalized_min_cut_gpu(graph).detach().cpu().numpy()
            part1 = np.array(graph.nodes, dtype='int')[parts > 0]
            part2 = np.array(graph.nodes, dtype='int')[parts < 0]
            part1_sims, part2_sims = table_by_desc_score[part1, 2], table_by_desc_score[part2, 2]
            parts_bounds = np.array([part1_sims.max(), part1_sims.min(), part2_sims.max(), part2_sims.min()])
            shift = np.mean(parts_bounds[np.argsort(np.power(parts_bounds, 2))][1:3])
        elif emb_sim_bias == 'sq':
            # table = np.concatenate([table, table[:, 2].reshape(-1, 1)], axis=1)
            table[:, 2] -= np.mean(table[:, 2])
            table[:, 2] /= max(abs(table[:, 2].max()), abs(table[:, 2].min()))
            for i, x in enumerate(table[:, 2]):
                sgn = 1. if x >= 0 else -1.
                table[i, 2] = x * x * sgn
            # table = table[:, :5]
            shift = 0
        elif emb_sim_bias == 'meanN':
            table[:, 2] -= np.mean(table[:, 2])
            table[:, 2] /= max(abs(table[:, 2].max()), abs(table[:, 2].min()))
            shift = 0
        table[:, 2] -= shift
        _range -= (shift * weights['emb_sim'])

    table[:, 2:] *= [weights[k] for k in wordcorrs.keys()]

    # ### ATTEMPT
    # for ir in range(len(table)):
    #     r = table[ir]
    #     sga, sgb = segment_ids[int(r[0])], segment_ids[int(r[1])]
    #     impa, impb = wordcorrs['impacts'].impacts[int(r[0])], wordcorrs['impacts'].impacts[int(r[1])]
    #     # if impa * impb < 0:
    #     #     table[ir, 2] = -r[2]
    #     if impa < 0 and impb < 0 and sga + sgb == 1:
    #         table[ir, 2] = -r[2]
    # ### end ATTEMPT

    table_pn = []
    for r in table:
        _r = r[0:2].tolist()
        pos_score, neg_score = 0., 0.
        for x in r[2:]:
            if x >= 0:
                pos_score += x
            else:
                neg_score += x
        pos_score += _epsilon  # noise
        neg_score -= _epsilon  # noise

        _r.append(pos_score)
        _r.append(-neg_score)
        # _r.extend([1, 0]) if r[2:].sum() > 0 else _r.extend([0, 1])  # KWIK

        table_pn.append(_r)

    ret = [np.array(table_pn)]
    if ret_range:
        ret.append(_range)
    return ret if len(ret) > 1 else ret[0]
