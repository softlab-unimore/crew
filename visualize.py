import argparse
import json
import os
import pickle
import re
from copy import deepcopy

import numpy as np

from prefix_words import get_word, get_idx


def split_groups(grs_str: list[str]):
    return [str(re.sub(r'^\d+__', '', gr_str)).split('__') for gr_str in grs_str]


def remove_prefix(pfx_words):
    return [get_word(pw) for pw in pfx_words]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('expls_dir', type=str,
                        help='The directory with the explanations in binary format (the CREW output)')
    parser.add_argument('expl_id', type=int,
                        help='The position of the explanation for the desired pair of entity descriptions')
    parser.add_argument('visual_dir', type=str,
                        help='The directory where to write the .html the visualizes the explanation')

    args = parser.parse_args()

    # for i in range(0, 20):
    #     main('dblp_scholar_dirty', 'lime_corrclust_100', i)

    with open('./visual/expl_template.html', 'r') as f:
        html = f.read()

    # _run(args.expls_dir, 'wexpls', args.expl_id, args.visual_dir, html)
    with open(f'{args.expls_dir}/wexpls', 'rb') as f:
        expls = pickle.load(f)
    expl = expls[expls['data_inx'] == args.expl_id]
    wids = expl['wid']
    words = expl['word']
    coefs = expl['impact']
    segments = expl['segment']
    pred_probs = [0.33, 0.67]  # todo

    new_wids = {wid: i for i, wid in enumerate(sorted(wids))}
    wids = [new_wids[wid] for wid in wids]

    words_coefs = [list()] * len(wids)
    for i, w, c, s in zip(wids, words, coefs, segments):
        words_coefs[i] = [w, 0, c, s]
    # words_coefs = {get_idx(x[0]): [get_word(x[0]), 0] + x[1:] for x in words_coefs}
    # words_coefs = [words_coefs[k] for k in sorted(words_coefs)]
    words_coefs_by_abs_imp = sorted(deepcopy(words_coefs), key=lambda x: abs(x[2]), reverse=True)

    # top_abs_arg = np.flip(np.abs(coefs).argsort())[:10]
    # top_words = np.array(words)[top_abs_arg]
    # top_coefs = np.array(coefs)[top_abs_arg]

    top_words_coefs = words_coefs_by_abs_imp[:10]
    bar_chart = [[w, c] for w, _, c, _ in top_words_coefs]

    whtml = (html + '.')[:-1]
    whtml = whtml.replace('"<!--PRED_PROB-->"', json.dumps(pred_probs))
    whtml = whtml.replace('"<!--BAR_CHART-->"', json.dumps(bar_chart))
    # whtml = whtml.replace('"<!--HIGHLIGHTED_TEXT-->"', json.dumps(words_coefs))

    tot_len = 0
    for x in words_coefs:
        x[1] = tot_len
        tot_len += len(x[0]) + 1
    tot_len += 1
    words_coefs_by_coef = sorted(deepcopy(words_coefs), key=lambda x: x[2], reverse=True)
    for x in words_coefs_by_coef:
        x[1] = tot_len
        tot_len += len(x[0]) + 1

    raw_l_entity = ' '.join([x[0] for x in words_coefs if x[3] == 0])
    raw_r_entity = ' '.join([x[0] for x in words_coefs if x[3] == 1])
    words = '\n'.join([x[0] for x in words_coefs_by_coef])

    visual_dir = f'{args.visual_dir}/wexpls'
    if not os.path.exists(visual_dir):
        os.makedirs(visual_dir)

    whtml = whtml.replace('"WORDS_COEFS"', json.dumps(words_coefs + words_coefs_by_coef))
    whtml = whtml.replace('"RAW_TEXT"', json.dumps(raw_l_entity + '\n' + raw_r_entity + '\n\n' + words))
    with open(f'{visual_dir}/expl{args.expl_id}.html', 'w') as f:
        f.write(whtml)
    # _run(args.expls_dir, 'gexpls', args.expl_id, args.visual_dir, html)

    with open(f'{args.expls_dir}/gexpls', 'rb') as f:
        expls = pickle.load(f)
    expl = expls[expls['data_inx'] == args.expl_id]
    groups = expl['wids']
    coefs = expl['impact']
    raw_groups = ['_'.join([words_coefs[new_wids[wid]][0] for wid in gr]) for gr in groups]
    bar_chart = [x for x in zip(raw_groups, coefs)]
    bar_chart = sorted(bar_chart, key=lambda x: abs(x[1]), reverse=True)

    ghtml = (html + '.')[:-1]
    ghtml = ghtml.replace('"<!--PRED_PROB-->"', json.dumps(pred_probs))
    ghtml = ghtml.replace('"<!--BAR_CHART-->"', json.dumps(bar_chart))

    visual_dir = f'{args.visual_dir}/gexpls'
    if not os.path.exists(visual_dir):
        os.makedirs(visual_dir)

    words_gcoefs = [None] * len(wids)
    for g, c in zip(groups, coefs):
        for ix in g:
            words_gcoefs[ix] = c

    for x, y in zip(words_coefs, words_gcoefs):
        x[2] = y

    tot_len = 0
    for x in words_coefs:
        tot_len += len(x[0]) + 1
    tot_len += 1
    groups_coefs_by_coef = sorted([(g, c) for g, c in zip(groups, coefs)], key=lambda x: x[1], reverse=True)
    words_coefs_by_coef = []
    raw_groups = ''
    for gr, _ in groups_coefs_by_coef:
        for wid in gr:
            x = deepcopy(words_coefs[new_wids[wid]])
            x[1] = tot_len
            tot_len += len(x[0]) + 1
            words_coefs_by_coef.append(x)
            raw_groups += words_coefs[new_wids[wid]][0] + ' '
        raw_groups = raw_groups[:-1] + '\n'

    ghtml = ghtml.replace('"WORDS_COEFS"', json.dumps(words_coefs + words_coefs_by_coef))
    ghtml = ghtml.replace('"RAW_TEXT"', json.dumps(raw_l_entity + '\n' + raw_r_entity + '\n\n' + raw_groups))
    with open(f'{visual_dir}/expl{args.expl_id}.html', 'w') as f:
        f.write(ghtml)
