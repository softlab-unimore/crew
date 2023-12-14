import argparse
import json
import os
import pickle
import re

import numpy as np

from prefix_words import get_word


def split_groups(grs_str: str):
    return [str(re.sub(r'^\d+__', '', gr_str)).split('__') for gr_str in grs_str]


def remove_prefix(pfx_words):
    return [get_word(pw) for pw in pfx_words]


# def _run(dataset, experiment, _expl, ix, html: str):
#     with open(f'./output/{dataset}/{experiment}/lime_{_expl}.pkl', 'rb') as f:
def _run(expls_dir: str, _expl: str, ix: int, visual_dir: str, html: str):
    with open(f'{expls_dir}/lime_{_expl}.pkl', 'rb') as f:
        lime_expl = pickle.load(f)
    words = lime_expl[0][ix]
    coefs = lime_expl[1][ix]
    pred_probs = lime_expl[2][ix]
    words_coefs = []
    for w, c in zip(words, coefs):
        words_coefs.append([w, c])

    top_abs_arg = np.flip(np.abs(coefs).argsort())[:10]
    top_words = np.array(words)[top_abs_arg]
    groups = split_groups(top_words)
    groups = [remove_prefix(g) for g in groups]
    groups = ['_'.join(g) for g in groups]
    # bar_chart = [[w, c] for w, c in zip(np.array(words)[top_abs_arg], np.array(coefs)[top_abs_arg])]
    bar_chart = [[w, c] for w, c in zip(groups, np.array(coefs)[top_abs_arg])]

    visual_dir = f'{visual_dir}/{_expl}'
    if not os.path.exists(visual_dir):
        os.makedirs(visual_dir)

    html = html.replace('"<!--PRED_PROB-->"', json.dumps(pred_probs))
    html = html.replace('"<!--BAR_CHART-->"', json.dumps(bar_chart))
    html = html.replace('"<!--HIGHLIGHTED_TEXT-->"', json.dumps(words_coefs))
    with open(f'{visual_dir}/expl{ix}.html', 'w') as f:
        f.write(html)


# def main(dataset, experiment, ix):
#     with open('./visual/expl_template.html', 'r') as f:
#         html = f.read()
#
#     _run(dataset, experiment, 'wexpl', ix, html)
#     _run(dataset, experiment, 'gexpl', ix, html)


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

    _run(args.expls_dir, 'wexpl', args.expl_id, args.visual_dir, html)
    _run(args.expls_dir, 'gexpl', args.expl_id, args.visual_dir, html)
