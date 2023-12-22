import argparse
import json
import os
import pickle
import re
from copy import deepcopy

import pandas as pd

from prefix_words import get_word


def split_groups(grs_str: list[str]):
    return [str(re.sub(r'^\d+__', '', gr_str)).split('__') for gr_str in grs_str]


def remove_prefix(pfx_words):
    return [get_word(pw) for pw in pfx_words]


def visualize(expls_dir, expl_id, visual_dir):
    with open('expl_template.html', 'r') as f:
        html = f.read()

    with open(f'{expls_dir}/pred_probs', 'rb') as f:
        pred_probs = pickle.load(f)
    pred_probs = pred_probs[expl_id]

    with open(f'{expls_dir}/wexpls', 'rb') as f:
        expls = pickle.load(f)
    expl = expls[expls['data_inx'] == expl_id].drop(columns=['data_inx', 'column']).set_index('wid').sort_index()

    expl_by_abs_imp = expl.sort_values(by='impact', ascending=False,
                                       key=lambda s: pd.Series.apply(s, lambda x: abs(x)))
    bar_chart = expl_by_abs_imp[['word', 'impact']].values.tolist()[:10]

    whtml = (html + '.')[:-1]
    whtml = whtml.replace('"<!--PRED_PROB-->"', json.dumps(pred_probs))
    whtml = whtml.replace('"<!--BAR_CHART-->"', json.dumps(bar_chart))

    vexpl = expl.apply(lambda s: pd.Series([s['word'], 0, s['impact']]), axis=1).values.tolist()
    tot_len = 0
    for x in vexpl:
        x[1] = tot_len
        tot_len += len(x[0]) + 1
    tot_len += 1
    expl_by_imp = expl.sort_values(by='impact', ascending=False)
    vexpl_by_imp = expl_by_imp.apply(lambda s: pd.Series([s['word'], 0, s['impact']]), axis=1).values.tolist()
    for x in vexpl_by_imp:
        x[1] = tot_len
        tot_len += len(x[0]) + 1

    raw_l_entity = ' '.join(expl[expl['segment'] == 0]['word'])
    raw_r_entity = ' '.join(expl[expl['segment'] == 1]['word'])
    raw_words = '\n'.join(expl_by_imp['word'])

    visual_dir = f'{visual_dir}/wexpls'
    if not os.path.exists(visual_dir):
        os.makedirs(visual_dir)

    whtml = whtml.replace('"WORDS_COEFS"', json.dumps(vexpl + vexpl_by_imp))
    whtml = whtml.replace('"RAW_TEXT"', json.dumps(raw_l_entity + '\n' + raw_r_entity + '\n\n' + raw_words))
    with open(f'{visual_dir}/expl{expl_id}.html', 'w') as f:
        f.write(whtml)

    with open(f'{expls_dir}/gexpls', 'rb') as f:
        expls = pickle.load(f)
    gexpl = expls[expls['data_inx'] == expl_id].drop(columns=['data_inx', 'group'])
    gexpl = gexpl.reset_index().rename(columns={'index': 'gid', 'impact': 'gimpact'})
    gexpl = gexpl.explode('wids').rename(columns={'wids': 'wid'}).set_index('wid')
    expl = pd.merge(expl, gexpl, left_index=True, right_index=True)

    grouped = expl.groupby('gid')
    bar_chart = [('_'.join(gr['word']), gr['gimpact'].mean()) for _, gr in grouped]
    bar_chart = sorted(bar_chart, key=lambda x: abs(x[1]), reverse=True)

    ghtml = (html + '.')[:-1]
    ghtml = ghtml.replace('"<!--PRED_PROB-->"', json.dumps(pred_probs))
    ghtml = ghtml.replace('"<!--BAR_CHART-->"', json.dumps(bar_chart))

    visual_dir = f'{visual_dir}/../gexpls'
    if not os.path.exists(visual_dir):
        os.makedirs(visual_dir)

    vexpl = expl.apply(lambda s: pd.Series([s['word'], 0, s['gimpact']]), axis=1).values.tolist()
    tot_len = 0
    for x in vexpl:
        x[1] = tot_len
        tot_len += len(x[0]) + 1
    tot_len += 1
    vexpl_by_imp = []
    raw_groups = ''
    for _, gr in expl.sort_values('gimpact', ascending=False).groupby('gid'):
        for x in gr.apply(lambda s: pd.Series([s['word'], 0, s['gimpact']]), axis=1).values.tolist():
            x[1] = tot_len
            tot_len += len(x[0]) + 1
            vexpl_by_imp.append(x)
            raw_groups += x[0] + ' '
        raw_groups = raw_groups[:-1] + '\n'

    ghtml = ghtml.replace('"WORDS_COEFS"', json.dumps(vexpl + vexpl_by_imp))
    ghtml = ghtml.replace('"RAW_TEXT"', json.dumps(raw_l_entity + '\n' + raw_r_entity + '\n\n' + raw_groups))
    with open(f'{visual_dir}/expl{expl_id}.html', 'w') as f:
        f.write(ghtml)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('expls_dir', type=str,
                        help='The directory with the explanations in binary format (the CREW output)')
    parser.add_argument('expl_id', type=int,
                        help='The position of the explanation for the desired pair of entity descriptions')
    parser.add_argument('visual_dir', type=str,
                        help='The directory where to write the .html the visualizes the explanation')

    args = parser.parse_args()

    visualize(args.expls_dir, args.expl_id, args.visual_dir)
