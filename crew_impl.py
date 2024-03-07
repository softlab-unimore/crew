import re

import numpy as np
from lime.lime_text import LimeTextExplainer

from groups import get_text_groups
from ptb_classifiers import WordsClassifier, GroupsClassifier
from models import ModelWrapper
from my_corrclust import get_cc_scores_table
from prefix_words import get_idx, get_word
from pyccalg import cc
from utils import EXEC_TIME_PROFILER
from wordcorr import EmbdngWordCorr, ImpactWordCorr, SchemaWordCorr


class CREW:

    def __init__(self, args, model: ModelWrapper, tokenizer, lime_cache=None):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.explainer = LimeTextExplainer(
            class_names=['no_match', 'match'], split_expression=r'[^\w#]+', random_state=args.seed
        )
        self.wclf = WordsClassifier(model, tokenizer, args.device, args.wordpieced, args.max_seq_length)
        self.lime_n_word_samples = args.lime_n_word_samples
        self.lime_n_word_features = args.lime_n_word_features

        if args.attribution_method == 'shap':
            lime_cache = None

        if lime_cache is None:
            self.lime_wexpl = [[], [], []]
            self.lime_cached = False
        else:
            self.lime_wexpl = lime_cache
            self.lime_cached = True

        self.gclf = GroupsClassifier(model, tokenizer, args.device, args.wordpieced, args.max_seq_length)
        self.num_features = args.lime_n_word_features
        self.lime_n_group_samples = args.lime_n_group_samples

    def _get_expl(self, clf, words, segments, num_features=0, num_samples=5000):
        expl = self.explainer.explain_instance(' '.join(words), clf.get_classifier_fn(segments),
                                               num_features=len(words), num_samples=num_samples)
        if num_features <= 0: num_features = len(words)
        expl = sorted(expl.as_list()[:num_features], key=lambda x: x[1], reverse=True)
        return expl

    def explain(self, prefix_words, attrs_mask, segments_ids, count=None):

        if not self.lime_cached:
            expl = self._get_expl(self.wclf, prefix_words, segments_ids,
                                  self.lime_n_word_features, self.lime_n_word_samples)
            top_prefix_words, top_scores = zip(*expl)

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

        expl = self._get_expl(self.gclf, text_groups, segments_ids,
                              num_samples=self.lime_n_group_samples)
        text_groups_by_score, groups_scores = zip(*expl)
        if len(groups) > 1:
            group_scores_desc_ix = [0] * len(text_groups)
            for tg, s in zip(text_groups_by_score, groups_scores):
                gix = int(re.match(r'^(\d+)__', tg).group(1))
                group_scores_desc_ix[gix] = s
            group_scores_desc_ix = np.array(group_scores_desc_ix).argsort()[::-1]
            groups = groups[group_scores_desc_ix]

        EXEC_TIME_PROFILER.timestep('lime_groups')

        return idxs, top_scores, groups, groups_scores
