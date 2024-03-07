import numpy as np
import shap

from crew_impl import CREW
from models import ModelWrapper


def _get_classifier_fn(clf, words, segments):
    def f(masks):
        inxs = np.arange(len(words))
        words_np = np.array(words)
        strings = []
        for mask in masks:
            mask = mask.astype(bool)
            inxs_, words_ = inxs[mask], words_np[mask]
            string = ' '.join([f'{i}_{w}' for i, w in zip(inxs_, words_)])
            strings.append(string)
        mt_probs = clf.get_classifier_fn(segments)(strings)[:, 1]
        return mt_probs

    return f


class CrewShap(CREW):

    def __init__(self, args, model: ModelWrapper, tokenizer, lime_cache=None):
        super().__init__(args, model, tokenizer, lime_cache)
        self.explainer = shap.KernelExplainer(lambda x: np.array([[0.5]]), np.zeros((1, 1)))  # dummy

    def _get_expl(self, clf, words, segments, num_features=0, num_samples=5000):
        # overrides parent class _get_expl
        explainer = shap.KernelExplainer(_get_classifier_fn(clf, words, segments), np.zeros((1, len(words))))
        shap_vals = explainer.shap_values(np.ones((1, len(words))), nsamples=num_samples)
        expl = [*zip(words, shap_vals[0])]
        if num_features <= 0: num_features = len(words)
        expl = sorted(expl, key=lambda x: abs(x[1]), reverse=True)
        expl = expl[:num_features]
        expl = sorted(expl, key=lambda x: x[1], reverse=True)
        return expl
