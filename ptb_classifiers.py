import re

import numpy as np
import torch
from lime.lime_text import LimeTextExplainer

from models import ModelWrapper
from prefix_words import get_idx, get_word
# from utils import EXEC_TIME_PROFILER
from data_parser import InputExample, convert_examples_to_features


# def parse_into_prefix_words(chunks: list[str]):
#     prefix_words = []
#     for c in chunks:
#         prefix_words.extend(str(re.sub(r'^\d+__', '', c)).split('__'))
#     prefix_words = [w for w in prefix_words if w]
#     return prefix_words


class WordsClassifier:

    def __init__(self, model: ModelWrapper, tokenizer, device='cuda', wordpieced=False, max_seq_length=200):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.wordpieced = wordpieced
        self.max_seq_length = max_seq_length

    def parse(self, text):
        chunks = re.split(r'[^\w#]+', text.strip())
        chunks = [c for c in chunks if c]
        return chunks

    def predict_proba(self, ptb_strings, segment_ids):
        examples = []
        for s in ptb_strings:
            # chunks = re.split(r'[^\w#]+', s.strip())
            # prefix_words = parse_into_prefix_words(chunks)
            prefix_words = self.parse(s)
            words_a, words_b = [], []
            for pw in prefix_words:
                (words_b if segment_ids[get_idx(pw)] else words_a).append(get_word(pw))

            examples.append(InputExample('#0', words_a=words_a, words_b=words_b, wordpieced=self.wordpieced))
        features = convert_examples_to_features(examples, [0, 1], self.max_seq_length, self.tokenizer, 'classification')

        input_ids_ = torch.tensor([f.input_ids for f in features], device=self.device)
        attn_mask_ = torch.tensor([f.input_mask for f in features], device=self.device)
        segment_ids_ = torch.tensor([f.segment_ids for f in features], device=self.device)
        logits = self.model.predict(None, input_ids_, attn_mask_, segment_ids_, self.wordpieced)
        results = logits.cpu().detach().numpy()

        return results

    def get_classifier_fn(self, segments):
        def classifier_fn(ptb_strings):
            return self.predict_proba(ptb_strings, segments)
        return classifier_fn


class GroupsClassifier(WordsClassifier):

    def parse(self, text):
        chunks = re.split(r'[^\w#]+', text.strip())
        chunks = [c for c in chunks if c]
        chunks_ = []
        for c in chunks:
            chunks_.extend(str(re.sub(r'^\d+__', '', c)).split('__'))
        chunks = [c for c in chunks_ if c]
        return chunks


# class MyLIMETextExplainer:
#
#     def __init__(self, model, tokenizer, device, wordpieced, num_features=0, num_samples=0, max_sequence_length=200, random_state=None):
#         self._device = device
#         self._wordpieced = wordpieced
#         self._lime_kwargs = {}
#         if num_features:
#             self._lime_kwargs['num_features'] = num_features
#         if num_samples > 0:
#             self._lime_kwargs['num_samples'] = num_samples
#         self.max_sequence_length = max_sequence_length
#
#         self._model = model
#         self._tokenizer = tokenizer
#         self._clf = WordsClassifier(model, tokenizer, device, wordpieced, max_sequence_length)
#         self._split_expression = r'[^\w#]+'
#         self._explainer = LimeTextExplainer(
#             class_names=['no_match', 'match'],
#             split_expression=self._split_expression,
#             random_state=random_state
#         )
#
#         self.num_features = num_features
#
#     def explain(self, text, segment_ids):
#
#         # def classifier_fn(strings):
#         #     return self._clf.predict_proba(strings, segment_ids)
#
#         num_words = len(re.split(self._split_expression, text))
#         self._lime_kwargs['num_features'] = num_words
#         exp = self._explainer.explain_instance(text, self._clf.get_classifier_fn(segment_ids), **self._lime_kwargs)
#         prefix_words_scores = np.array(exp.as_list())  # sorted by scores absolute value, desc
#         prefix_words, scores = prefix_words_scores[:self.num_features, 0], prefix_words_scores[:self.num_features, 1].astype(float)
#
#         scores_argsort = np.argsort(scores)[::-1]
#         ret = [prefix_words[scores_argsort], scores[scores_argsort]]
#         return tuple(ret)
