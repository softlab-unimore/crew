import re

import numpy as np
import torch
from lime.lime_text import LimeTextExplainer

from embeddings_cache import ModelWrapper
from prefix_words import get_idx, get_word
# from utils import EXEC_TIME_PROFILER
from data_parser import InputExample, convert_examples_to_features


def parse_into_prefix_words(chunks: list[str]):
    # todo: subclasses or strategy pattern
    # prefix_words = re.split(r'[^\w#]+', text.strip())
    # chunks = re.split(r'[^\w#]+', text.strip())
    prefix_words = []
    for c in chunks:
        # prefix_words.extend(c.split('__'))
        prefix_words.extend(str(re.sub(r'^\d+__', '', c)).split('__'))
    prefix_words = [w for w in prefix_words if w]
    return prefix_words


class _Classifier:

    def __init__(self, model: ModelWrapper, tokenizer, device='cuda', wordpieced=False, max_seq_length=200):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.wordpieced = wordpieced
        self.max_seq_length = max_seq_length

    def predict_proba(self, strings, segment_ids):
        examples = []
        for s in strings:
            # prefix_words = re.split(r'[^\w#]+', s.strip())
            # prefix_words = [w for w in prefix_words if w]
            chunks = re.split(r'[^\w#]+', s.strip())
            prefix_words = parse_into_prefix_words(chunks)
            words_a, words_b = [], []
            for pw in prefix_words:
                # (words_b if get_segment_id(pw) else words_a).append(get_word(pw))
                (words_b if segment_ids[get_idx(pw)] else words_a).append(get_word(pw))

            examples.append(InputExample('#0', words_a=words_a, words_b=words_b, wordpieced=self.wordpieced))
        features = convert_examples_to_features(examples, [0, 1], self.max_seq_length, self.tokenizer, 'classification')

        # results = []
        # for f in features:
        #     input_ids_ = torch.tensor([f.input_ids], device=self.device)
        #     attn_mask_ = torch.tensor([f.input_mask], device=self.device)
        #     segment_ids_ = torch.tensor([f.segment_ids], device=self.device)
        #     logits = predict(self.model, input_ids_, attn_mask_, segment_ids_)
        #     results.append(logits.cpu().detach().numpy()[0])
        # results = np.array(results)

        # results = []
        # batch_size = 8
        # for i in range(0, len(features), batch_size):
        #     feats_batch = features[i: min(i + batch_size, len(features))]
        #     input_ids_ = torch.tensor([f.input_ids for f in feats_batch], device=self.device)
        #     attn_mask_ = torch.tensor([f.input_mask for f in feats_batch], device=self.device)
        #     segment_ids_ = torch.tensor([f.segment_ids for f in feats_batch], device=self.device)
        #     logits = predict(self.model, input_ids_, attn_mask_, segment_ids_)
        #     results.extend(logits.cpu().detach().numpy().tolist())
        # results = np.array(results)

        input_ids_ = torch.tensor([f.input_ids for f in features], device=self.device)
        attn_mask_ = torch.tensor([f.input_mask for f in features], device=self.device)
        segment_ids_ = torch.tensor([f.segment_ids for f in features], device=self.device)
        # logits = predict(self.model, input_ids_, attn_mask_, segment_ids_)
        logits = self.model.predict(None, input_ids_, attn_mask_, segment_ids_, self.wordpieced)
        results = logits.cpu().detach().numpy()

        return results


class MyLIMETextExplainer:

    def __init__(self, model, tokenizer, device, wordpieced, num_features=0, num_samples=0, max_sequence_length=200, random_state=None):
        self._device = device
        self._wordpieced = wordpieced
        self._lime_kwargs = {}
        if num_features:
            self._lime_kwargs['num_features'] = num_features
        if num_samples > 0:
            self._lime_kwargs['num_samples'] = num_samples
        self.max_sequence_length = max_sequence_length

        self._model = model
        self._tokenizer = tokenizer
        self._clf = _Classifier(model, tokenizer, device, wordpieced, max_sequence_length)
        self._split_expression = r'[^\w#]+'
        self._explainer = LimeTextExplainer(
            class_names=['no_match', 'match'],
            split_expression=self._split_expression,
            random_state=random_state
        )

        self.num_features = num_features

    def explain(self, text, segment_ids):
        # split_expression = r'[^\w#]+'
        # explainer = LimeTextExplainer(
        #     class_names=['no_match', 'match'],
        #     split_expression=split_expression
        # )

        def classifier_fn(strings):
            return self._clf.predict_proba(strings, segment_ids)

        num_words = len(re.split(self._split_expression, text))
        # if self._lime_kwargs['num_features'] == -1:
        #     self._lime_kwargs['num_features'] = num_words
        self._lime_kwargs['num_features'] = num_words
        # exp = explainer.explain_instance(text, self._clf.predict_proba, **self._lime_kwargs)
        exp = self._explainer.explain_instance(text, classifier_fn, **self._lime_kwargs)
        # EXEC_TIME_PROFILER.timestep('lime')
        prefix_words_scores = np.array(exp.as_list())  # sorted by scores absolute value, desc
        prefix_words, scores = prefix_words_scores[:self.num_features, 0], prefix_words_scores[:self.num_features, 1].astype(float)

        # words_a_dict, words_b_dict = dict(), dict()
        # prefix_words_ = np.array(parse_into_prefix_words(prefix_words))
        # for pw in prefix_words_:
        #     ix = get_idx(pw)
        #     # (words_b_dict if get_segment_id(pw) else words_a_dict)[ix] = get_word(pw)
        #     (words_b_dict if segment_ids[ix] else words_a_dict)[ix] = get_word(pw)
        # words_a = [words_a_dict[k] for k in sorted(words_a_dict)]
        # words_b = [words_b_dict[k] for k in sorted(words_b_dict)]

        # example = InputExample('#0', words_a=words_a, words_b=words_b, wordpieced=self._wordpieced)
        # features = convert_examples_to_features([example], [0, 1], self.max_sequence_length, self._tokenizer, 'classification')
        # f = features[0]
        # input_ids_ = torch.tensor([f.input_ids], device=self._device)
        # attention_mask_ = torch.tensor([f.input_mask], device=self._device)
        # segment_ids_ = torch.tensor([f.segment_ids], device=self._device)
        # with torch.no_grad():
        #     inputs = {
        #         'input_ids': input_ids_,
        #         'attention_mask': attention_mask_,
        #         'token_type_ids': segment_ids_,
        #     }
        #     outputs = self._model(**inputs)
        #     logits = outputs[0]
        # logits = torch.softmax(logits, dim=1)
        # pred = logits.argmax(dim=1)

        # if np.argmax(exp.predict_proba) == 0:
        #     scores = -scores
        scores_argsort = np.argsort(scores)[::-1]

        # EXEC_TIME_PROFILER.timestep('post_lime')

        ret = [prefix_words[scores_argsort], scores[scores_argsort]]
        return tuple(ret)
