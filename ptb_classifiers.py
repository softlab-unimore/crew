import re

import torch

from models import ModelWrapper
from prefix_words import prefix_words_to_features


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
        prefix_words_batch = []
        for s in ptb_strings:
            prefix_words = self.parse(s)
            prefix_words_batch.append(prefix_words)

        features = prefix_words_to_features(prefix_words_batch, [segment_ids] * len(prefix_words_batch), self.tokenizer,
                                            self.wordpieced, self.max_seq_length)

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
