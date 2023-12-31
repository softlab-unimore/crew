from __future__ import division, absolute_import, print_function

from abc import ABC, abstractmethod
from typing import Union

import torch

from adapters import words_to_wordpieces


class ModelWrapper(ABC):

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_embedding(self, s: Union[str, int]):
        pass


class BERT4SeqClf(ModelWrapper):

    def __init__(self, model, tokenizer, device, model_embs=None):
        super().__init__(model)
        self.tokenizer = tokenizer
        self.device = device
        if model_embs:
            self.model_embs = model_embs
        else:
            self.model_embs = self.model

    def predict(self, words=None, token_ids=None, attn_mask=None, segment_ids=None, tokenized=False, batch_size=10):
        if words is not None:
            if tokenized:
                tokens = words
            else:
                tokens, segment_ids = [], []
                for w in words:
                    t, _, s = words_to_wordpieces(self.tokenizer, segment_ids, w, None)
                    tokens.append(t)
                    segment_ids.append(s)
            token_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in tokens]
            attn_mask = [[1] * len(t) for t in token_ids]
            token_ids = torch.tensor(token_ids, device=self.device)
            attn_mask = torch.tensor(attn_mask, device=self.device)
            segment_ids = torch.tensor(segment_ids, device=self.device)

        logits = []
        with torch.no_grad():
            for i in range(0, len(token_ids), batch_size):
                token_ids_ = token_ids[i: min(i + batch_size, len(token_ids))]
                attn_mask_ = attn_mask[i: min(i + batch_size, len(token_ids))]
                segment_ids_ = segment_ids[i: min(i + batch_size, len(token_ids))]
                inputs = {
                    'input_ids': token_ids_,
                    'attention_mask': attn_mask_,
                    'token_type_ids': segment_ids_,
                }
                outputs = self.model(**inputs)
                logits.append(outputs[0])
        logits = torch.cat(logits)
        logits = torch.softmax(logits, dim=1)
        return logits

    def get_embedding(self, s: Union[str, int]):
        if isinstance(s, str):
            if s.startswith('##'):
                encoding = self.tokenizer.convert_tokens_to_ids([s])
            else:
                encoding = self.tokenizer.encode(s)
        else:  # assuming it is a token id, thus an integer possibly disguised as a float
            encoding = [int(s)]
        token_ids = torch.LongTensor(encoding).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model_embs(input_ids=token_ids)
        last_hidden_state = out[0]
        embedding = torch.mean(last_hidden_state, dim=1).squeeze()
        return embedding
