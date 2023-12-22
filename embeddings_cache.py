from __future__ import division, absolute_import, print_function

from typing import Union

import torch

from models import ModelWrapper


class EmbeddingsCache:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(EmbeddingsCache, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.model = None
        self.cache = dict()

    def _reset(self):
        self.model = None
        self.cache = dict()

    def work_with(self, model: ModelWrapper):
        self._reset()
        self.model = model

    def get(self, s: Union[str, int]):
        e = self.cache.get(s, None)
        if e is None:
            e = self.model.get_embedding(s)
            self.cache[s] = e
        return e

    def __getitem__(self, item: Union[str, int]):
        return self.get(item)


EMBS = EmbeddingsCache()


def embs_cos_sim(X, Y, embeddings=EMBS):
    embs_X = torch.stack([embeddings[w] for w in X])
    embs_Y = torch.stack([embeddings[w] for w in Y])
    return torch.cosine_similarity(embs_X, embs_Y)
