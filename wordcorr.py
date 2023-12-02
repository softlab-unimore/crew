from abc import ABC, abstractmethod

import numpy as np

from embeddings_cache import embs_cos_sim
from utils import normalize_preserve_zero


class WordCorrelation(ABC):

    @abstractmethod
    def get_corr(self, ix_a, ix_b):
        pass

    @abstractmethod
    def get_range(self):
        pass


class EmbdngWordCorr(WordCorrelation):

    def __init__(self, words):
        self.__words = words

    def get_corr(self, ix_a, ix_b):
        wa = self.__words[ix_a]
        wb = self.__words[ix_b]
        emb_sim_ = float(embs_cos_sim([wa], [wb])[0])
        return emb_sim_

    def get_range(self):
        return np.array([-1., 1.])


class ImpactWordCorr(WordCorrelation):
    # todo: instead of using only impacts specifically, generalize to the correlation given by any kind of number
    # todo: associated to each word

    def __init__(self, impacts):
        self.impacts = normalize_preserve_zero(np.array(impacts))

    def get_corr(self, ix_a, ix_b):
        if self.impacts is not None and len(self.impacts) > 0:
            a = self.impacts[ix_a]
            b = self.impacts[ix_b]
            if a == 0 or b == 0:
                return 0
            absa, absb = abs(a), abs(b)
            signa = a / absa
            signb = b / absb
            absmax, absmin = (absa, absb) if absa > absb else (absb, absa)
            # impact_sim = signa / signb * (absa + absb) / 2 * absmin / absmax
            impact_sim = signa / signb * (absa + absb) * absmin / absmax
            return impact_sim

    def get_range(self):
        return np.array([-1., 1.])


class SchemaWordCorr(WordCorrelation):

    def __init__(self, attrs_mask, schema_scores=None):
        if schema_scores is None:
            self.__schema_scores = np.array([-0.5, 0])
        else:
            self.__schema_scores = np.array(schema_scores)
        self.__attrs_mask = attrs_mask

    def get_corr(self, ix_a, ix_b):
        attr_a = self.__attrs_mask[ix_a]
        attr_b = self.__attrs_mask[ix_b]
        # schema_scores = np.array([-0.5, 0.])
        schema_penalty_ = self.__schema_scores[1 if attr_a == attr_b else 0]  # 0. if attr_a == attr_b else -0.5
        return schema_penalty_

    def get_range(self):
        # schema_scores = np.array([-0.5, 0.])
        return self.__schema_scores
