import numpy as np
import torch

from adapters import words_to_wordpieces
from phevals import DegradPredict
from utils import flatten


class CReWDegradPredict(DegradPredict):

    def __init__(self, model, device, idxs, input_ids, attention_mask, segment_ids, wordpieced=True, tokenizer=None,
                 always_drop_idxs=None):
        self.model = model
        self.device = device
        self.idxs = idxs
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.segment_ids = segment_ids
        self.wordpieced = wordpieced
        self.tokenizer = tokenizer
        if always_drop_idxs is None:
            always_drop_idxs = []
        self.always_drop_idxs = always_drop_idxs

        self._size = len(self.idxs)

    def get_degrad_probs(self, n: int = 0, pct: float = 0., more_relv_first=True):
        # if n > 0:
        if float(n) >= pct and n >= 0:
            i = n
        # elif pct > 0.:
        elif pct >= float(n) and pct >= 0:
            i = round(pct * float(self.size()))
            # i = max(i, 1)  # degradation must be greater than 0
        else:
            return [0, 0]  # possibly raise exception
        if i == 0:
            idxs_to_drop = []
        else:
            # idxs_to_drop = np.array(flatten(self.idxs[:i]) if more_relv_first else flatten(self.idxs[-i:]))
            idxs_to_drop = np.array(self.idxs[:i] if more_relv_first else self.idxs[-i:])

        # input_ids_, attention_mask_, segment_ids_ = self._degrad(idxs_to_drop)
        #
        # input_ids_ = input_ids_.unsqueeze(0)
        # attention_mask_ = attention_mask_.unsqueeze(0)
        # segment_ids_ = segment_ids_.unsqueeze(0)
        #
        # # degrad_probs = predict(self.model, input_ids_, attention_mask_, segment_ids_)[0]
        # degrad_probs = self.model.predict(None, input_ids_, attention_mask_, segment_ids_, self.wordpieced)[0]
        # return degrad_probs
        return self._degrad_predict(idxs_to_drop)

    def size(self):
        return self._size

    def _degrad(self, idxs_to_drop):
        keep = np.array([True] * len(self.input_ids))
        if len(idxs_to_drop) > 0:
            keep[idxs_to_drop] = False
        if len(self.always_drop_idxs) > 0:
            keep[self.always_drop_idxs] = False
        input_ids_ = self.input_ids[keep]
        segments_ = self.segment_ids[keep]

        if self.wordpieced:
            attention_mask_ = self.attention_mask[keep]
            segment_ids_ = self.segment_ids[keep]
        else:
            words = input_ids_
            wordpieces, _, segment_ids_ = words_to_wordpieces(self.tokenizer, segments_, words, None)
            input_ids_ = self.tokenizer.convert_tokens_to_ids(wordpieces)
            input_ids_ = torch.tensor(input_ids_, device=self.device).long()
            attention_mask_ = torch.tensor([1] * len(input_ids_), device=self.device).long()
            segment_ids_ = torch.tensor(segment_ids_, device=self.device).long()

        return input_ids_, attention_mask_, segment_ids_

    def _degrad_predict(self, idxs_to_drop):
        input_ids_, attention_mask_, segment_ids_ = self._degrad(idxs_to_drop)

        input_ids_ = input_ids_.unsqueeze(0)
        attention_mask_ = attention_mask_.unsqueeze(0)
        segment_ids_ = segment_ids_.unsqueeze(0)

        # degrad_probs = predict(self.model, input_ids_, attention_mask_, segment_ids_)[0]
        degrad_probs = self.model.predict(None, input_ids_, attention_mask_, segment_ids_, self.wordpieced)[0]
        return degrad_probs


class CReWDegradPredictGroups(CReWDegradPredict):

    def get_degrad_probs(self, n: int = 0, pct: float = 0., more_relv_first=True):
        # if n > 0:
        if float(n) >= pct and n >= 0:
            i = n
        # elif pct > 0.:
        elif pct >= float(n) and pct >= 0:
            i = round(pct * float(self.size()))
        else:
            return [0, 0]  # possibly raise exception
        if i == 0:
            idxs_to_drop = []
        else:
            idxs_to_drop = np.array(flatten(self.idxs[:i]) if more_relv_first else flatten(self.idxs[-i:]))

        # input_ids_, attention_mask_, segment_ids_ = self._degrad(idxs_to_drop)
        #
        # input_ids_ = input_ids_.unsqueeze(0)
        # attention_mask_ = attention_mask_.unsqueeze(0)
        # segment_ids_ = segment_ids_.unsqueeze(0)
        #
        # # degrad_probs = predict(self.model, input_ids_, attention_mask_, segment_ids_)[0]
        # degrad_probs = self.model.predict(None, input_ids_, attention_mask_, segment_ids_, self.wordpieced)[0]
        # return degrad_probs
        return self._degrad_predict(idxs_to_drop)
