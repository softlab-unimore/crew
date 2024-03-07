import numpy as np
import torch

from adapters import words_to_wordpieces
from phevals import DegradPredict
from utils import flatten


class CReWDegradPredict(DegradPredict):

    def __init__(self, words, impacts, attn, segments, model, device, wordpieced=True, tokenizer=None,
                 always_drop_idxs=None):
        self.model = model
        self.device = device
        self.words = words
        self.impacts = impacts
        self.attn = attn
        self.segments = segments
        self.wordpieced = wordpieced
        self.tokenizer = tokenizer

        if always_drop_idxs is None:
            always_drop_idxs = []
        self.always_drop_idxs = always_drop_idxs

        self._size = len(words)

        self.imps_argsort = np.argsort(self.impacts)
        if np.argmax(self.get_degrad_probs(0, 0)) == 1:
            self.imps_argsort = np.flip(self.imps_argsort)

    def _get_arg_to_drop(self, n, pct, more_relv_first):
        if float(n) >= pct and n >= 0:
            i = n
        elif pct >= float(n) and pct >= 0:
            i = round(pct * float(self.size()))
        else:
            return [0, 0]  # raise exception
        if i == 0:
            arg_to_drop = []
        else:
            arg_to_drop = np.array(self.imps_argsort[:i] if more_relv_first else self.imps_argsort[-i:])
        return arg_to_drop

    def get_degrad_probs(self, n: int = 0, pct: float = 0., more_relv_first=True):
        idxs_to_drop = self._get_arg_to_drop(n, pct, more_relv_first)
        input_ids_, attention_mask_, segment_ids_ = self._degrad(idxs_to_drop)

        input_ids_ = input_ids_.unsqueeze(0)
        attention_mask_ = attention_mask_.unsqueeze(0)
        segment_ids_ = segment_ids_.unsqueeze(0)
        degrad_probs = self.model.predict(None, input_ids_, attention_mask_, segment_ids_, self.wordpieced)
        degrad_probs = degrad_probs[0].detach().cpu().numpy()
        return degrad_probs

    def size(self):
        return self._size

    def _degrad(self, idxs_to_drop):
        keep = np.array([True] * len(self.words))
        if len(idxs_to_drop) > 0:
            keep[idxs_to_drop] = False
        if len(self.always_drop_idxs) > 0:
            keep[self.always_drop_idxs] = False
        words_ = self.words[keep]
        segments_ = self.segments[keep]

        if self.wordpieced:
            wordpieces = words_
            attention_mask_ = self.attn[keep]
            segment_ids_ = self.segments[keep]
        else:
            wordpieces, _, segment_ids_ = words_to_wordpieces(self.tokenizer, segments_, words_, None)
            attention_mask_ = torch.tensor([1] * len(wordpieces), device=self.device).long()
            segment_ids_ = torch.tensor(segment_ids_, device=self.device).long()
        input_ids_ = self.tokenizer.convert_tokens_to_ids(wordpieces)
        input_ids_ = torch.tensor(input_ids_, device=self.device).long()

        return input_ids_, attention_mask_, segment_ids_


class CReWDegradPredictGroups(CReWDegradPredict):

    def __init__(self, words, groups, impacts, attn, segments, model, device, wordpieced=True, tokenizer=None):
        super().__init__(words, impacts, attn, segments, model, device, wordpieced, tokenizer)
        self.groups = groups
        self._size = len(impacts)

    def _get_arg_to_drop(self, n, pct, more_relv_first):
        if float(n) >= pct and n >= 0:
            i = n
        elif pct >= float(n) and pct >= 0:
            i = round(pct * float(self.size()))
        else:
            return [0, 0]  # raise exception
        if i == 0:
            idxs_to_drop = []
        else:
            if more_relv_first:
                idxs_to_drop = np.array(flatten(self.groups[self.imps_argsort[:i]]))
            else:
                idxs_to_drop = flatten(self.groups[self.imps_argsort[-i:]])
        return idxs_to_drop
