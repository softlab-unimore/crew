from abc import ABC, abstractmethod

import numpy as np
import torch
from torcheval.metrics.functional import multiclass_f1_score

from adapters import words_to_wordpieces


class PostHocAccuracy:

    def __init__(self):
        self.X_true, self.X_pred = [], []

    def append(self, x_true: int, x_pred: int):
        self.X_true.append(x_true)
        self.X_pred.append(x_pred)

    def clear(self):
        self.X_true = []
        self.X_pred = []

    def get_score(self):
        X_true = np.array(self.X_pred)
        X_pred = np.array(self.X_pred)
        return (X_true == X_pred).astype(float).mean()


class DegradPredict(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_degrad_probs(self, n: int = 0, pct: float = 0, more_relv_first=True):
        pass

    @abstractmethod
    def size(self):
        pass


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


def flatten(arr2d):
    try:
        ret = []
        for arr1d in iter(arr2d):
            try:
                for x in iter(arr1d):
                    ret.append(x)
            except TypeError:
                ret.append(arr1d)
    except TypeError:
        ret = arr2d
    return ret


class DegradationMetric(ABC):

    def __init__(self, degrad_pct_step=0., degrad_pcts: list[float] = None, top_u=0, degrad_idxs: list[int] = None):
        if degrad_pct_step > 0.:
            self.degrad_pcts = [degrad_pct_step * i for i in range(int(1. / degrad_pct_step), 0, -1)]
            self.degrad_pcts.reverse()
            self.top_u = 0
            self.degrad_idxs = None
            self.degrad_steps = self.degrad_pcts
        elif degrad_pcts:
            self.degrad_pcts = degrad_pcts
            self.top_u = 0
            self.degrad_idxs = None
            self.degrad_steps = self.degrad_pcts
        elif top_u > 0:
            self.degrad_pcts = None
            self.top_u = top_u
            self.degrad_idxs = [*range(1, top_u + 1)]
            self.degrad_steps = self.degrad_idxs
        elif degrad_idxs:
            self.degrad_pcts = None
            self.top_u = 0
            self.degrad_idxs = degrad_idxs.sort()
            self.degrad_steps = self.degrad_idxs
        # else:
        #     # raise exception

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def append(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_score(self):
        pass

    def get_degrad_steps(self):
        return self.degrad_steps


class AOPC(DegradationMetric):

    def __init__(self, degrad_pct_step=0., degrad_pcts: list[float] = None, top_u=0, degrad_idxs: list[int] = None):
        super().__init__(degrad_pct_step, degrad_pcts, top_u, degrad_idxs)

        self.y_preds = []
        self.y_pred_probs = []
        self.y_degrad_pred_probs = []

        self.y_probs_avg = None
        self.y_degrad_probs_avgs = None

    def append(self, degrad_predict: DegradPredict, y_pred: int, pred_probs):
        self.y_preds.append(y_pred)
        self.y_pred_probs.append(float(pred_probs[y_pred]))
        y_degrad_pred_probs = []

        if self.degrad_pcts:
            for p in self.degrad_pcts:
                if p > 0.:
                    degrad_pred_probs = degrad_predict.get_degrad_probs(0, p, True)
                    y_degrad_pred_prob = float(degrad_pred_probs[y_pred])
                    y_degrad_pred_probs.append(y_degrad_pred_prob)
        elif self.degrad_idxs:
            for i in self.degrad_idxs:
                if i <= degrad_predict.size():
                    degrad_pred_probs = degrad_predict.get_degrad_probs(i, 0, True)
                    y_degrad_pred_prob = float(degrad_pred_probs[y_pred])
                else:
                    y_degrad_pred_prob = float(pred_probs[y_pred])
                y_degrad_pred_probs.append(y_degrad_pred_prob)

        self.y_degrad_pred_probs.append(y_degrad_pred_probs)

    def clear(self):
        self.y_preds = []
        self.y_pred_probs = []
        self.y_degrad_pred_probs = []

        self.y_probs_avg = None
        self.y_degrad_probs_avgs = None

    def get_score(self):
        y_probs_avg = self.get_y_probs_avg()
        y_degrad_probs_avgs = self.get_y_degrad_probs_avgs()
        ret = (len(self.degrad_steps) * y_probs_avg - y_degrad_probs_avgs.sum()) / (len(self.degrad_steps) + 1)
        return ret

    def get_y_probs_avg(self):
        if self.y_probs_avg is None:
            self.y_probs_avg = np.array(self.y_pred_probs).mean()
        return self.y_probs_avg

    def get_y_degrad_probs_avgs(self):
        if self.y_degrad_probs_avgs is None:
            self.y_degrad_probs_avgs = np.array(self.y_degrad_pred_probs).mean(axis=0)
        return self.y_degrad_probs_avgs


class DegradationScoreF1(DegradationMetric):

    def __init__(self, degrad_pct_step=0., degrad_pcts: list[float] = None, top_u=0, degrad_idxs: list[int] = None):
        super().__init__(degrad_pct_step, degrad_pcts, top_u, degrad_idxs)

        self.morf = []
        self.lerf = []
        self.true = []

        self._lerf_f1 = None
        self._morf_f1 = None

    def clear(self):
        self.morf = []
        self.lerf = []
        self.true = []

        self._lerf_f1 = None
        self._morf_f1 = None

    def append(self, degrad_predict: DegradPredict):#, y_true: int):
        y_true = np.argmax(degrad_predict.get_degrad_probs(0, 0).detach().cpu().numpy())
        self.true.append(y_true)
        morf_degrads, lerf_degrads = [], []

        if self.degrad_pcts:
            for p in self.degrad_pcts:
                if p > 0.:
                    degrad_probs = degrad_predict.get_degrad_probs(0, p, True)
                    degrad_pred = torch.argmax(degrad_probs)
                    morf_degrads.append(int(degrad_pred))

                    degrad_probs = degrad_predict.get_degrad_probs(0, p, False)
                    degrad_pred = torch.argmax(degrad_probs)
                    lerf_degrads.append(int(degrad_pred))
        elif self.degrad_idxs:
            for i in self.degrad_idxs:
                if i <= degrad_predict.size():
                    degrad_probs = degrad_predict.get_degrad_probs(i, 0, True)
                    degrad_pred = torch.argmax(degrad_probs)
                else:
                    degrad_pred = -1
                morf_degrads.append(int(degrad_pred))

                if i <= degrad_predict.size():
                    degrad_probs = degrad_predict.get_degrad_probs(i, 0, False)
                    degrad_pred = torch.argmax(degrad_probs)
                else:
                    degrad_pred = -1
                lerf_degrads.append(int(degrad_pred))

        self.morf.append(morf_degrads)
        self.lerf.append(lerf_degrads)

    def get_score(self):
        degrad_scores = self.get_lerf_f1() - self.get_morf_f1()
        return degrad_scores.mean()

    def get_lerf_f1(self):
        if self._lerf_f1 is None:
            lerf = torch.transpose(torch.tensor(self.lerf), 0, 1)
            lerf_f1 = []
            true = torch.tensor(self.true)
            for l in lerf:
                lerf_f1.append(multiclass_f1_score(l, true, num_classes=2))
            self._lerf_f1 = torch.tensor(lerf_f1).detach().cpu().numpy()
        return self._lerf_f1

    def get_morf_f1(self):
        if self._morf_f1 is None:
            morf = torch.transpose(torch.tensor(self.morf), 0, 1)
            morf_f1 = []
            true = torch.tensor(self.true)
            for m in morf:
                morf_f1.append(multiclass_f1_score(m, true, num_classes=2))
            self._morf_f1 = torch.tensor(morf_f1).detach().cpu().numpy()
        return self._morf_f1
