from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import f1_score


class PostHocAccuracy:

    def __init__(self):
        self.y_true, self.y_pred = [], []

    def append(self, y_true: int, y_pred: int):
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)

    def clear(self):
        self.y_true = []
        self.y_pred = []

    def get_score(self):
        y_true = np.array(self.y_pred)
        y_pred = np.array(self.y_pred)
        return (y_true == y_pred).astype(float).mean()


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


class DegradationMetric(ABC):

    def __init__(self, degrad_pct_step=0., top_u=0):
        if degrad_pct_step > 0.:
            self.degrad_pcts = [degrad_pct_step * i for i in range(1, int(1. / degrad_pct_step) + 1)]
            self.top_u = 0
            self.degrad_idxs = None
            self.degrad_steps = self.degrad_pcts
        elif top_u > 0:
            self.degrad_pcts = None
            self.top_u = top_u
            self.degrad_idxs = [*range(1, top_u + 1)]
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

    def __init__(self, degrad_pct_step=0., top_u=0):
        super().__init__(degrad_pct_step, top_u)

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

    def __init__(self, degrad_pct_step=0., top_u=0):
        super().__init__(degrad_pct_step, top_u)

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

    def append(self, degrad_predict: DegradPredict):
        y_true = np.argmax(degrad_predict.get_degrad_probs(0, 0))
        self.true.append(y_true)
        morf_degrads, lerf_degrads = [], []

        if self.degrad_pcts:
            for p in self.degrad_pcts:
                if p > 0.:
                    degrad_probs = degrad_predict.get_degrad_probs(0, p, True)
                    degrad_pred = np.argmax(degrad_probs)
                    morf_degrads.append(degrad_pred)

                    degrad_probs = degrad_predict.get_degrad_probs(0, p, False)
                    degrad_pred = np.argmax(degrad_probs)
                    lerf_degrads.append(degrad_pred)
        elif self.degrad_idxs:
            for i in self.degrad_idxs:
                if i <= degrad_predict.size():
                    degrad_probs = degrad_predict.get_degrad_probs(i, 0, True)
                    degrad_pred = np.argmax(degrad_probs)
                    morf_degrads.append(degrad_pred)

                    degrad_probs = degrad_predict.get_degrad_probs(i, 0, False)
                    degrad_pred = np.argmax(degrad_probs)
                    lerf_degrads.append(degrad_pred)

        self.morf.append(morf_degrads)
        self.lerf.append(lerf_degrads)

    def get_score(self):
        degrad_scores = self.get_lerf_f1() - self.get_morf_f1()
        return degrad_scores.mean()

    def get_lerf_f1(self):
        if self._lerf_f1 is None:
            lerf = np.transpose(np.array(self.lerf))
            lerf_f1 = []
            true = np.array(self.true)
            for l in lerf:
                lerf_f1.append(f1_score(l, true))
            self._lerf_f1 = np.array(lerf_f1)
        return self._lerf_f1

    def get_morf_f1(self):
        if self._morf_f1 is None:
            morf = np.transpose(np.array(self.morf))
            morf_f1 = []
            true = np.array(self.true)
            for m in morf:
                morf_f1.append(f1_score(m, true))
            self._morf_f1 = np.array(morf_f1)
        return self._morf_f1
