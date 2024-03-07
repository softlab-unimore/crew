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

    def __init__(self, degrad_pct_step, average='micro', pos_label=1):
        super().__init__(degrad_pct_step)
        self.average = average
        self.pos_label = pos_label
        self.changed = True

        self._morf_labels_mtx = []
        self._lerf_labels_mtx = []
        self._true = []

        self._lerf_curve = []
        self._morf_curve = []

    def clear(self):
        self._morf_labels_mtx = []
        self._lerf_labels_mtx = []
        self._true = []

        self._lerf_curve = []
        self._morf_curve = []

    def append(self, degrad_predict: DegradPredict):
        y_true = np.argmax(degrad_predict.get_degrad_probs(0, 0))
        self._true.append(y_true)
        morf_labels, lerf_labels = [], []

        for p in self.degrad_pcts:
            if p > 0.:
                probs = degrad_predict.get_degrad_probs(0, p, True)
                label = np.argmax(probs)
                morf_labels.append(label)

                probs = degrad_predict.get_degrad_probs(0, p, False)
                label = np.argmax(probs)
                lerf_labels.append(label)

        self._morf_labels_mtx.append(morf_labels)
        self._lerf_labels_mtx.append(lerf_labels)
        self.changed = True

    def _curves(self):
        if self.changed:
            self._lerf_curve = self.get_degrad_curve(self._lerf_labels_mtx)
            self._morf_curve = self.get_degrad_curve(self._morf_labels_mtx)
            self.changed = False

    def get_score(self):
        degrad_score = self.get_lerf_f1() - self.get_morf_f1()
        return degrad_score.mean()

    def get_lerf_f1(self):
        self._curves()
        return self._lerf_curve

    def get_morf_f1(self):
        self._curves()
        return self._morf_curve

    def get_degrad_curve(self, degrad_labels_mtx):
        degrad_labels_mtx = np.transpose(np.array(degrad_labels_mtx))
        degrad_curve = []
        true = np.array(self._true)
        for degrad_labels in degrad_labels_mtx:
            degrad_curve.append(f1_score(degrad_labels, true, average=self.average, pos_label=self.pos_label))
        return np.array(degrad_curve)
