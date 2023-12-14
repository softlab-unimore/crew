import os
import pickle
import random
from time import time

import networkx as nx
import numpy as np
import torch


def uniquify(path, n_digits=2):
    filename, extension = os.path.splitext(path)
    counter = 0

    while True:
        path = f'{filename}_{counter:0{n_digits}d}{extension}'
        if not os.path.exists(path):
            break
        counter += 1

    return path


def normalize_preserve_zero(array: np.ndarray):
    max_ = max(array)
    min_ = min(array)
    r_max = max(abs(max_), abs(min_))
    # r_min = -r_max
    # t_max = 1
    # t_min = -1
    # if isinstance(array, np.ndarray) or isinstance(array, torch.Tensor):
    #     pass
    # else:
    #     array = np.array(array)
    # array = np.array((array - r_min) / (r_max - r_min) * (t_max - t_min) + t_min)
    return array / r_max


def entropy(labels, base=2):
    value, counts = np.unique(labels, return_counts=True)
    pct_counts = counts / counts.sum()
    pct_counts[pct_counts == 0] = 1  # 0 * log(0) -> 1 * log(1) = 0
    return -(pct_counts * np.log(pct_counts) / np.log(base)).sum()


def normalized_min_cut(graph):
    """Clusters graph nodes according to normalized minimum cut algorithm.
    All nodes must have at least 1 edge. Uses zero as decision boundary.

    Parameters
    -----------
        graph: a networkx graph to cluster

    Returns
    -----------
        vector containing -1 or 1 for every node
    References
    ----------
        J. Shi and J. Malik, *Normalized Cuts and Image Segmentation*,
        IEEE Transactions on Pattern Analysis and Machine Learning, vol. 22, pp. 888-905
    """
    m_adjacency = nx.to_numpy_array(graph)

    D = np.diag(np.sum(m_adjacency, 0))
    D_half_inv = np.diag(1.0 / np.sqrt(np.sum(m_adjacency, 0)))
    M = np.dot(D_half_inv, np.dot((D - m_adjacency), D_half_inv))

    (w, v) = np.linalg.eig(M)
    # find index of second smallest eigenvalue
    index = np.argsort(w)[1]

    v_partition = v[:, index]
    v_partition = np.sign(v_partition)
    return v_partition


def normalized_min_cut_gpu(graph):
    """Clusters graph nodes according to normalized minimum cut algorithm.
    All nodes must have at least 1 edge. Uses zero as decision boundary.

    Parameters
    -----------
        graph: a networkx graph to cluster

    Returns
    -----------
        vector containing -1 or 1 for every node
    References
    ----------
        J. Shi and J. Malik, *Normalized Cuts and Image Segmentation*,
        IEEE Transactions on Pattern Analysis and Machine Learning, vol. 22, pp. 888-905
    """
    m_adjacency = torch.tensor(nx.to_numpy_array(graph), device='cuda')

    D = torch.diag(torch.sum(m_adjacency, 0))
    D_half_inv = torch.diag(1.0 / torch.sqrt(torch.sum(m_adjacency, 0)))
    M = torch.matmul(D_half_inv, torch.matmul((D - m_adjacency), D_half_inv))

    (w, v) = torch.linalg.eig(M)
    # find index of second smallest eigenvalue
    index = torch.argsort(w.real)[1]

    v_partition = v[:, index]
    v_partition = torch.sign(v_partition.real)
    return v_partition


class _ExecutionTimeProfiler:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(_ExecutionTimeProfiler, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self._list = list()
        self._dict = dict()
        self._last_time = -1

    def reset(self):
        self._list = list()
        self._dict = dict()
        self._last_time = -1

    def _append_dict(self):
        if len(self._dict) > 0:
            self._list.append(self._dict)
        self._dict = dict()

    def start(self):
        self._append_dict()
        self._last_time = time()

    def timestep(self, key: str):
        cur_time = time()
        if key not in self._dict:
            self._dict[key] = []
        self._dict[key].append(cur_time - self._last_time)
        self._last_time = cur_time

    def get_list(self):
        self._append_dict()
        return self._list


EXEC_TIME_PROFILER = _ExecutionTimeProfiler()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def bindump(data, filepath: str, overwrite=True):
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    if os.path.exists(filepath) and not overwrite:
        return
    binfile = open(filepath, 'wb')
    pickle.dump(data, binfile)
    binfile.close()
