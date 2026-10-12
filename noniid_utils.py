"""Non-IID (Dirichlet) data partitioning for rebuttal experiments.

Adds label-skew / quantity-imbalance partitions without touching the original
i.i.d. path in data_utils.py.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils import _InfiniteSampler


class _DirichletSampler(torch.utils.data.Sampler):
    """Infinite sampler: each batch draws per-class counts ~ Multinomial(bs, dist),
    then samples those classes' indices with replacement. Implements label skew
    (the worker only ever sees its own Dirichlet class distribution)."""

    def __init__(self, class_indices, dist, bs, seed):
        self.class_indices = class_indices
        self.dist = np.asarray(dist, dtype=np.float64)
        self.bs = int(bs)
        self.seed = int(seed)

    def __iter__(self):
        rng = np.random.RandomState(self.seed)
        dist = self.dist / self.dist.sum()
        while True:
            counts = rng.multinomial(self.bs, dist)
            idxs = []
            for c, cnt in enumerate(counts):
                if cnt > 0:
                    idxs.extend(int(x) for x in rng.choice(
                        self.class_indices[c], size=int(cnt), replace=True))
            yield from idxs

    def __len__(self):
        return 2 ** 31 - 1


def make_worker_loaders_noniid(trainset, batch_sizes, seed, alpha, n_honest,
                               num_classes=10):
    """Dirichlet(alpha) partition.

    Honest workers i < n_honest each sample their own class distribution from
    Dirichlet(alpha); Byzantine workers i >= n_honest get an i.i.d. loader
    (unused -- the attacks generate gradients without sampling data).
    """
    targets = getattr(trainset, "targets", None)
    if targets is None:
        targets = [y for _, y in trainset]
    targets = np.asarray(targets)
    class_indices = [np.where(targets == c)[0] for c in range(num_classes)]
    rng = np.random.RandomState(seed)
    loaders = []
    for i, bs in enumerate(batch_sizes):
        if i < n_honest:
            dist = rng.dirichlet([alpha] * num_classes)
            sampler = _DirichletSampler(class_indices, dist, bs, seed + i)
        else:
            s = seed + abs(bs) % 10000
            sampler = _InfiniteSampler(len(trainset), s)
        loader = DataLoader(trainset, batch_size=bs, sampler=sampler)
        loaders.append(iter(loader))
    return loaders
