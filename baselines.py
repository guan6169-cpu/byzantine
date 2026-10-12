from __future__ import annotations
"""Prior-work baselines for R2C4 rebuttal.

Implements the WAggF-based prior methods the reviewer asked to compare against:
  - Byrd-DBSGD      : WAggF + plain SGD (no momentum, no variance reduction)
  - Byrd-DBSAGA     : WAggF + SAGA variance reduction (no momentum)
  - Byz-VR-MARINA-DB: WAggF + variance reduction via full-batch control variate
                      (loopless-SVRG / MARINA-style, periodic full-batch refresh)

All three are the "pre-existing WAggF(+VR)" baselines. The new momentum-based
Byz-WNSGDM / Byz-EF21-WSGDM (see algorithms.py) should beat them.

Reuses helpers from algorithms.py / aggregation.py. Original files untouched.
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from algorithms import compute_stochastic_gradient, gradient_clip, apply_gradient
from aggregation import compute_waggf_weights, aggregate, generate_byzantine


# ============================================================
# Helpers
# ============================================================

class _IndexedDataset(torch.utils.data.Dataset):
    """Wrap a dataset so __getitem__ also returns the global sample index (for SAGA)."""
    def __init__(self, base):
        self.base = base

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, y, i

    def __len__(self):
        return len(self.base)


def _make_worker_iters(trainset, batch_sizes, seed, indexed=False):
    """Build one iterator per worker. indexed=True also yields sample indices."""
    ds = _IndexedDataset(trainset) if indexed else trainset
    iters = []
    for bs in batch_sizes:
        g = torch.Generator()
        g.manual_seed(seed + abs(hash(bs)) % 10000)
        loader = DataLoader(ds, batch_size=bs, shuffle=True, generator=g)
        iters.append(iter(loader))
    return iters


def _next_batch(it):
    try:
        return next(it)
    except StopIteration:
        return None


def compute_full_gradient(model, trainset, device, l2_reg=1e-4, bs=1000):
    """Full-batch gradient (1/N) sum_i grad f_i(x) + l2 regularization term."""
    loader = DataLoader(trainset, batch_size=bs, shuffle=False)
    total = 0
    grads = None
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        model.zero_grad()
        out = model(data)
        nn.functional.nll_loss(out, target).backward()
        flat = torch.cat([p.grad.data.view(-1) for p in model.parameters()])
        grads = flat * data.size(0) if grads is None else grads + flat * data.size(0)
        total += data.size(0)
    grads = grads / total
    if l2_reg > 0:
        with torch.no_grad():
            l2_grad = torch.cat([(2 * l2_reg * p.data.view(-1)) for p in model.parameters()])
        grads = grads + l2_grad
    return grads


def _snapshot(model):
    return copy.deepcopy(model.state_dict())


def _load_snapshot(model, sd):
    model.load_state_dict(sd)


# ============================================================
# Baseline 1: Byrd-DBSGD  (WAggF + plain SGD)
# ============================================================

class ByrdDBSGD:
    def __init__(self, model, l2_reg=1e-4, grad_clip=2.0):
        self.model = model
        self.l2_reg = l2_reg
        self.grad_clip = grad_clip

    def step(self, worker_iters, batch_sizes, gar, attack_name, f_byz, n_honest,
             device, lr):
        honest = []
        for i in range(n_honest):
            b = _next_batch(worker_iters[i])
            if b is None:
                return None
            data, target = b[0].to(device), b[1].to(device)
            g = compute_stochastic_gradient(self.model, data, target, self.l2_reg)
            honest.append(gradient_clip(g.detach(), self.grad_clip))
        byz = generate_byzantine(honest, attack_name, f_byz)
        all_g = honest + byz
        weights = compute_waggf_weights(batch_sizes, device=device)
        g_agg = aggregate(all_g, gar, weights=weights, f=f_byz)
        apply_gradient(self.model, lr * g_agg, 1.0)
        return g_agg


# ============================================================
# Baseline 2: Byz-VR-MARINA-DB  (WAggF + variance reduction)
# ============================================================

class ByzVRMARINA_DB:
    """Variance-reduced baseline: full-batch control variate g_tilde refreshed with
    probability p (loopless-SVRG / MARINA-style). Between refreshes each worker
    sends  g_tilde + (grad_f_S(x^t) - grad_f_S(x_ref))."""

    def __init__(self, model, l2_reg=1e-4, grad_clip=2.0, p=0.1):
        self.model = model
        self.l2_reg = l2_reg
        self.grad_clip = grad_clip
        self.p = p
        self.g_tilde = None
        self.x_ref = None

    def init_control(self, trainset, device):
        self.g_tilde = compute_full_gradient(self.model, trainset, device, self.l2_reg)
        self.x_ref = _snapshot(self.model)

    def step(self, worker_iters, batch_sizes, gar, attack_name, f_byz, n_honest,
             device, lr, trainset):
        honest = []
        for i in range(n_honest):
            b = _next_batch(worker_iters[i])
            if b is None:
                return None
            data, target = b[0].to(device), b[1].to(device)
            g_cur = compute_stochastic_gradient(self.model, data, target, self.l2_reg)
            # gradient of the SAME batch at the reference point x_ref
            cur_state = _snapshot(self.model)
            _load_snapshot(self.model, self.x_ref)
            g_ref = compute_stochastic_gradient(self.model, data, target, self.l2_reg)
            _load_snapshot(self.model, cur_state)
            m = self.g_tilde + (g_cur - g_ref)
            honest.append(gradient_clip(m.detach(), self.grad_clip))

        byz = generate_byzantine(honest, attack_name, f_byz)
        all_g = honest + byz
        weights = compute_waggf_weights(batch_sizes, device=device)
        g_agg = aggregate(all_g, gar, weights=weights, f=f_byz)
        apply_gradient(self.model, lr * g_agg, 1.0)

        if torch.rand(1).item() < self.p:
            self.g_tilde = compute_full_gradient(self.model, trainset, device, self.l2_reg)
            self.x_ref = _snapshot(self.model)
        return g_agg


# ============================================================
# Baseline 3: Byrd-DBSAGA  (WAggF + SAGA variance reduction)
# ============================================================

class ByrdDBSAGA:
    """SAGA with a per-worker gradient table on CPU.

    For worker xi drawing batch S (indices known):
        g_xi = (1/|S|) sum_{i in S} [grad f_i(x) - alpha_i] + alpha_bar_xi
    with alpha_i the last stored gradient of sample i and alpha_bar_xi the mean
    of the worker's table (maintained via a running sum, O(1) per step).
    """

    def __init__(self, model, l2_reg=1e-4, grad_clip=2.0):
        self.model = model
        self.l2_reg = l2_reg
        self.grad_clip = grad_clip
        self.tables = {}       # worker_idx -> {sample_idx: cpu tensor}
        self.sum_alpha = {}    # worker_idx -> cpu tensor (running sum of table)
        self.n_alpha = {}      # worker_idx -> int (table size)

    def _per_sample_grad(self, data, target):
        self.model.zero_grad()
        out = self.model(data)
        loss = nn.functional.nll_loss(out, target)
        if self.l2_reg > 0:
            loss = loss + self.l2_reg * sum(p.pow(2).sum() for p in self.model.parameters())
        loss.backward()
        return torch.cat([p.grad.data.view(-1) for p in self.model.parameters()]).detach()

    def step(self, worker_iters_idx, batch_sizes, gar, attack_name, f_byz, n_honest,
             device, lr):
        honest = []
        for i in range(n_honest):
            b = _next_batch(worker_iters_idx[i])
            if b is None:
                return None
            data, target, idxs = b[0].to(device), b[1].to(device), b[2]
            B = data.size(0)

            if i not in self.tables:
                self.tables[i] = {}
                self.sum_alpha[i] = None
                self.n_alpha[i] = 0
            tab = self.tables[i]

            raw = []
            alpha_sum = None
            for k in range(B):
                idx = int(idxs[k].item())
                g_k = self._per_sample_grad(data[k:k + 1], target[k:k + 1])
                raw.append(g_k)
                a = tab.get(idx)
                if a is not None:
                    alpha_sum = a.to(device) if alpha_sum is None else alpha_sum + a.to(device)

            # batch gradient = (1/B) sum grad f_i (average over batch)
            batch_grad = torch.stack(raw).mean(dim=0)

            if alpha_sum is None:
                g_xi = batch_grad
            else:
                alpha_bar = (self.sum_alpha[i].to(device) / self.n_alpha[i]) \
                    if self.sum_alpha[i] is not None else torch.zeros_like(batch_grad)
                g_xi = batch_grad - alpha_sum / B + alpha_bar

            g_xi = gradient_clip(g_xi.detach(), self.grad_clip)
            honest.append(g_xi)

            # update table + running sum
            for k in range(B):
                idx = int(idxs[k].item())
                g_cpu = raw[k].cpu()
                if idx in tab:
                    if self.sum_alpha[i] is None:
                        self.sum_alpha[i] = g_cpu - tab[idx]
                    else:
                        self.sum_alpha[i] = self.sum_alpha[i] + g_cpu - tab[idx]
                else:
                    self.sum_alpha[i] = g_cpu if self.sum_alpha[i] is None \
                        else self.sum_alpha[i] + g_cpu
                    self.n_alpha[i] += 1
                tab[idx] = g_cpu

        byz = generate_byzantine(honest, attack_name, f_byz)
        all_g = honest + byz
        weights = compute_waggf_weights(batch_sizes, device=device)
        g_agg = aggregate(all_g, gar, weights=weights, f=f_byz)
        apply_gradient(self.model, lr * g_agg, 1.0)
        return g_agg


# ============================================================
# Unified training loop for baselines
# ============================================================

def train_baseline_run(baseline_name, model, trainset, batch_sizes, test_loader,
                       gar, attack, f_byz, lr, T=300, eval_every=5,
                       device=None, seed=0):
    """Train one baseline config; returns list of accuracies."""
    from train import evaluate

    n_workers = len(batch_sizes)
    n_honest = n_workers - f_byz

    if baseline_name == "byrd-dbsgd":
        trainer = ByrdDBSGD(model)
        worker_iters = _make_worker_iters(trainset, batch_sizes, seed, indexed=False)
    elif baseline_name == "byz-vr-marina-db":
        trainer = ByzVRMARINA_DB(model)
        trainer.init_control(trainset, device)
        worker_iters = _make_worker_iters(trainset, batch_sizes, seed, indexed=False)
    elif baseline_name == "byrd-dbsaga":
        trainer = ByrdDBSAGA(model)
        worker_iters = _make_worker_iters(trainset, batch_sizes, seed, indexed=True)
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")

    accuracies = []
    for t in range(T):
        if baseline_name == "byz-vr-marina-db":
            trainer.step(worker_iters, batch_sizes, gar, attack, f_byz, n_honest,
                         device, lr, trainset)
        else:
            trainer.step(worker_iters, batch_sizes, gar, attack, f_byz, n_honest,
                         device, lr)
        if (t + 1) % eval_every == 0:
            accuracies.append(evaluate(model, test_loader, device))
    return accuracies
