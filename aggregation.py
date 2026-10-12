from __future__ import annotations
"""WAggF-weighted aggregation rules and Byzantine attacks."""

import torch
import numpy as np


# ============================================================
# WAggF
# ============================================================

def compute_waggf_weights(batch_sizes: list[int], device: torch.device = None) -> torch.Tensor:
    """Compute WAggF weights: p_ξ = b_ξ / sqrt(∑ b_j²)."""
    squares = torch.tensor([b * b for b in batch_sizes], dtype=torch.float32)
    norm = squares.sum().sqrt()
    w = torch.tensor(batch_sizes, dtype=torch.float32) / norm
    if device is not None:
        w = w.to(device)
    return w


# ============================================================
# Weighted aggregation rules
# ============================================================

def weighted_average(grads: list[torch.Tensor], weights: torch.Tensor, **kwargs):
    """Weighted average aggregation."""
    total_w = weights.sum()
    stacked = torch.stack(grads)  # [W, d]
    return (stacked * weights.view(-1, 1)).sum(dim=0) / total_w


def weighted_median(grads: list[torch.Tensor], weights: torch.Tensor, **kwargs):
    """Coordinate-wise weighted median.
    
    Sort each coordinate across workers, then find the index where
    cumulative weight crosses half the total weight.
    """
    stacked = torch.stack(grads)  # [W, d]
    W, d = stacked.shape
    sorted_vals, sort_idx = stacked.sort(dim=0)
    sorted_weights = weights[sort_idx]  # [W, d]
    cum_weights = sorted_weights.cumsum(dim=0)
    half_total = weights.sum() / 2.0
    # Find first index where cum_weight >= half_total
    mask = cum_weights >= half_total
    # Get the first True index per coordinate
    indices = mask.int().argmax(dim=0)  # [d]
    result = sorted_vals.gather(0, indices.unsqueeze(0)).squeeze(0)
    return result


def weighted_trimmed_mean(grads: list[torch.Tensor], weights: torch.Tensor,
                          trim_frac: float = 0.1, **kwargs):
    """Weighted trimmed mean: trim by unscaled value per coordinate (removes
    Byzantine outliers), then compute the weighted mean of the kept values."""
    stacked = torch.stack(grads)  # [W, d]
    W = stacked.size(0)
    trim_count = max(1, int(W * trim_frac))
    sorted_vals, sort_idx = stacked.sort(dim=0)
    kept = sorted_vals[trim_count:W - trim_count]        # [W-2*trim, d]
    kept_weights = weights[sort_idx[trim_count:W - trim_count]]  # [W-2*trim, d]
    return (kept * kept_weights).sum(dim=0) / kept_weights.sum(dim=0)


def weighted_krum(grads: list[torch.Tensor], weights: torch.Tensor,
                  f: int = 5, **kwargs):
    """Weighted Krum: standard Krum score divided by the batch-size weight, so
    larger-batch (lower-variance) clients are preferred."""
    W = len(grads)
    stacked = torch.stack(grads)  # [W, d]
    dists = torch.cdist(stacked.unsqueeze(0), stacked.unsqueeze(0)).squeeze(0)  # [W, W]
    scores = []
    n_neighbors = W - f - 2
    for i in range(W):
        d_i = dists[i].clone()
        d_i[i] = float('inf')  # exclude self
        _, topk_idx = d_i.topk(n_neighbors, largest=False)
        score = dists[i, topk_idx].sum() / weights[i].item()
        scores.append(score)
    best = torch.tensor(scores).argmin().item()
    return stacked[best]


# ============================================================
# Non-weighted (classical) aggregation rules
# ============================================================

def average(grads: list[torch.Tensor], **kwargs):
    """Classical average aggregation."""
    return torch.stack(grads).mean(dim=0)


def median(grads: list[torch.Tensor], **kwargs):
    """Coordinate-wise median."""
    stacked = torch.stack(grads)
    return stacked.median(dim=0).values


def trimmed_mean(grads: list[torch.Tensor], trim_frac: float = 0.1, **kwargs):
    """Classical trimmed mean."""
    stacked = torch.stack(grads)
    W = stacked.size(0)
    trim_count = max(1, int(W * trim_frac))
    sorted_vals, _ = stacked.sort(dim=0)
    return sorted_vals[trim_count:W - trim_count].mean(dim=0)


def krum(grads: list[torch.Tensor], f: int = 5, **kwargs):
    """Classical Krum."""
    W = len(grads)
    stacked = torch.stack(grads)
    dists = torch.cdist(stacked.unsqueeze(0), stacked.unsqueeze(0)).squeeze(0)
    scores = []
    n_neighbors = W - f - 2
    for i in range(W):
        d_i = dists[i].clone()
        d_i[i] = float('inf')
        _, topk_idx = d_i.topk(n_neighbors, largest=False)
        scores.append(dists[i, topk_idx].sum())
    return stacked[torch.tensor(scores).argmin().item()]


# Aggregation registry
AGGREGATORS = {
    "average": average,
    "median": median,
    "trmean": trimmed_mean,
    "krum": krum,
    "waverage": weighted_average,
    "wmedian": weighted_median,
    "wtrmean": weighted_trimmed_mean,
    "wkrum": weighted_krum,
}


def aggregate(grads: list[torch.Tensor], gar: str, weights: torch.Tensor = None,
              f: int = 5, **kwargs):
    """Apply the named aggregation rule."""
    fn = AGGREGATORS[gar]
    if gar.startswith("w"):
        return fn(grads, weights=weights, f=f, **kwargs)
    else:
        return fn(grads, f=f, **kwargs)


# ============================================================
# Byzantine attacks
# ============================================================

def attack_little(honest_grads: list[torch.Tensor], f_byz: int) -> list[torch.Tensor]:
    """Little attack: g_byz = avg - 1.5 * std."""
    stacked = torch.stack(honest_grads)
    avg = stacked.mean(dim=0)
    std = stacked.std(dim=0)
    byz_grad = avg - 1.5 * std
    return [byz_grad.clone() for _ in range(f_byz)]


def attack_empire(honest_grads: list[torch.Tensor], f_byz: int) -> list[torch.Tensor]:
    """Empire (Fall-of-Empires) attack: coordinated sign-flip g_byz = -c * avg,
    with c = (W - B) / B so the unweighted average is exactly neutralized and the
    non-robust aggregation collapses (while weighted/robust rules resist)."""
    stacked = torch.stack(honest_grads)
    avg = stacked.mean(dim=0)
    W = len(honest_grads) + f_byz
    c = (W - f_byz) / max(f_byz, 1)
    byz_grad = -c * avg
    return [byz_grad.clone() for _ in range(f_byz)]


ATTACKS = {
    "little": attack_little,
    "empire": attack_empire,
}


def generate_byzantine(honest_grads, attack_name, f_byz):
    """Generate Byzantine gradients. Returns empty list if attack_name is 'none'."""
    if attack_name == "none" or f_byz == 0:
        return []
    return ATTACKS[attack_name](honest_grads, f_byz)