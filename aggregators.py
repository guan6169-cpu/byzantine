# coding: utf-8
"""
Gradient aggregation rules with WAggF weighting support.

Weighted variants (waverage, wmedian, wkrum, wtrmean) apply p_ξ scaling
BEFORE aggregation, as per Eq.(2) in the paper.
"""

import torch
import math


# ---------------------------------------------------------------------------
# Base aggregators (operate on ALREADY weighted gradients)
# ---------------------------------------------------------------------------

def _average(grads, f, **kwargs):
    """Mean aggregation."""
    return torch.stack(grads).mean(dim=0)


def _median(grads, f, **kwargs):
    """Coordinate-wise median."""
    stacked = torch.stack(grads)
    return stacked.median(dim=0).values


def _trmean(grads, f, **kwargs):
    """Coordinate-wise trimmed mean: discard f smallest and f largest per coordinate."""
    stacked = torch.stack(grads)
    n = stacked.shape[0]
    sorted_grads, _ = torch.sort(stacked, dim=0)
    if 2 * f < n:
        trimmed = sorted_grads[f:n - f]
        return trimmed.mean(dim=0)
    else:
        return stacked.mean(dim=0)


def _krum(grads, f, **kwargs):
    """
    Multi-Krum: select the n-f-2 gradients with smallest sum of squared distances
    to their n-f-2 nearest neighbors, then average them.
    """
    n = len(grads)
    if n - f - 2 <= 0:
        return torch.stack(grads).mean(dim=0)

    # Compute pairwise squared distances
    stacked = torch.stack(grads)
    dists = torch.zeros(n, n, device=stacked.device)
    for i in range(n):
        diff = stacked[i] - stacked
        dists[i] = (diff * diff).sum(dim=tuple(range(1, diff.dim())))

    # For each gradient, sum distances to its n-f-2 closest neighbors
    scores = torch.zeros(n, device=stacked.device)
    for i in range(n):
        sorted_dists, _ = torch.sort(dists[i])
        scores[i] = sorted_dists[:n - f - 2].sum()

    # Select the n-f gradients with smallest scores
    _, idx = torch.topk(scores, n - f, largest=False)
    return stacked[idx].mean(dim=0)


# ---------------------------------------------------------------------------
# Aggregator registry
# ---------------------------------------------------------------------------

def get_aggregator(name):
    """Return an aggregator function.

    Weighted variants (prefixed with 'w') apply WAggF weighting by using
    the pre-weighted gradients passed from attack.py.

    Since WAggF weighting is applied in attack.py BEFORE calling the aggregator,
    the 'w' variants are identical to their base counterparts here.
    """
    name = name.lower().replace("w", "")  # strip 'w' prefix
    registry = {
        "average": _average,
        "median": _median,
        "krum": _krum,
        "trmean": _trmean,
    }
    if name not in registry:
        raise ValueError(f"Unknown aggregator: {name}. Choices: {list(registry.keys())}")
    return registry[name]