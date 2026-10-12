# coding: utf-8
"""
Byzantine-Robust Non-Convex Stochastic Optimization
with Different Batch Sizes and Momentum (Byz-WNSGDM / Byz-EF21-WSGDM)

Based on: RESAM (ICLR 2021) + NSGDM (ICML 2020) + EF21-SGDM (NeurIPS 2023)
"""

import argparse
import json
import math
import os
import pathlib
import random
import sys
import time

import numpy as np
import torch

from aggregators import get_aggregator
from attacks import get_attack
from experiments import make_datasets, evaluate, compute_gradient
from models import get_model

# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # Experiment
    p.add_argument("--dataset", type=str, default="mnist",
                   choices=["mnist", "fashionmnist", "cifar10"])
    p.add_argument("--model", type=str, default="mlp",
                   choices=["mlp", "cnn"])
    p.add_argument("--algorithm", type=str, default="byz-wnsgdm",
                   choices=["byz-wnsgdm", "byz-ef21-wsgdm"])
    p.add_argument("--gar", type=str, default="average",
                   choices=["average", "median", "krum", "trmean",
                            "waverage", "wmedian", "wkrum", "wtrmean"])

    # Federated
    p.add_argument("--nb-workers", type=int, default=25)
    p.add_argument("--nb-byz", type=int, default=5)
    p.add_argument("--batch-sizes", type=str, default="1,1,1,1,2,2,2,2,4,4,4,4,8,8,8,8,16,16,16,16",
                   help="Comma-separated batch sizes for honest workers")

    # Optimization
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--nb-steps", type=int, default=300)
    p.add_argument("--l2-reg", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=2.0)
    p.add_argument("--eval-delta", type=int, default=5)

    # Compression (for EF21)
    p.add_argument("--top-k", type=int, default=10)

    # Attack
    p.add_argument("--attack", type=str, default="little",
                   choices=["none", "little", "empire"])
    p.add_argument("--attack-factor", type=float, default=1.5,
                   help="Factor for Little/Empire attack")
    p.add_argument("--attack-negative", action="store_true", default=True,
                   help="Use negative factor (Little attack default)")

    # Reproducibility
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=str, default="results")

    return p.parse_args()


# ---------------------------------------------------------------------------
# WAggF weighting
# ---------------------------------------------------------------------------

def compute_waggf_weights(honest_bs, byz_bs):
    """Compute WAggF weights: p_ξ = b_ξ / sqrt(sum b_j^2)."""
    all_bs = honest_bs + byz_bs
    norm = math.sqrt(sum(b ** 2 for b in all_bs))
    weights = [b / norm for b in all_bs]
    return weights


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_run(args):
    """Single training run."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Parse batch sizes
    honest_bs = [int(x) for x in args.batch_sizes.split(",")]
    nb_honest = len(honest_bs)
    nb_byz = args.nb_byz
    byz_bs = [4] * nb_byz  # Byzantine workers use batch size 4

    # WAggF weights
    weights = compute_waggf_weights(honest_bs, byz_bs)
    honest_weights = weights[:nb_honest]

    # Datasets
    train_set, test_set = make_datasets(args.dataset, device)

    # Model
    model = get_model(args.model, args.dataset).to(device)
    loss_fn = torch.nn.NLLLoss()

    # Aggregator
    aggregator = get_aggregator(args.gar)

    # Attack
    attack_fn = get_attack(args.attack, args.attack_factor, args.attack_negative)

    # Momentum buffers
    # For Byz-WNSGDM: per-worker momentum, aggregated by GAR, then normalized
    # For Byz-EF21-WSGDM: per-worker momentum + error feedback + compression
    momentum_buffers = [torch.zeros_like(p) for p in model.parameters()]
    error_buffers = [torch.zeros_like(p) for p in model.parameters()]  # for EF21

    beta = args.momentum

    # Results
    history = {"step": [], "accuracy": [], "loss": []}

    for step in range(args.nb_steps):
        # Each honest worker computes gradient
        honest_grads = []
        honest_losses = []
        for i in range(nb_honest):
            bs = honest_bs[i]
            grad_flat, loss_val = compute_gradient(
                model, train_set, loss_fn, bs, args.l2_reg, args.grad_clip, device
            )
            honest_grads.append(grad_flat)
            honest_losses.append(loss_val)

        avg_loss = np.mean(honest_losses)

        # Apply momentum at workers
        for i, grad in enumerate(honest_grads):
            momentum_buffers = [
                beta * m + (1 - beta) * g
                for m, g in zip(momentum_buffers, _unflatten(grad, model))
            ]
            honest_grads[i] = _flatten(momentum_buffers)

        # Byzantine attack
        byz_grads = attack_fn(honest_grads, nb_byz) if args.attack != "none" else []

        # WAggF weighting
        weighted_grads = []
        for i, g in enumerate(honest_grads):
            weighted_grads.append(g * honest_weights[i])
        for i, g in enumerate(byz_grads):
            weighted_grads.append(g * weights[nb_honest + i])

        # Aggregate
        all_grads = weighted_grads  # honest + byz, all weighted
        aggregated = aggregator(all_grads, nb_byz)

        # Algorithm-specific update
        if args.algorithm == "byz-wnsgdm":
            # Normalized SGD: x = x - lr * g / ||g||
            agg_norm = aggregated.norm()
            if agg_norm > 1e-8:
                aggregated = aggregated / agg_norm
            update = args.lr * aggregated
            _apply_update(model, -update)

        elif args.algorithm == "byz-ef21-wsgdm":
            # Error feedback + compression
            # Top-K compression
            compressed = _topk_compress(aggregated, args.top_k)
            # Error feedback
            error_buffers = [e + c for e, c in zip(error_buffers, _unflatten(compressed, model))]
            update = args.lr * _flatten(error_buffers)
            _apply_update(model, -update)

        # Evaluation
        if step % args.eval_delta == 0:
            acc = evaluate(model, test_set, device)
            history["step"].append(step)
            history["accuracy"].append(acc)
            history["loss"].append(avg_loss)
            print(f"  Step {step:4d} | Loss {avg_loss:.4f} | Accuracy {acc:.4f}")

    return history


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten(tensors):
    return torch.cat([t.view(-1) for t in tensors])

def _unflatten(flat, model):
    shapes = [p.shape for p in model.parameters()]
    tensors = []
    offset = 0
    for shape in shapes:
        n = shape.numel()
        tensors.append(flat[offset:offset + n].view(shape))
        offset += n
    return tensors

def _apply_update(model, update_flat):
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.add_(update_flat[offset:offset + n].view_as(p))
        offset += n

def _topk_compress(grad, k):
    """Top-K compression: keep k largest absolute values, zero the rest."""
    flat = grad.clone().view(-1)
    if k >= flat.numel():
        return grad
    _, idx = torch.topk(flat.abs(), k)
    mask = torch.zeros_like(flat)
    mask[idx] = 1.0
    return (flat * mask).view_as(grad)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    print(f"=== Byz-{args.algorithm.upper()} ===")
    print(f"Dataset: {args.dataset} | Model: {args.model}")
    print(f"Workers: {len(args.batch_sizes.split(','))}H + {args.nb_byz}B")
    print(f"GAR: {args.gar} | Attack: {args.attack} | LR: {args.lr}")
    print()

    history = train_one_run(args)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    name = f"{args.dataset}_{args.algorithm}_{args.gar}_{args.attack}_lr{args.lr}_b{args.nb_byz}_s{args.seed}"
    path = os.path.join(args.output_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(history, f)
    print(f"\nSaved: {path}")
    print(f"Final accuracy: {history['accuracy'][-1]:.4f}")


if __name__ == "__main__":
    main()