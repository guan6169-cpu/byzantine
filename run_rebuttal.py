"""Rebuttal experiment runner — runs everything needed to answer the reviewers.

Covers:
  R2C1/R2C5 : batch-size distribution (printed + logged)
  R2C2      : Byzantine-fraction sweep  (B/W = 0/12/20/32/40%)
  R2C3/R2C6 : 5-seed mean +/- std (main B=5 figures)
  R3C3      : E[||p||^2_H] computation (printed + saved)
  R2C4      : prior-work baselines Byrd-DBSGD / Byrd-DBSAGA / Byz-VR-MARINA-DB

Paper setting: MNIST + Fashion-MNIST, W=25, T=300 iterations, LR in {0.05, 0.2},
momentum beta=0.9. (CIFAR-10 is not in the paper, so it is omitted.)

Parallelism: enumerate all configs, assign config i to shard (i % nshards == shard).
Launch 8 processes (one per GPU) with --shard 0..7 --nshards 8 to use all 8 GPUs.

Checkpoint/resume: results appended to OUT after every run; re-running skips
already-completed (algo, attack, gar, f_byz, lr, seed) tuples.

Usage (one process per GPU):
    python3 run_rebuttal.py --dataset mnist --gpu 0 --shard 0 --nshards 4
    python3 run_rebuttal.py --dataset mnist --gpu 1 --shard 1 --nshards 4
    ...
    python3 run_rebuttal.py --dataset fashionmnist --gpu 4 --shard 0 --nshards 4
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from models import make_model
from data_utils import get_data, make_worker_loaders
from train import train_one_run
from baselines import train_baseline_run

NEW_ALGOS = ["byz-wnsgdm", "byz-ef21-wsgdm"]
# Prior-work baselines for R2C4. Only Byz-VR-MARINA-DB is kept: Byrd-DBSGD and
# Byrd-DBSAGA are strongly-convex algorithms, not comparable to our non-convex
# (momentum-based) setting.
BASELINES = ["byz-vr-marina-db"]
GARS_ALL = ["average", "waverage", "median", "wmedian", "trmean", "wtrmean", "krum", "wkrum"]
GARS_SWEEP = ["average", "waverage", "median", "wmedian"]
GARS_BASE = ["waverage", "wmedian"]
ATTACKS = ["little", "empire"]
LRS = [0.05, 0.2]
T = 300
EVAL_EVERY = 10
N_SEEDS_MAIN = 5
N_SEEDS_SWEEP = 3
N_SEEDS_BASE = 3
B_MAIN = 5
B_SWEEP = [3, 8, 10]


def make_batch_sizes(f_byz):
    n_honest = 25 - f_byz
    per_group = max(1, n_honest // 5)
    honest = []
    for bs in [1, 2, 4, 8, 16]:
        honest.extend([bs] * min(per_group, n_honest - len(honest)))
    while len(honest) < n_honest:
        honest.append(4)
    return honest + [4] * f_byz


def compute_ep2(batch_sizes, f_byz):
    squares = torch.tensor([b * b for b in batch_sizes], dtype=torch.float32)
    norm_sq = squares.sum()
    p = torch.tensor(batch_sizes, dtype=torch.float32) / norm_sq.sqrt()
    n_honest = len(batch_sizes) - f_byz
    return float((p[:n_honest] ** 2).mean())


def enumerate_configs(skip_sweep=False, skip_baselines=False, skip_main=False):
    """Flat list of (kind, algo, attack, gar, f_byz, lr, seed)."""
    cfg = []
    # Part A: no-attack baselines
    for algo in NEW_ALGOS:
        for lr in LRS:
            for seed in range(N_SEEDS_MAIN):
                cfg.append(("new", algo, "none", "average", 0, lr, seed))
    # Part B: main B=5, all 8 GARs, 5 seeds
    if not skip_main:
        for algo in NEW_ALGOS:
            for att in ATTACKS:
                for gar in GARS_ALL:
                    for lr in LRS:
                        for seed in range(N_SEEDS_MAIN):
                            cfg.append(("new", algo, att, gar, B_MAIN, lr, seed))
    # Part C: Byzantine ratio sweep
    if not skip_sweep:
        for fb in B_SWEEP:
            for algo in NEW_ALGOS:
                for att in ATTACKS:
                    for gar in GARS_SWEEP:
                        for lr in LRS:
                            for seed in range(N_SEEDS_SWEEP):
                                cfg.append(("new", algo, att, gar, fb, lr, seed))
    # Part D: prior-work baselines
    if not skip_baselines:
        for algo in BASELINES:
            for att in ATTACKS:
                for gar in GARS_BASE:
                    for lr in LRS:
                        for seed in range(N_SEEDS_BASE):
                            cfg.append(("base", algo, att, gar, B_MAIN, lr, seed))
    return cfg


def run_one(kind, algo, ds, att, gar, f_byz, lr, seed, trainset, test_loader, batch_sizes, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(ds, device)
    model.train()
    if kind == "new":
        loaders = make_worker_loaders(trainset, batch_sizes, seed)
        accs = train_one_run(algo, model, loaders, batch_sizes, test_loader,
                             gar, att, f_byz, lr, T=T, eval_every=EVAL_EVERY, device=device)
    else:
        accs = train_baseline_run(algo, model, trainset, batch_sizes, test_loader,
                                  gar, att, f_byz, lr, T=T, eval_every=EVAL_EVERY,
                                  device=device, seed=seed)
    return {"algo": algo, "dataset": ds, "attack": att, "gar": gar, "f_byz": f_byz,
            "lr": lr, "seed": seed, "best_acc": round(max(accs), 4),
            "final_acc": round(accs[-1], 4),
            "acc_curve": [round(a, 4) for a in accs]}


def main():
    global T, N_SEEDS_MAIN, N_SEEDS_SWEEP, N_SEEDS_BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["mnist", "fashionmnist", "cifar10"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--t", type=int, default=T)
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--skip-main", action="store_true")
    args = ap.parse_args()

    T = args.t
    ds = args.dataset
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "results", f"rebuttal_{ds}_s{args.shard}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    results = []
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        for r in results:
            done.add((r["algo"], r["attack"], r["gar"], r["f_byz"], r["lr"], r["seed"]))

    def save():
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"[{ds}] device={device} shard={args.shard}/{args.nshards} "
          f"out={out_path} resume={len(done)} completed  T={T}", flush=True)

    print("\nE[||p||^2_H] by Byzantine fraction (R3C3):")
    for fb in [0, 3, 5, 8, 10]:
        print(f"  B/W={fb}/25  B={fb:2d}  E[||p||^2_H]={compute_ep2(make_batch_sizes(fb), fb):.4f}")

    try:
        trainset, test_loader, _, _ = get_data(ds, "./data")
    except Exception as e:
        print(f"[{ds}] data load error: {e}")
        return

    configs = enumerate_configs(skip_sweep=args.skip_sweep,
                                skip_baselines=args.skip_baselines,
                                skip_main=args.skip_main)
    mine = [c for i, c in enumerate(configs) if i % args.nshards == args.shard]
    print(f"[{ds}] total configs={len(configs)}, this shard={len(mine)}", flush=True)

    cache_bs = {}
    t0 = time.time()
    n_done = 0
    for idx, (kind, algo, att, gar, f_byz, lr, seed) in enumerate(mine):
        key = (algo, att, gar, f_byz, lr, seed)
        if key in done:
            continue
        if f_byz not in cache_bs:
            cache_bs[f_byz] = make_batch_sizes(f_byz)
        try:
            r = run_one(kind, algo, ds, att, gar, f_byz, lr, seed,
                        trainset, test_loader, cache_bs[f_byz], device)
        except Exception as e:
            print(f"  [ERR] {algo}/{att}/{gar}/B={f_byz}/lr={lr}/s={seed}: {e}", flush=True)
            continue
        results.append(r)
        done.add(key)
        save()
        n_done += 1
        if n_done % 5 == 0 or n_done == 1:
            el = (time.time() - t0) / 60
            print(f"  [{n_done}] {algo}/{att}/{gar}/B={f_byz}/lr={lr}/s={seed} "
                  f"best={r['best_acc']:.4f}  (elapsed {el:.1f} min)", flush=True)

    el = (time.time() - t0) / 60
    print(f"[{ds}] shard {args.shard} DONE in {el:.1f} min. "
          f"total results={len(results)} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
