"""Non-IID (Dirichlet) rebuttal experiment runner.

Answers the reviewer's "heterogeneous non-IID" request: does batch-size-aware
weighting still help when honest clients hold label-skewed (Dirichlet) data?

Core comparison: weighted (waverage) vs uniform (average) aggregation, under
Dirichlet(alpha) partitions with alpha in {0.1, 0.5}, on MNIST + Fashion-MNIST.

Paper setting: W=25, B=5, batch sizes {1,2,4,8,16} (4 workers/group),
T=300, beta=0.9, LR {0.05, 0.2}, attacks {little, empire}, 5 seeds.

Config tuple: (algo, attack, gar, alpha, lr, seed)

Usage (one process per GPU, 8 GPUs):
    python3 run_non_iid.py --dataset mnist --gpu 0 --shard 0 --nshards 8
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
from data_utils import get_data
from noniid_utils import make_worker_loaders_noniid
from train import train_one_run

ALGOS = ["byz-wnsgdm", "byz-ef21-wsgdm"]
GARS = ["average", "waverage"]  # uniform vs weighted -- the core comparison
ATTACKS = ["little", "empire"]
ALPHAS = [0.1, 0.5]
LRS = [0.05, 0.2]
B_MAIN = 5
T = 300
EVAL_EVERY = 10
N_SEEDS = 5


def make_batch_sizes(f_byz):
    n_honest = 25 - f_byz
    per_group = max(1, n_honest // 5)
    honest = []
    for bs in [1, 2, 4, 8, 16]:
        honest.extend([bs] * min(per_group, n_honest - len(honest)))
    while len(honest) < n_honest:
        honest.append(4)
    return honest + [4] * f_byz


def enumerate_configs():
    cfg = []
    for algo in ALGOS:
        for att in ATTACKS:
            for gar in GARS:
                for alpha in ALPHAS:
                    for lr in LRS:
                        for seed in range(N_SEEDS):
                            cfg.append((algo, att, gar, alpha, lr, seed))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["mnist", "fashionmnist", "cifar10"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--t", type=int, default=T)
    args = ap.parse_args()

    T_ = args.t
    ds = args.dataset
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "results", f"non_iid_{ds}_s{args.shard}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    results = []
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        for r in results:
            done.add((r["algo"], r["attack"], r["gar"], r["alpha"], r["lr"], r["seed"]))

    def save():
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"[{ds}] device={device} shard={args.shard}/{args.nshards} "
          f"out={out_path} resume={len(done)} T={T_}", flush=True)

    try:
        trainset, test_loader, _, num_classes = get_data(ds, "./data")
    except Exception as e:
        print(f"[{ds}] data load error: {e}", flush=True)
        return

    batch_sizes = make_batch_sizes(B_MAIN)
    n_honest = len(batch_sizes) - B_MAIN

    configs = enumerate_configs()
    mine = [c for i, c in enumerate(configs) if i % args.nshards == args.shard]
    print(f"[{ds}] total configs={len(configs)}, this shard={len(mine)}", flush=True)

    t0 = time.time()
    n_done = 0
    for idx, (algo, att, gar, alpha, lr, seed) in enumerate(mine):
        key = (algo, att, gar, alpha, lr, seed)
        if key in done:
            continue
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = make_model(ds, device)
        model.train()
        try:
            loaders = make_worker_loaders_noniid(trainset, batch_sizes, seed, alpha,
                                                 n_honest, num_classes)
            accs = train_one_run(algo, model, loaders, batch_sizes, test_loader,
                                 gar, att, B_MAIN, lr, T=T_, eval_every=EVAL_EVERY,
                                 device=device)
            r = {"algo": algo, "dataset": ds, "attack": att, "gar": gar,
                 "alpha": alpha, "lr": lr, "seed": seed,
                 "best_acc": round(max(accs), 4), "final_acc": round(accs[-1], 4),
                 "acc_curve": [round(a, 4) for a in accs]}
        except Exception as e:
            print(f"  [ERR] {algo}/{att}/{gar}/a={alpha}/lr={lr}/s={seed}: {e}", flush=True)
            continue
        results.append(r)
        done.add(key)
        save()
        n_done += 1
        if n_done % 5 == 0 or n_done == 1:
            el = (time.time() - t0) / 60
            print(f"  [{n_done}] {algo}/{att}/{gar}/a={alpha}/lr={lr}/s={seed} "
                  f"best={r['best_acc']:.4f} (elapsed {el:.1f} min)", flush=True)

    el = (time.time() - t0) / 60
    print(f"[{ds}] shard {args.shard} DONE in {el:.1f} min -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
