"""Run paper experiments: MNIST, Fashion-MNIST, CIFAR-10, B=5, 2 seeds.

Usage:
    python run_paper.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import json
import time

from models import make_model
from data_utils import get_data, make_worker_loaders
from train import train_one_run

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

datasets = ["mnist", "fashionmnist", "cifar10"]
algos = ["byz-wnsgdm", "byz-ef21-wsgdm"]
gars = ["average", "waverage", "median", "wmedian", "trmean", "wtrmean", "krum", "wkrum"]
attacks = ["little", "empire"]  # attacked runs
lrs = [0.05, 0.2]
seeds = [0, 1]
f_byz = 5

total = len(datasets) * len(algos) * len(attacks) * len(gars) * len(lrs) * len(seeds)
# Plus no-attack baselines per (dataset, algo, lr, seed)
total += len(datasets) * len(algos) * len(lrs) * len(seeds)
print(f"Total configs: {total}")

results = []
count = 0
t0 = time.time()

for ds in datasets:
    print(f"\n{'='*60}")
    print(f"Loading {ds}...")
    try:
        trainset, test_loader, _, _ = get_data(ds, "./data")
    except Exception as e:
        print(f"  ERROR loading {ds}: {e}, skipping")
        continue

    # Batch sizes: 25 workers, 20 honest (1,2,4,8,16 ×4 each), 5 Byzantine (bs=4)
    batch_sizes = [1]*4 + [2]*4 + [4]*4 + [8]*4 + [16]*4 + [4]*5

    for algo in algos:
        for att in attacks:
            for gar in gars:
                for lr in lrs:
                    for seed in seeds:
                        count += 1
                        label = f"[{count}/{total}] {ds}/{algo}/{att}/{gar}/lr={lr}/s={seed}"
                        print(f"{label}...", end=" ", flush=True)

                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        model = make_model(ds, device)
                        model.train()
                        loaders = make_worker_loaders(trainset, batch_sizes, seed)

                        try:
                            accs = train_one_run(algo, model, loaders, batch_sizes,
                                                 test_loader, gar, att, f_byz, lr,
                                                 T=300, eval_every=5, device=device)
                            best = round(max(accs), 4)
                        except Exception as e:
                            print(f"ERROR: {e}")
                            best = 0.0
                            accs = [0.0]

                        results.append({
                            "algo": algo, "dataset": ds, "attack": att,
                            "gar": gar, "f_byz": f_byz, "lr": lr, "seed": seed,
                            "best_acc": best,
                            "final_acc": round(accs[-1], 4) if accs else 0.0,
                            "acc_curve": [round(a, 4) for a in accs],
                        })
                        print(f"best={best:.4f}")

        # No-attack baselines
        for algo in algos:
            for lr in lrs:
                for seed in seeds:
                    count += 1
                    label = f"[{count}/{total}] {ds}/{algo}/none/average/lr={lr}/s={seed}"
                    print(f"{label}...", end=" ", flush=True)

                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    model = make_model(ds, device)
                    model.train()
                    loaders = make_worker_loaders(trainset, batch_sizes, seed)

                    try:
                        accs = train_one_run(algo, model, loaders, batch_sizes,
                                             test_loader, "average", "none", 0, lr,
                                             T=300, eval_every=5, device=device)
                        best = round(max(accs), 4)
                    except Exception as e:
                        print(f"ERROR: {e}")
                        best = 0.0
                        accs = [0.0]

                    results.append({
                        "algo": algo, "dataset": ds, "attack": "none",
                        "gar": "average", "f_byz": 0, "lr": lr, "seed": seed,
                        "best_acc": best,
                        "final_acc": round(accs[-1], 4) if accs else 0.0,
                        "acc_curve": [round(a, 4) for a in accs],
                    })
                    print(f"best={best:.4f}")

elapsed = time.time() - t0
out_path = os.path.join(os.path.dirname(__file__), "results_paper.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone in {elapsed/60:.1f} min. Saved to {out_path}")
print(f"Total results: {len(results)}")