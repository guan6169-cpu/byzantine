"""Quick run for plot testing: MNIST, 2 seeds, 2 GARs, 2 LR, 2 attacks, B=5 only.
Generates enough data to test plotting in ~5 minutes.
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

trainset, test_loader, _, _ = get_data("mnist", "./data")
batch_sizes = [1]*4 + [2]*4 + [4]*4 + [8]*4 + [16]*4 + [4]*5

algos = ["byz-wnsgdm", "byz-ef21-wsgdm"]
gars  = ["average", "waverage", "median", "wmedian", "trmean", "wtrmean", "krum", "wkrum"]
attacks = ["little", "empire", None]
lrs  = [0.05, 0.2]
seeds = [0, 1]
f_byz = 5

total = len(algos) * len(gars) * len(attacks) * len(lrs) * len(seeds)
print(f"Total: {total} configs")

results = []
count = 0
t0 = time.time()

for algo in algos:
    for att in attacks:
        gars_here = ["average", "waverage"] if att is None else gars
        for gar in gars_here:
            for lr in lrs:
                for seed in seeds:
                    count += 1
                    att_name = att or "none"
                    print(f"[{count}/{total}] {algo}/{att_name}/{gar}/lr={lr}/seed={seed}...",
                          end=" ", flush=True)
                    
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    model = make_model("mnist", device)
                    model.train()
                    loaders = make_worker_loaders(trainset, batch_sizes, seed)
                    
                    accs = train_one_run(
                        algo, model, loaders, batch_sizes, test_loader,
                        gar, att_name, f_byz, lr, T=300, eval_every=5,
                        device=device, verbose=False,
                    )
                    
                    results.append({
                        "algo": algo, "dataset": "mnist", "attack": att_name,
                        "gar": gar, "f_byz": f_byz, "lr": lr, "seed": seed,
                        "best_acc": round(max(accs), 4),
                        "final_acc": round(accs[-1], 4),
                        "acc_curve": [round(a, 4) for a in accs],
                    })
                    print(f"best={results[-1]['best_acc']:.4f}")

elapsed = time.time() - t0
out_path = os.path.join(os.path.dirname(__file__), "results_quick.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone in {elapsed:.0f}s. Saved to {out_path}")