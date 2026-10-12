"""Server experiment runner: one dataset per process.
Usage: python3 run_server.py <dataset> <gpu_id>
Example: python3 run_server.py mnist 0
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from models import make_model
from data_utils import get_data, make_worker_loaders
from train import train_one_run

ds = sys.argv[1]
gpu_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0

device = torch.device(f"cuda:{gpu_id}")
print(f"[{ds}] Device: {device}")

trainset, test_loader, _, _ = get_data(ds, "./data")

ALGOS = ["byz-wnsgdm", "byz-ef21-wsgdm"]
GARS = ["average", "waverage", "median", "wmedian", "trmean", "wtrmean", "krum", "wkrum"]
ATTACKS = ["little", "empire"]
LRS = [0.05, 0.2]
T = 500


def make_bs(fb):
    nh = 25 - fb
    pg = max(1, nh // 5)
    h = []
    for b in [1, 2, 4, 8, 16]:
        h.extend([b] * min(pg, nh - len(h)))
    while len(h) < nh:
        h.append(4)
    return h + [4] * fb


results = []

# No-attack baselines (5 seeds)
bs0 = make_bs(0)
for algo in ALGOS:
    for lr in LRS:
        for s in range(5):
            torch.manual_seed(s)
            np.random.seed(s)
            m = make_model(ds, device)
            m.train()
            ld = make_worker_loaders(trainset, bs0, s)
            accs = train_one_run(algo, m, ld, bs0, test_loader, "average", "none", 0, lr,
                                 T=T, eval_every=10, device=device)
            results.append({"algo": algo, "dataset": ds, "attack": "none", "gar": "average",
                            "f_byz": 0, "lr": lr, "seed": s,
                            "best_acc": round(max(accs), 4),
                            "acc_curve": [round(a, 4) for a in accs]})
            print(f"  [{ds}] baseline {algo} lr={lr} s={s} best={results[-1]['best_acc']:.4f}")

# Byzantine sweep + B=5 main
for fb, n_seeds in [(3, 3), (5, 5), (8, 3), (10, 3)]:
    bs = make_bs(fb)
    gars_use = GARS if fb == 5 else ["waverage", "wmedian"]
    for algo in ALGOS:
        for att in ATTACKS:
            for gar in gars_use:
                for lr in LRS:
                    for s in range(n_seeds):
                        torch.manual_seed(s)
                        np.random.seed(s)
                        m = make_model(ds, device)
                        m.train()
                        ld = make_worker_loaders(trainset, bs, s)
                        accs = train_one_run(algo, m, ld, bs, test_loader, gar, att, fb, lr,
                                             T=T, eval_every=10, device=device)
                        results.append({"algo": algo, "dataset": ds, "attack": att, "gar": gar,
                                        "f_byz": fb, "lr": lr, "seed": s,
                                        "best_acc": round(max(accs), 4),
                                        "acc_curve": [round(a, 4) for a in accs]})
                        print(f"  [{ds}] B={fb} {algo}/{att}/{gar}/lr={lr}/s={s} "
                              f"best={results[-1]['best_acc']:.4f}")

out_path = f"results/results_{ds}.json"
os.makedirs("results", exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"[{ds}] Done. {len(results)} results -> {out_path}")