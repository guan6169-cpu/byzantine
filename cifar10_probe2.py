"""CIFAR-10 probe2: more iterations (T=2000) vs larger batch size (scale=16).
Small batch {1..16} -> large batch {16..256} (the large-batch distribution used in the diagnostic).
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from models import make_model
from data_utils import get_data, make_worker_loaders
from train import train_one_run

DS = "cifar10"
GPU = int(sys.argv[1]) if len(sys.argv) > 1 else 0

def make_batch_sizes(f_byz, scale=1):
    n_honest = 25 - f_byz
    per_group = max(1, n_honest // 5)
    base = [b * scale for b in [1, 2, 4, 8, 16]]
    honest = []
    for bs in base:
        honest.extend([bs] * min(per_group, n_honest - len(honest)))
    while len(honest) < n_honest:
        honest.append(4 * scale)
    byz_batch = 4 * scale if scale == 1 else 64
    return honest + [byz_batch] * f_byz

# (scale, T, attack, gar, lr)
CONFIGS = [
    (1,  2000, "little", "waverage", 0.05),
    (1,  2000, "empire", "waverage", 0.05),
    (16, 1000, "little", "waverage", 0.05),
    (16, 1000, "empire", "waverage", 0.05),
    (16, 2000, "little", "waverage", 0.05),
    (16, 2000, "empire", "waverage", 0.05),
]

device = torch.device(f"cuda:{GPU}")
print(f"[probe2] device={device}", flush=True)
trainset, test_loader, _, _ = get_data(DS, "./data")

out_path = "results/cifar10_probe2.json"
results = []
if os.path.exists(out_path):
    results = json.load(open(out_path))
done = {(r["scale"], r["T"], r["attack"], r["gar"], r["lr"]) for r in results}

for scale, T, att, gar, lr in CONFIGS:
    if (scale, T, att, gar, lr) in done:
        print(f"[skip] scale={scale} T={T} {att}/{gar}/lr={lr}", flush=True)
        continue
    fb = 5
    bs = make_batch_sizes(fb, scale)
    torch.manual_seed(0); np.random.seed(0)
    m = make_model(DS, device); m.train()
    ld = make_worker_loaders(trainset, bs, 0)
    t0 = time.time()
    accs = train_one_run("byz-wnsgdm", m, ld, bs, test_loader, gar, att, fb, lr,
                         T=T, eval_every=10, device=device)
    el = (time.time() - t0) / 60
    r = {"algo": "byz-wnsgdm", "scale": scale, "T": T, "attack": att, "gar": gar,
         "f_byz": fb, "lr": lr, "seed": 0,
         "best_acc": round(max(accs), 4), "final_acc": round(accs[-1], 4),
         "acc_curve": [round(a, 4) for a in accs]}
    results.append(r)
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"[done] scale={scale} T={T} {att}/{gar}/lr={lr} "
          f"best={r['best_acc']:.4f} final={r['final_acc']:.4f} ({el:.1f}min)", flush=True)

print("ALL PROBE2 DONE", flush=True)
