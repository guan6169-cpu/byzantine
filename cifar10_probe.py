"""CIFAR-10 probe: increase T + try lr=0.1. One process, saves per-config."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from models import make_model
from data_utils import get_data, make_worker_loaders
from train import train_one_run

DS = "cifar10"
GPU = int(sys.argv[1]) if len(sys.argv) > 1 else 0
T = 1000

def make_batch_sizes(f_byz):
    n_honest = 25 - f_byz
    per_group = max(1, n_honest // 5)
    honest = []
    for bs in [1, 2, 4, 8, 16]:
        honest.extend([bs] * min(per_group, n_honest - len(honest)))
    while len(honest) < n_honest:
        honest.append(4)
    return honest + [4] * f_byz

# (algo, attack, gar, f_byz, lr) -- waverage/wtrmean are the weighted aggregators (our core); lr=0.1 validated by the diagnostic
CONFIGS = [
    ("byz-wnsgdm", "little", "waverage", 5, 0.05),
    ("byz-wnsgdm", "empire", "waverage", 5, 0.05),
    ("byz-wnsgdm", "little", "waverage", 5, 0.10),
    ("byz-wnsgdm", "empire", "waverage", 5, 0.10),
    ("byz-wnsgdm", "little", "wtrmean",  5, 0.10),
    ("byz-wnsgdm", "empire", "wtrmean",  5, 0.10),
]

device = torch.device(f"cuda:{GPU}")
print(f"[probe] device={device} T={T}", flush=True)
trainset, test_loader, _, _ = get_data(DS, "./data")

out_path = "results/cifar10_T1000_probe.json"
results = []
if os.path.exists(out_path):
    results = json.load(open(out_path))
done = {(r["algo"], r["attack"], r["gar"], r["f_byz"], r["lr"]) for r in results}

for algo, att, gar, fb, lr in CONFIGS:
    if (algo, att, gar, fb, lr) in done:
        print(f"[skip] {algo}/{att}/{gar}/B={fb}/lr={lr}", flush=True)
        continue
    bs = make_batch_sizes(fb)
    torch.manual_seed(0); np.random.seed(0)
    m = make_model(DS, device); m.train()
    ld = make_worker_loaders(trainset, bs, 0)
    t0 = time.time()
    accs = train_one_run(algo, m, ld, bs, test_loader, gar, att, fb, lr,
                         T=T, eval_every=10, device=device)
    el = (time.time() - t0) / 60
    r = {"algo": algo, "attack": att, "gar": gar, "f_byz": fb, "lr": lr, "seed": 0,
         "best_acc": round(max(accs), 4), "final_acc": round(accs[-1], 4),
         "acc_curve": [round(a, 4) for a in accs]}
    results.append(r)
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"[done] {algo}/{att}/{gar}/B={fb}/lr={lr} best={r['best_acc']:.4f} "
          f"final={r['final_acc']:.4f}  ({el:.1f}min)", flush=True)

print("ALL PROBE DONE", flush=True)
