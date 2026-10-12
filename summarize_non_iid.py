"""Summarize non-IID (Dirichlet) results: weighted vs uniform."""
import json, glob
import numpy as np
from collections import defaultdict

data = defaultdict(list)
for f in sorted(glob.glob('results/non_iid_*.json')):
    for r in json.load(open(f)):
        key = (r['dataset'], r['algo'], r['attack'], r['alpha'], r['gar'])
        data[key].append(r['best_acc'])

print(f"{'dataset':12s} {'algo':16s} {'attack':8s} {'alpha':>5s} | {'uniform':>16s} {'weighted':>16s} {'d(w-u)':>8s}")
print('-' * 92)
for ds in ['mnist', 'fashionmnist']:
    for algo in ['byz-wnsgdm', 'byz-ef21-wsgdm']:
        for att in ['little', 'empire']:
            for alpha in [0.1, 0.5]:
                u = data.get((ds, algo, att, alpha, 'average'))
                w = data.get((ds, algo, att, alpha, 'waverage'))
                if not u or not w:
                    continue
                um, us = np.mean(u), np.std(u)
                wm, ws = np.mean(w), np.std(w)
                print(f"{ds:12s} {algo:16s} {att:8s} {alpha:5.1f} | {um:6.2f}+-{us:4.2f}   {wm:6.2f}+-{ws:4.2f}   {wm-um:+7.3f}")

wins = 0; total = 0; losses = 0
for ds in ['mnist', 'fashionmnist']:
    for algo in ['byz-wnsgdm', 'byz-ef21-wsgdm']:
        for att in ['little', 'empire']:
            for alpha in [0.1, 0.5]:
                u = data.get((ds, algo, att, alpha, 'average'))
                w = data.get((ds, algo, att, alpha, 'waverage'))
                if u and w:
                    total += 1
                    d = np.mean(w) - np.mean(u)
                    if d > 0.005: wins += 1
                    elif d < -0.005: losses += 1
print(f"\nweighted beats uniform(>0.005): {wins}/{total}   loses(< -0.005): {losses}/{total}   tie: {total-wins-losses}")

# average gain grouped by alpha
for alpha in [0.1, 0.5]:
    ds_ = []
    for ds in ['mnist', 'fashionmnist']:
        for algo in ['byz-wnsgdm', 'byz-ef21-wsgdm']:
            for att in ['little', 'empire']:
                u = data.get((ds, algo, att, alpha, 'average'))
                w = data.get((ds, algo, att, alpha, 'waverage'))
                if u and w:
                    ds_.append(np.mean(w) - np.mean(u))
    print(f"alpha={alpha}: avg gain {np.mean(ds_):+.4f} (n={len(ds_)})")
