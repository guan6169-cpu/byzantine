"""Analyze ablation results -> clean tables (mean+-std over seeds)."""
import glob, json, sys
from collections import defaultdict
import numpy as np

DS = {"mnist": "MNIST", "fashionmnist": "Fashion-MNIST"}

rows = []
for f in glob.glob("results/ablation_*.json"):
    rows.extend(json.load(open(f)))

print(f"loaded {len(rows)} ablation runs\n")

# group by (label, dataset, attack, lr) -> best_acc list
g = defaultdict(list)
for r in rows:
    g[(r["label"], r["dataset"], r["attack"], r["lr"])].append(r["best_acc"])

def fmt(key):
    v = g.get(key)
    if not v:
        return "   -    "
    return f"{np.mean(v):.4f}±{np.std(v):.3f}"

# Order: WNSGDM 4 variants then EF21 4 variants
WN = ["W: NSGD (no mom, no w)", "W: + momentum", "W: + WAggF weight", "W: FULL Byz-WNSGDM"]
EF = ["E: SGD+EF21 (no compr, no w)", "E: + Top-K compression", "E: + WAggF weight", "E: FULL Byz-EF21-WSGDM"]

for ds in ["mnist", "fashionmnist"]:
    for att in ["little", "empire"]:
        print(f"=== {DS[ds]} / {att} ===")
        print(f"{'variant':34s} | {'lr=0.05':>16s} | {'lr=0.2':>16s}")
        print("-" * 72)
        for lab in WN + EF:
            a = fmt((lab, ds, att, 0.05))
            b = fmt((lab, ds, att, 0.2))
            print(f"{lab:34s} | {a:>16s} | {b:>16s}")
        print()

# ---- key deltas (prove each module helps) ----
print("=" * 72)
print("KEY DELTAS (lr=0.05, little, mean best_acc):")
print("=" * 72)
for ds in ["mnist", "fashionmnist"]:
    # momentum: W:+momentum - W:NSGD(no mom)
    m_on = np.mean(g[("W: + momentum", ds, "little", 0.05)])
    m_off = np.mean(g[("W: NSGD (no mom, no w)", ds, "little", 0.05)])
    # weight: W:FULL - W:+momentum
    w_on = np.mean(g[("W: FULL Byz-WNSGDM", ds, "little", 0.05)])
    # compression (EF21): +compression - SGD+EF21(no compr)
    c_on = np.mean(g[("E: + Top-K compression", ds, "little", 0.05)])
    c_off = np.mean(g[("E: SGD+EF21 (no compr, no w)", ds, "little", 0.05)])
    # weight (EF21): FULL - +compression
    e_w = np.mean(g[("E: FULL Byz-EF21-WSGDM", ds, "little", 0.05)])
    print(f"{DS[ds]:14s} momentum: {m_off:.4f}->{m_on:.4f} ({m_on-m_off:+.4f}) | "
          f"WAggF(WNSGDM): {m_on:.4f}->{w_on:.4f} ({w_on-m_on:+.4f}) | "
          f"compression: {c_off:.4f}->{c_on:.4f} ({c_on-c_off:+.4f}) | "
          f"WAggF(EF21): {c_on:.4f}->{e_w:.4f} ({e_w-c_on:+.4f})")
