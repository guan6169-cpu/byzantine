"""Generate ONE combined Byzantine-fraction figure: 2x2 grid (dataset x attack),
weighted (waverage) vs unweighted (average) accuracy vs B=0/3/5/8/10.
Corresponds exactly to Table 1 in the rebuttal."""
import glob, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = []
for f in glob.glob("results/rebuttal_*.json"):
    rows.extend(json.load(open(f)))

g = defaultdict(list)
for r in rows:
    g[(r["algo"], r["dataset"], r["attack"], r["gar"], r["f_byz"], r["lr"])].append(r["best_acc"])

ALGO = "byz-wnsgdm"
DS = {"mnist": "MNIST", "fashionmnist": "Fashion-MNIST"}
ATT = {"little": "Little", "empire": "Empire"}
B_VALS = [0, 3, 5, 8, 10]
# clean (no-attack) baseline at B=0 (both lines start here)
CLEAN = {"mnist": 0.9142, "fashionmnist": 0.7642}

def mean(key):
    v = g.get(key)
    return float(np.mean(v)) if v else np.nan

def series(ds, att, weighted):
    gar = "waverage" if weighted else "average"
    ys = []
    for b in B_VALS:
        if b == 0:
            ys.append(CLEAN[ds])
        else:
            ys.append(mean((ALGO, ds, att, gar, b, 0.05)))
    return ys

fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), sharex=True)
colors_w = "#1f77b4"; colors_u = "#d62728"

for i, ds in enumerate(["mnist", "fashionmnist"]):
    for j, att in enumerate(["little", "empire"]):
        ax = axes[i][j]
        yw = series(ds, att, True)
        yu = series(ds, att, False)
        ax.plot(B_VALS, yw, "-o", color=colors_w, lw=2, ms=5, label="WAggF (weighted)")
        ax.plot(B_VALS, yu, "--s", color=colors_u, lw=1.8, ms=5, label="unweighted")
        ax.set_title(f"{DS[ds]} — {ATT[att]}", fontsize=12, fontweight="bold")
        ax.set_ylim(0.0, 1.02)
        ax.set_xticks(B_VALS)
        ax.grid(True, linestyle=":", alpha=0.5)
        if i == 1:
            ax.set_xlabel("Number of Byzantine workers B", fontsize=11)
        if j == 0:
            ax.set_ylabel("Best test accuracy", fontsize=11)
        ax.legend(fontsize=9, loc="lower left", framealpha=0.9)

fig.suptitle("Effect of the Byzantine fraction on accuracy (Byz-WNSGDM, LR = 0.05, mean over 5 seeds)",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = "figures_rebuttal/fig_bsweep_combined.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
