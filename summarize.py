"""Consolidate rebuttal results into a markdown report (reviewer-comment -> result).

Usage: python3 summarize.py results/rebuttal_*.json
"""

import sys
import json
import os
import glob
from collections import defaultdict

import numpy as np

DS_NAMES = {"mnist": "MNIST", "fashionmnist": "Fashion-MNIST", "cifar10": "CIFAR-10"}
ALGO_NAMES = {"byz-wnsgdm": "Byz-WNSGDM", "byz-ef21-wsgdm": "Byz-EF21-WSGDM",
              "byrd-dbsgd": "Byrd-DBSGD", "byrd-dbsaga": "Byrd-DBSAGA",
              "byz-vr-marina-db": "Byz-VR-MARINA-DB"}


def load(paths):
    out = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        out.extend(d if isinstance(d, list) else d.get("results", []))
    return out


def agg(results):
    g = defaultdict(list)
    for r in results:
        g[(r["algo"], r["dataset"], r["attack"], r["gar"], r["f_byz"], r["lr"])].append(r)
    return g


def make_batch_sizes(fb):
    nh = 25 - fb
    pg = max(1, nh // 5)
    h = []
    for b in [1, 2, 4, 8, 16]:
        h.extend([b] * min(pg, nh - len(h)))
    while len(h) < nh:
        h.append(4)
    return h + [4] * fb


def ep2(fb):
    bs = make_batch_sizes(fb)
    s = sum(b * b for b in bs)
    p = [b / np.sqrt(s) for b in bs]
    return float(np.mean([x ** 2 for x in p[:25 - fb]]))


def main():
    files = []
    for a in sys.argv[1:]:
        files.extend(sorted(glob.glob(a)))
    if not files:
        print("no files"); return
    results = load(files)
    g = agg(results)
    ds_list = sorted(set(r["dataset"] for r in results if r["dataset"] in DS_NAMES))

    lines = []
    lines.append("# Rebuttal experimental results")
    lines.append("")
    lines.append(f"Loaded {len(results)} runs over {len(files)} shard files.\n")

    # ---- R2C1/R2C5 batch size ----
    lines.append("## R2C1/R2C5 — batch-size distribution (W=25)")
    lines.append("")
    bs5 = make_batch_sizes(5)
    lines.append("20 honest workers = 5 groups × 4, batch sizes {1,2,4,8,16}; "
                 "5 Byzantine workers, batch size 4.")
    lines.append("")
    lines.append("`" + str(bs5) + "`")
    lines.append("")

    # ---- R3C3 E[||p||^2_H] ----
    lines.append("## R3C3 — E[||p||^2_H] vs uniform baseline")
    lines.append("")
    lines.append("| B/W | B | Honest | sum b² | ‖b‖₂ | E[‖p‖²_H] | 1/(W−B) |")
    lines.append("|---|---|---|---|---|---|---|")
    for fb in [0, 3, 5, 8, 10]:
        bs = make_batch_sizes(fb)
        s = sum(b * b for b in bs)
        lines.append(f"| {fb}/25 | {fb} | {25-fb} | {s} | {np.sqrt(s):.1f} | "
                     f"{ep2(fb):.4f} | {1/(25-fb):.4f} |")
    lines.append("")
    lines.append("E[‖p‖²_H] < 1/(W−B) for every B>0 → WAggF reduces the effective "
                 "variance factor vs uniform weighting.")
    lines.append("")

    # ---- R2C2 ratio sweep ----
    lines.append("## R2C2 — Byzantine-fraction sweep (weighted vs unweighted)")
    lines.append("")
    lines.append("| Dataset | Algo | Attack | LR | B=3 wAvg/Avg | B=5 wAvg/Avg | B=8 wAvg/Avg | B=10 wAvg/Avg |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for ds in ds_list:
        for algo in ["byz-wnsgdm", "byz-ef21-wsgdm"]:
            for att in ["little", "empire"]:
                for lr in [0.05, 0.2]:
                    row = []
                    for fb in [3, 5, 8, 10]:
                        kw = g.get((algo, ds, att, "waverage", fb, lr))
                        ku = g.get((algo, ds, att, "average", fb, lr))
                        if kw and ku:
                            w = np.mean([r["best_acc"] for r in kw])
                            u = np.mean([r["best_acc"] for r in ku])
                            row.append(f"{w:.3f}/{u:.3f}")
                        else:
                            row.append("—")
                    if any(x != "—" for x in row):
                        lines.append(f"| {DS_NAMES[ds]} | {ALGO_NAMES[algo]} | {att} | {lr} | "
                                     f"{' | '.join(row)} |")
    lines.append("")

    # ---- R2C4 baselines ----
    lines.append("## R2C4 — new methods vs prior WAggF(+VR) baselines (B=5, waverage)")
    lines.append("")
    lines.append("Best accuracy mean±std over seeds.")
    lines.append("")
    for ds in ds_list:
        lines.append(f"### {DS_NAMES[ds]}")
        lines.append("")
        for att in ["little", "empire"]:
            for lr in [0.05, 0.2]:
                lines.append(f"**{att}, LR={lr}**  | Method | best acc (mean±std) |")
                lines.append("|---|---|")
                for m in ["byz-wnsgdm", "byz-ef21-wsgdm", "byz-vr-marina-db"]:
                    grp = g.get((m, ds, att, "waverage", 5, lr))
                    if grp:
                        accs = [r["best_acc"] for r in grp]
                        lines.append(f"| {ALGO_NAMES[m]} | "
                                     f"{np.mean(accs):.4f} ± {np.std(accs):.4f} |")
                lines.append("")
    lines.append("")

    # ---- R2C3/R2C6 seed reporting ----
    lines.append("## R2C3/R2C6 — 5 independent seeds, mean±std")
    lines.append("")
    lines.append("Main figures (fig_main_*) aggregate 5 seeds and plot mean with "
                 "±1 std shaded bands. See figures/ for the plots.")
    lines.append("")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "REBUTTAL_SUMMARY.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
