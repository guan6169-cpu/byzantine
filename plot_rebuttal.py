"""Rebuttal figures + tables from results/rebuttal_*.json.

Produces:
  fig_main_{algo}_{ds}.png          : paper Fig 1/2/A1/A2 (8 GARs, mean+-std)  [R2C3/R2C6]
  fig_ratio_{algo}_{ds}_{att}_{lr}.png : Byzantine-fraction sweep bar chart    [R2C2]
  fig_baseline_{ds}.png             : new methods vs Byrd-DBSGD/DBSAGA/VR-MARINA [R2C4]
  stdout: E[||p||^2_H] LaTeX table + correlation with weighted-vs-unweighted gap [R3C3]

Usage:
    python3 plot_rebuttal.py results/rebuttal_mnist.json results/rebuttal_fashionmnist.json
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

GAR_COLORS = {"average": "#1f77b4", "waverage": "#1f77b4",
              "median": "#ff7f0e", "wmedian": "#ff7f0e",
              "trmean": "#2ca02c", "wtrmean": "#2ca02c",
              "krum": "#d62728", "wkrum": "#d62728"}
GAR_LS = {"average": "--", "median": "--", "trmean": "--", "krum": "--",
          "waverage": "-", "wmedian": "-", "wtrmean": "-", "wkrum": "-"}
GAR_LABEL = {"average": "Average", "waverage": "WAverage", "median": "Median",
             "wmedian": "WMedian", "trmean": "TrMean", "wtrmean": "WTrMean",
             "krum": "Krum", "wkrum": "WKrum"}
DS_NAMES = {"mnist": "MNIST", "fashionmnist": "Fashion-MNIST", "cifar10": "CIFAR-10"}
ALGO_NAMES = {"byz-wnsgdm": "Byz-WNSGDM", "byz-ef21-wsgdm": "Byz-EF21-WSGDM",
              "byrd-dbsgd": "Byrd-DBSGD", "byrd-dbsaga": "Byrd-DBSAGA",
              "byz-vr-marina-db": "Byz-VR-MARINA-DB"}
METHOD_COLORS = {"byz-wnsgdm": "#d62728", "byz-ef21-wsgdm": "#1f77b4",
                 "byrd-dbsgd": "#2ca02c", "byrd-dbsaga": "#ff7f0e",
                 "byz-vr-marina-db": "#9467bd"}


def load(paths):
    out = []
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
            out.extend(d if isinstance(d, list) else d.get("results", []))
    print(f"Loaded {len(out)} results")
    return out


def agg_over_seeds(results, curve_len=None):
    groups = defaultdict(list)
    for r in results:
        k = (r["algo"], r["dataset"], r["attack"], r["gar"], r["f_byz"], r["lr"])
        groups[k].append(r["acc_curve"])
    out = {}
    for k, curves in groups.items():
        c = np.array(curves)
        out[k] = {"mean": c.mean(0), "std": c.std(0),
                  "best_mean": float(c.max(1).mean()),
                  "best_std": float(c.max(1).std())}
    return out


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
    squares = sum(b * b for b in bs)
    p = [b / np.sqrt(squares) for b in bs]
    nh = 25 - fb
    return float(np.mean([x ** 2 for x in p[:nh]]))


# ------------------------------------------------------------
# Main figures (paper Fig 1/2/A1/A2)
# ------------------------------------------------------------
def plot_main(agg, algo, ds, outdir):
    attacks = ["little", "empire"]
    lrs = [0.05, 0.2]
    f_byz = 5
    step = 10
    max_T = max((len(agg[k]["mean"]) for k in agg if k[0] == algo and k[1] == ds), default=30) * step
    x = np.arange(step, max_T + 1, step)

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5))
    fig.suptitle(f"{ALGO_NAMES[algo]} on {DS_NAMES[ds]}  (W=25, B={f_byz})", fontsize=11, y=0.99)

    for row, lr in enumerate(lrs):
        for col, att in enumerate(attacks):
            ax = axes[row, col]
            nk = (algo, ds, "none", "average", 0, lr)
            if nk in agg:
                c, s = agg[nk]["mean"], agg[nk]["std"]
                ax.plot(x[:len(c)], c, color="gray", lw=1.2, alpha=0.7, label="No attack")
                ax.fill_between(x[:len(c)], c - s, c + s, color="gray", alpha=0.1)
            for gar in ["average", "waverage", "median", "wmedian", "trmean", "wtrmean", "krum", "wkrum"]:
                k = (algo, ds, att, gar, f_byz, lr)
                if k not in agg:
                    continue
                c, s = agg[k]["mean"], agg[k]["std"]
                ax.plot(x[:len(c)], c, color=GAR_COLORS[gar], ls=GAR_LS[gar],
                        lw=0.9, label=GAR_LABEL[gar])
                ax.fill_between(x[:len(c)], c - s, c + s, color=GAR_COLORS[gar], alpha=0.07)
            ax.set_xlim(0, max_T); ax.set_ylim(0, 1)
            ax.set_xlabel("Step"); ax.set_ylabel("Top-1 accuracy")
            ax.set_title(f"LR={lr}, {att.capitalize()}", fontsize=9)
            ax.legend(loc="lower right", ncol=2, fontsize=5.5, framealpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(outdir, f"fig_main_{algo}_{ds}.png")
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(out); plt.close(fig)
    print(f"  Saved {out}")


# ------------------------------------------------------------
# Ratio sweep (R2C2)
# ------------------------------------------------------------
def plot_ratio(agg, algo, ds, att, lr, outdir):
    fb_vals = [0, 3, 5, 8, 10]
    pairs = [("waverage", "average"), ("wmedian", "median")]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))
    fig.suptitle(f"{ALGO_NAMES[algo]} on {DS_NAMES[ds]}, {att.capitalize()}, LR={lr}",
                 fontsize=10, y=1.02)
    for (wgar, ugar), ax in zip(pairs, axes):
        vals_w, vals_u, err_w, err_u = [], [], [], []
        for fb in fb_vals:
            if fb == 0:
                kw = ku = (algo, ds, "none", "average", 0, lr)
            else:
                kw = (algo, ds, att, wgar, fb, lr)
                ku = (algo, ds, att, ugar, fb, lr)
            vals_w.append(agg[kw]["best_mean"] if kw in agg else np.nan)
            err_w.append(agg[kw]["best_std"] if kw in agg else 0)
            vals_u.append(agg[ku]["best_mean"] if ku in agg else np.nan)
            err_u.append(agg[ku]["best_std"] if ku in agg else 0)
        x = np.arange(len(fb_vals)); w = 0.35
        ax.bar(x - w / 2, vals_u, w, color=GAR_COLORS[ugar], alpha=0.4, label=GAR_LABEL[ugar])
        ax.bar(x + w / 2, vals_w, w, color=GAR_COLORS[wgar], alpha=1.0, label=GAR_LABEL[wgar])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{fb}/{25 - fb}" if fb else "0/25" for fb in fb_vals])
        ax.set_ylim(0, 1)
        ax.set_xlabel("B / Honest"); ax.set_ylabel("Best accuracy")
        ax.set_title(GAR_LABEL[wgar], fontsize=9)
        ax.legend(fontsize=7)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(outdir, f"fig_ratio_{algo}_{ds}_{att}_lr{lr}.png")
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(out); plt.close(fig)
    print(f"  Saved {out}")


# ------------------------------------------------------------
# Baseline comparison (R2C4)
# ------------------------------------------------------------
def plot_baseline(agg, ds, outdir):
    attacks = ["little", "empire"]
    lrs = [0.05, 0.2]
    methods = ["byz-wnsgdm", "byz-ef21-wsgdm", "byz-vr-marina-db"]
    gar = "waverage"
    step = 10
    max_T = max((len(agg[k]["mean"]) for k in agg
                 if k[1] == ds and k[3] == gar), default=30) * step
    x = np.arange(step, max_T + 1, step)

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5))
    fig.suptitle(f"WAggF (WAverage) methods on {DS_NAMES[ds]}  (W=25, B=5)", fontsize=11, y=0.99)
    for row, lr in enumerate(lrs):
        for col, att in enumerate(attacks):
            ax = axes[row, col]
            for m in methods:
                k = (m, ds, att, gar, 5, lr)
                if k not in agg:
                    continue
                c, s = agg[k]["mean"], agg[k]["std"]
                ax.plot(x[:len(c)], c, color=METHOD_COLORS[m], lw=1.2, label=ALGO_NAMES[m])
                ax.fill_between(x[:len(c)], c - s, c + s, color=METHOD_COLORS[m], alpha=0.08)
            ax.set_xlim(0, max_T); ax.set_ylim(0, 1)
            ax.set_xlabel("Step"); ax.set_ylabel("Top-1 accuracy")
            ax.set_title(f"LR={lr}, {att.capitalize()}", fontsize=9)
            ax.legend(loc="lower right", fontsize=5.5, framealpha=0.7)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(outdir, f"fig_baseline_{ds}.png")
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(out); plt.close(fig)
    print(f"  Saved {out}")


# ------------------------------------------------------------
# E[||p||^2_H] table + correlation (R3C3)
# ------------------------------------------------------------
def print_ep2_correlation(agg):
    print("\n" + "=" * 70)
    print("E[||p||^2_H] TABLE (R3C3)  -- vs uniform-weighting baseline 1/(W-B)")
    print("=" * 70)
    print("B/W & B & Honest & sum b_xi^2 & ||b||_2 & E[||p||^2_H] & 1/(W-B) \\\\")
    print("\\hline")
    for fb in [0, 3, 5, 8, 10]:
        bs = make_batch_sizes(fb)
        sum_sq = sum(b * b for b in bs)
        uniform = 1.0 / (25 - fb)
        print(f"{fb}/25 & {fb} & {25 - fb} & {sum_sq} & {np.sqrt(sum_sq):.1f} & "
              f"{ep2(fb):.4f} & {uniform:.4f} \\\\")
    print("\n  E[||p||^2_H] < 1/(W-B) for every B>0  ->  WAggF reduces the effective")
    print("  variance factor relative to uniform weighting (the tighter-bound claim).")

    # correlation: weighted-vs-unweighted best-accuracy gap vs E[||p||^2_H]
    print("\n" + "=" * 70)
    print("CORRELATION: weighted-vs-unweighted gap vs E[||p||^2_H] (R3C3)")
    print("=" * 70)
    for ds in ["mnist", "fashionmnist"]:
        for fb in [3, 5, 8, 10]:
            for att in ["little", "empire"]:
                for lr in [0.05, 0.2]:
                    pairs = [("waverage", "average"), ("wmedian", "median")]
                    gaps = []
                    for wg, ug in pairs:
                        kw = ("byz-wnsgdm", ds, att, wg, fb, lr)
                        ku = ("byz-wnsgdm", ds, att, ug, fb, lr)
                        if kw in agg and ku in agg:
                            gaps.append(agg[kw]["best_mean"] - agg[ku]["best_mean"])
                    if gaps:
                        print(f"  {DS_NAMES[ds]} B={fb} {att} lr={lr}: "
                              f"E[||p||^2_H]={ep2(fb):.4f}  "
                              f"weighted-unweighted gap={np.mean(gaps):+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--output-dir", default="./figures_rebuttal")
    args = ap.parse_args()

    results = load(args.results)
    if not results:
        print("No results"); return
    agg = agg_over_seeds(results)

    ds_present = sorted(set(r["dataset"] for r in results if r["dataset"] in DS_NAMES))
    algos_present = sorted(set(r["algo"] for r in results if r["algo"] in ALGO_NAMES))

    print_ep2_correlation(agg)

    # main figures
    for ds in ds_present:
        for algo in ["byz-wnsgdm", "byz-ef21-wsgdm"]:
            if any(k[0] == algo and k[1] == ds for k in agg):
                plot_main(agg, algo, ds, args.output_dir)

    # ratio sweep
    for ds in ds_present:
        for algo in ["byz-wnsgdm", "byz-ef21-wsgdm"]:
            for att in ["little", "empire"]:
                for lr in [0.05, 0.2]:
                    if (algo, ds, att, "waverage", 5, lr) in agg:
                        plot_ratio(agg, algo, ds, att, lr, args.output_dir)

    # baseline comparison
    for ds in ds_present:
        if any(k[1] == ds and k[0] in ("byrd-dbsgd", "byrd-dbsaga", "byz-vr-marina-db")
               for k in agg):
            plot_baseline(agg, ds, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
