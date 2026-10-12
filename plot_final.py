"""Plot paper figures + E[‖p‖²_H] table from results_final.json.

Usage:
    python plot_final.py results_final.json
"""

import argparse, json, os, sys
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

# ============================================================
# Styling
# ============================================================
GAR_COLORS = {"average": "#1f77b4", "waverage": "#1f77b4",
              "median": "#ff7f0e", "wmedian": "#ff7f0e",
              "trmean": "#2ca02c", "wtrmean": "#2ca02c",
              "krum": "#d62728", "wkrum": "#d62728"}
GAR_LS = {"average": "--", "median": "--", "trmean": "--", "krum": "--",
          "waverage": "-", "wmedian": "-", "wtrmean": "-", "wkrum": "-"}
GAR_LABEL = {"average": "Avg", "waverage": "WAvg", "median": "Med", "wmedian": "WMed",
             "trmean": "TrMean", "wtrmean": "WTrM", "krum": "Krum", "wkrum": "WKrum"}
DATASET_NAMES = {"mnist": "MNIST", "fashionmnist": "Fashion-MNIST", "cifar10": "CIFAR-10"}
ALGO_NAMES = {"byz-wnsgdm": "Byz-WNSGDM", "byz-ef21-wsgdm": "Byz-EF21-WSGDM"}


def load_results(*paths):
    all_r = []
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
            all_r.extend(d.get("results", d if isinstance(d, list) else []))
    print(f"Loaded {len(all_r)} results")
    return all_r


def agg_over_seeds(results):
    groups = defaultdict(list)
    for r in results:
        k = (r["algo"], r["dataset"], r["attack"], r["gar"], r["f_byz"], r["lr"])
        groups[k].append(r["acc_curve"])
    out = {}
    for k, curves in groups.items():
        c = np.array(curves)  # [n, T_eval]
        out[k] = {"mean": c.mean(0), "std": c.std(0),
                  "best_mean": float(c.max(1).mean()),
                  "best_std": float(c.max(1).std())}
    return out


# ============================================================
# Figure 1/2: 2×2 (LR × attack)
# ============================================================
def plot_main_figure(agg, algo, ds, attacks, lrs, f_byz, outpath):
    ag_name = ALGO_NAMES.get(algo, algo)
    ds_name = DATASET_NAMES.get(ds, ds)
    
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5))
    fig.suptitle(f"{ag_name} on {ds_name}  (W=25, B={f_byz})", fontsize=11, y=0.99)
    
    step = 10  # eval_every
    max_T = 0
    for k in agg:
        if k[0]==algo and k[1]==ds and k[4]==f_byz:
            max_T = max(max_T, len(agg[k]["mean"]) * step)
    x = np.arange(step, max_T+1, step)
    
    for row, lr in enumerate(lrs):
        for col, att in enumerate(attacks):
            ax = axes[row, col]
            
            # No-attack reference
            nk = (algo, ds, "none", "average", 0, lr)
            if nk in agg:
                c, s = agg[nk]["mean"], agg[nk]["std"]
                ax.plot(x[:len(c)], c, color="gray", lw=1.2, alpha=0.7, label="No attack")
                ax.fill_between(x[:len(c)], c-s, c+s, color="gray", alpha=0.1)
            
            # All 8 GARs
            for gar in ["average","waverage","median","wmedian","trmean","wtrmean","krum","wkrum"]:
                k = (algo, ds, att, gar, f_byz, lr)
                if k not in agg: continue
                c, s = agg[k]["mean"], agg[k]["std"]
                ax.plot(x[:len(c)], c, color=GAR_COLORS[gar], ls=GAR_LS[gar],
                       lw=0.9, label=GAR_LABEL[gar])
                ax.fill_between(x[:len(c)], c-s, c+s, color=GAR_COLORS[gar], alpha=0.07)
            
            ax.set_xlim(0, max_T)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Step")
            ax.set_ylabel("Top-1 accuracy")
            ax.set_title(f"LR={lr}, {att.capitalize()}", fontsize=9)
            ax.legend(loc="lower right", ncol=2, fontsize=5.5, framealpha=0.7)
    
    plt.tight_layout(rect=[0,0,1,0.96])
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ============================================================
# Byzantine ratio bar chart
# ============================================================
def plot_ratio_bars(agg, results_raw, algo, ds, attack, lr, outpath):
    ag_name = ALGO_NAMES.get(algo, algo)
    ds_name = DATASET_NAMES.get(ds, ds)
    
    f_byz_vals = sorted(set(r["f_byz"] for r in results_raw
                           if r.get("algo")==algo and r.get("dataset")==ds
                           and r.get("attack")==attack and abs(r.get("lr",0)-lr)<1e-6
                           and r.get("gar","").startswith("w")))
    if len(f_byz_vals) < 2: return
    
    pairs = [("waverage", "average"), ("wmedian", "median"),
             ("wtrmean", "trmean"), ("wkrum", "krum")]
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.5))
    fig.suptitle(f"{ag_name} on {ds_name}, {attack.capitalize()}, LR={lr}", fontsize=10, y=0.99)
    
    for (wgar, ugar), ax in zip(pairs, axes.flat):
        vals_w, vals_u, errs_w, errs_u = [], [], [], []
        for fb in f_byz_vals:
            kw = (algo, ds, attack, wgar, fb, lr)
            ku = (algo, ds, attack, ugar, fb, lr)
            vals_w.append(agg[kw]["best_mean"] if kw in agg else 0)
            errs_w.append(agg[kw]["best_std"] if kw in agg else 0)
            vals_u.append(agg[ku]["best_mean"] if ku in agg else 0)
            errs_u.append(agg[ku]["best_std"] if ku in agg else 0)
        
        x = np.arange(len(f_byz_vals))
        w = 0.35
        ax.bar(x-w/2, vals_u, w, color=GAR_COLORS[ugar], alpha=0.4, label=GAR_LABEL[ugar])
        ax.bar(x+w/2, vals_w, w, color=GAR_COLORS[wgar], alpha=1.0, label=GAR_LABEL[wgar])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{fb}/{25-fb}" for fb in f_byz_vals])
        ax.set_ylim(0, 1)
        ax.set_xlabel("B / Honest")
        ax.set_ylabel("Best accuracy")
        ax.set_title(GAR_LABEL[wgar].replace("W",""), fontsize=9)
        ax.legend(fontsize=7)
    
    plt.tight_layout(rect=[0,0,1,0.95])
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ============================================================
# E[‖p‖²_H] table
# ============================================================
def print_ep2_table(results_raw):
    """Print LaTeX table of E[‖p‖²_H]."""
    from aggregation import compute_waggf_weights
    
    # Find all unique f_byz from any dataset
    f_byz_set = sorted(set(r["f_byz"] for r in results_raw if isinstance(r.get("f_byz"), (int,float))))
    
    # Build batch sizes
    def make_bs(fb):
        nh = 25 - fb
        pg = max(1, nh//5)
        h = []
        for b in [1,2,4,8,16]:
            h.extend([b]*min(pg, nh-len(h)))
        return h + [4]*fb
    
    print("\n" + "="*60)
    print("E[‖p‖²_H] COMPUTATION (for LaTeX)")
    print("="*60)
    print("B/W & B & Honest & $\\sum b_\\xi^2$ & $\\|b\\|_2$ & $E[\\|p\\|^2_H]$ \\\\")
    print("\\hline")
    for fb in f_byz_set:
        bs = make_bs(fb)
        nh = 25 - fb
        sum_sq = sum(b*b for b in bs)
        norm = np.sqrt(sum_sq)
        w = compute_waggf_weights(bs)
        ep2 = float((w[:nh]**2).mean())
        ratio_str = f"{fb}/{25}"
        print(f"{ratio_str} & {fb} & {nh} & {sum_sq} & {norm:.1f} & {ep2:.4f} \\\\")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    parser.add_argument("--output-dir", default="./figures")
    args = parser.parse_args()
    
    results = load_results(*args.results)
    if not results:
        print("No results!"); return
    agg = agg_over_seeds(results)
    
    lrs = [0.05, 0.2]
    attacks = ["little", "empire"]
    
    # E[‖p‖²_H] table (for reviewer R3C3)
    print_ep2_table(results)
    
    # Main figures (one per algo×dataset combination)
    for algo in ALGO_NAMES:
        for ds in ["mnist", "fashionmnist", "cifar10"]:
            for f_byz in [5]:
                sample = (algo, ds, attacks[0], "waverage", f_byz, lrs[0])
                if sample not in agg:
                    print(f"  No data for {algo}/{ds}, skip")
                    continue
                outpath = os.path.join(args.output_dir, f"{algo}_{ds}_b{f_byz}.png")
                plot_main_figure(agg, algo, ds, attacks, lrs, f_byz, outpath)
    
    # Byzantine ratio bar charts
    for algo in ALGO_NAMES:
        for ds in ["mnist", "fashionmnist", "cifar10"]:
            for att in attacks:
                for lr in lrs:
                    sample = (algo, ds, att, "waverage", 5, lr)
                    if sample not in agg: continue
                    outpath = os.path.join(args.output_dir, f"bar_{algo}_{ds}_{att}_lr{lr}.png")
                    plot_ratio_bars(agg, results, algo, ds, att, lr, outpath)
    
    print("\nDone. All figures in:", args.output_dir)

if __name__ == "__main__":
    main()