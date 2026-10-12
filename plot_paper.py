"""Plot paper figures matching Byz-WNSGDM / Byz-EF21-WSGDM paper style.

Figure 1: MNIST, Byz-WNSGDM, 4 GAR families, 2 LR × 2 attacks
Figure 2: Fashion-MNIST, Byz-WNSGDM
Figure A1: MNIST, Byz-EF21-WSGDM
Figure A2: Fashion-MNIST, Byz-EF21-WSGDM

Usage:
    python plot_paper.py results_mnist.json results_fashionmnist.json --algo byz-wnsgdm
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ============================================================
# Styling (matches the-journal / academic paper style)
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Color scheme for GAR families (unweighted / weighted)
GAR_COLORS = {
    "average":  "#1f77b4",   # blue
    "waverage": "#1f77b4",
    "median":   "#ff7f0e",   # orange
    "wmedian":  "#ff7f0e",
    "trmean":   "#2ca02c",   # green
    "wtrmean":  "#2ca02c",
    "krum":     "#d62728",   # red
    "wkrum":    "#d62728",
}

GAR_LINESTYLE = {
    # Unweighted: dashed
    "average": "--", "median": "--", "trmean": "--", "krum": "--",
    # Weighted: solid
    "waverage": "-", "wmedian": "-", "wtrmean": "-", "wkrum": "-",
}

GAR_LABEL = {
    "average": "Average", "waverage": "WAverage",
    "median": "Median",   "wmedian": "WMedian",
    "trmean": "TrMean",   "wtrmean": "WTrMean",
    "krum": "Krum",       "wkrum": "WKrum",
}

UNWEIGHTED_GARS = ["average", "median", "trmean", "krum"]
WEIGHTED_GARS   = ["waverage", "wmedian", "wtrmean", "wkrum"]
ALL_GARS = UNWEIGHTED_GARS + WEIGHTED_GARS

ATTACK_NAMES = {"little": "Little", "empire": "Empire", "none": "No attack"}
DATASET_NAMES = {"mnist": "MNIST", "fashionmnist": "Fashion-MNIST", "cifar10": "CIFAR-10"}


# ============================================================
# Data loading
# ============================================================

def load_results(*json_paths: str) -> list[dict]:
    """Load and merge result JSON files."""
    all_results = []
    for path in json_paths:
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping")
            continue
        with open(path) as f:
            data = json.load(f)
        all_results.extend(data.get("results", data if isinstance(data, list) else []))
    print(f"Loaded {len(all_results)} result entries")
    return all_results


def filter_results(results: list[dict], algo: str = None, dataset: str = None,
                   attack: str = None, gar: str = None, f_byz: int = None,
                   lr: float = None) -> list[dict]:
    """Filter results by criteria. None = keep all."""
    out = results
    if algo is not None:
        out = [r for r in out if r["algo"] == algo]
    if dataset is not None:
        out = [r for r in out if r["dataset"] == dataset]
    if attack is not None:
        out = [r for r in out if r["attack"] == attack]
    if gar is not None:
        out = [r for r in out if r["gar"] == gar]
    if f_byz is not None:
        out = [r for r in out if r.get("f_byz") == f_byz]
    if lr is not None:
        out = [r for r in out if abs(r.get("lr", 0) - lr) < 1e-6]
    return out


def aggregate_over_seeds(results: list[dict]) -> dict:
    """Group by (algo, dataset, attack, gar, f_byz, lr) and compute mean ± std over seeds.
    
    Returns: {(algo, dataset, attack, gar, f_byz, lr): {
        "curve_mean": [...], "curve_std": [...], "final_mean": float, "final_std": float
    }}
    """
    groups = defaultdict(list)
    for r in results:
        key = (r["algo"], r["dataset"], r["attack"], r["gar"], r["f_byz"], r["lr"])
        groups[key].append(r["acc_curve"])
    
    aggregated = {}
    for key, curves in groups.items():
        curves_arr = np.array(curves)  # [n_seeds, n_evals]
        aggregated[key] = {
            "curve_mean": curves_arr.mean(axis=0).tolist(),
            "curve_std": curves_arr.std(axis=0).tolist(),
            "final_mean": float(curves_arr[:, -1].mean()),
            "final_std": float(curves_arr[:, -1].std()),
        }
    return aggregated


# ============================================================
# Plotting
# ============================================================

def plot_subplot(ax, aggregated: dict, algo: str, dataset: str, attack: str,
                 f_byz: int, lr: float, x_vals: np.ndarray, title: str = None):
    """Plot one subplot: accuracy curves for all GARs + no-attack reference."""
    
    # No-attack baseline
    noatt_key = (algo, dataset, "none", "average", f_byz, lr)
    if noatt_key in aggregated:
        curve = np.array(aggregated[noatt_key]["curve_mean"])
        std   = np.array(aggregated[noatt_key]["curve_std"])
        ax.plot(x_vals, curve, color="gray", linewidth=1.2, alpha=0.7, label="No attack")
        ax.fill_between(x_vals, curve - std, curve + std, color="gray", alpha=0.1)
    
    # Plot GARs: unweighted first (dashed), then weighted (solid)
    for gar in ALL_GARS:
        key = (algo, dataset, attack, gar, f_byz, lr)
        if key not in aggregated:
            continue
        curve = np.array(aggregated[key]["curve_mean"])
        std   = np.array(aggregated[key]["curve_std"])
        color = GAR_COLORS.get(gar, "black")
        ls    = GAR_LINESTYLE.get(gar, "-")
        label = GAR_LABEL.get(gar, gar)
        ax.plot(x_vals, curve, color=color, linestyle=ls, linewidth=1.0, label=label)
        ax.fill_between(x_vals, curve - std, curve + std, color=color, alpha=0.08)
    
    ax.set_xlim(0, x_vals[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Top-1 cross-accuracy")
    ax.yaxis.set_major_locator(MaxNLocator(5))
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(loc="lower right", ncol=2, framealpha=0.8)


def make_figure(aggregated: dict, algo: str, dataset: str,
                lrs: list[float], attacks: list[str], f_byz: int = 5,
                output_path: str = None):
    """Create a 2×2 figure: rows=LR, cols=attack.
    
    Matches paper Figure 1/2 (Byz-WNSGDM) and A1/A2 (Byz-EF21-WSGDM).
    """
    ds_name = DATASET_NAMES.get(dataset, dataset)
    algo_name = {"byz-wnsgdm": "Byz-WNSGDM", "byz-ef21-wsgdm": "Byz-EF21-WSGDM"}.get(algo, algo)
    
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
    fig.suptitle(f"{algo_name} on {ds_name}  (W=25, B={f_byz})", fontsize=12, y=0.98)
    
    x_vals = np.arange(5, 301, 5)  # eval every 5 steps
    
    for row, lr in enumerate(lrs):
        for col, att in enumerate(attacks):
            ax = axes[row, col]
            title = f"LR = {lr}, {ATTACK_NAMES.get(att, att)}"
            plot_subplot(ax, aggregated, algo, dataset, att, f_byz, lr, x_vals, title)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path)
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Plot paper figures")
    parser.add_argument("results", nargs="+", help="Result JSON files")
    parser.add_argument("--algo", type=str, default=None,
                        help="Algorithm filter (byz-wnsgdm / byz-ef21-wsgdm)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset filter (mnist / fashionmnist / cifar10)")
    parser.add_argument("--f-byz", type=int, default=5, help="Byzantine count")
    parser.add_argument("--output-dir", type=str, default="./figures",
                        help="Output directory")
    args = parser.parse_args()
    
    results = load_results(*args.results)
    if not results:
        print("No results found!")
        return
    
    aggregated = aggregate_over_seeds(results)
    
    algorithms = [args.algo] if args.algo else ["byz-wnsgdm", "byz-ef21-wsgdm"]
    datasets = [args.dataset] if args.dataset else ["mnist", "fashionmnist"]
    lrs = [0.05, 0.2]
    attacks = ["little", "empire"]
    
    for algo in algorithms:
        for ds in datasets:
            # Check if we have data
            sample_key = (algo, ds, attacks[0], ALL_GARS[0], args.f_byz, lrs[0])
            if sample_key not in aggregated:
                print(f"No data for {algo}/{ds}, skipping")
                continue
            
            figname = f"{algo}_{ds}_b{args.f_byz}.png"
            output_path = os.path.join(args.output_dir, figname)
            make_figure(aggregated, algo, ds, lrs, attacks, args.f_byz, output_path)
    
    # Generate Byzantine ratio bar charts
    print("\nGenerating Byzantine ratio bar charts...")
    for algo in algorithms:
        for ds in datasets:
            for att in attacks:
                for lr in lrs:
                    figname = f"bar_{algo}_{ds}_{att}_lr{lr}.png"
                    output_path = os.path.join(args.output_dir, figname)
                    plot_byzantine_ratio(aggregated, results, algo, ds, att, lr, output_path)
    
    print("Done.")


if __name__ == "__main__":
    main()