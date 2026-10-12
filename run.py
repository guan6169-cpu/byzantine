"""Main experiment runner.

Usage:
    python run.py                                    # run all experiments
    python run.py --dataset mnist --algo byz-wnsgdm  # single config
    python run.py --b-ratio 0.2 --dataset cifar10     # specific Byzantine ratio
"""

import argparse
import json
import os
import time
from datetime import datetime

import torch
import numpy as np

from models import make_model
from data_utils import get_data, make_worker_loaders
from train import train_one_run


# ============================================================
# Configuration
# ============================================================

# Default batch size distribution across 25 workers
# 4 workers × 5 groups = 20 honest, 5 Byzantine (batch size 4)
DEFAULT_BATCH_SIZES = [1]*4 + [2]*4 + [4]*4 + [8]*4 + [16]*4 + [4]*5

# Byzantine ratios to sweep (B/W)
BYZANTINE_RATIOS = [0.0, 0.12, 0.20, 0.32, 0.40]  # 0, 3, 5, 8, 10 out of 25

DATASETS = ["mnist", "fashionmnist", "cifar10"]
ALGORITHMS = ["byz-wnsgdm", "byz-ef21-wsgdm"]
GARS = ["average", "waverage", "median", "wmedian", "trmean", "wtrmean", "krum", "wkrum"]
ATTACKS = ["little", "empire", None]  # None = no attack
LEARNING_RATES = [0.05, 0.2]
SEEDS = [0, 1, 2, 3, 4]


def make_batch_sizes(f_byz: int) -> list[int]:
    """Generate batch size list for W=25 workers with f_byz Byzantine."""
    n_honest = 25 - f_byz
    # Distribute honest workers proportionally
    honest_sizes = []
    base_groups = [1, 2, 4, 8, 16]
    per_group = max(1, n_honest // 5)
    for bs in base_groups:
        honest_sizes.extend([bs] * per_group)
    # Trim or pad to exact n_honest
    honest_sizes = honest_sizes[:n_honest]
    while len(honest_sizes) < n_honest:
        honest_sizes.append(4)
    # Byzantine workers
    byz_sizes = [4] * f_byz
    return honest_sizes + byz_sizes


def run_experiment(
    algo: str,
    dataset: str,
    gar: str,
    attack: str,
    f_byz: int,
    lr: float,
    seed: int,
    device: torch.device,
    data_dir: str = "./data",
    verbose: bool = False,
) -> dict:
    """Run one experiment config. Returns final accuracy."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load data
    trainset, test_loader, input_dim, num_classes = get_data(dataset, data_dir)
    
    # Build model
    model = make_model(dataset, device)
    model.train()
    
    # Batch sizes
    batch_sizes = make_batch_sizes(f_byz)
    
    # Worker loaders
    worker_loaders = make_worker_loaders(trainset, batch_sizes, seed)
    
    # Train
    try:
        accs = train_one_run(
            algorithm=algo,
            model=model,
            worker_loaders=worker_loaders,
            batch_sizes=batch_sizes,
            test_loader=test_loader,
            gar=gar,
            attack=attack,
            f_byz=f_byz,
            lr=lr,
            T=300,
            eval_every=5,
            device=device,
            verbose=verbose,
        )
    except Exception as e:
        if verbose:
            print(f"  ERROR: {e}")
        accs = [0.0]
    
    final_acc = accs[-1] if accs else 0.0
    best_acc = max(accs) if accs else 0.0
    
    return {
        "algo": algo,
        "dataset": dataset,
        "gar": gar,
        "attack": attack if attack else "none",
        "f_byz": f_byz,
        "lr": lr,
        "seed": seed,
        "final_acc": round(final_acc, 4),
        "best_acc": round(best_acc, 4),
        "acc_curve": [round(a, 4) for a in accs],
    }


def main():
    parser = argparse.ArgumentParser(description="Byzantine-robust non-convex optimization")
    parser.add_argument("--dataset", type=str, default=None,
                        help=f"Dataset: {DATASETS}. Omit for all.")
    parser.add_argument("--algo", type=str, default=None,
                        help=f"Algorithm: {ALGORITHMS}. Omit for all.")
    parser.add_argument("--b-ratio", type=float, default=None,
                        help=f"Byzantine ratio (e.g., 0.2). Omit for all ratios: {BYZANTINE_RATIOS}")
    parser.add_argument("--gar", type=str, default=None,
                        help="Aggregation rule. Omit for all.")
    parser.add_argument("--attack", type=str, default=None,
                        help="Attack name. Omit for all.")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate. Omit for {0.05, 0.2}.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed. Omit for 5-seed sweep.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu.")
    parser.add_argument("--output", type=str, default="results.json",
                        help="Output JSON file.")
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Build experiment grid
    datasets = [args.dataset] if args.dataset else DATASETS
    algos = [args.algo] if args.algo else ALGORITHMS
    lrs = [args.lr] if args.lr else LEARNING_RATES
    seeds = [args.seed] if args.seed is not None else SEEDS
    
    if args.b_ratio is not None:
        b_ratios = [args.b_ratio]
    else:
        b_ratios = BYZANTINE_RATIOS
    
    if args.gar:
        gars_to_run = [args.gar]
    else:
        gars_to_run = GARS
    
    if args.attack is not None:
        attacks_to_run = [args.attack if args.attack != "none" else None]
    else:
        attacks_to_run = ATTACKS
    
    # Count total experiments
    total = (len(datasets) * len(algos) * len(b_ratios) * len(attacks_to_run)
             * len(gars_to_run) * len(lrs) * len(seeds))
    print(f"Total experiments: {total}")
    print(f"Datasets: {datasets}")
    print(f"Algorithms: {algos}")
    print(f"Byzantine ratios: {b_ratios}")
    print(f"Attacks: {attacks_to_run}")
    print(f"GARs: {gars_to_run}")
    print(f"LRs: {lrs}")
    print(f"Seeds: {seeds}")
    print("=" * 60)
    
    results = []
    count = 0
    start_time = time.time()
    
    for ds in datasets:
        for algo in algos:
            for br in b_ratios:
                f_byz = max(0, round(br * 25))
                # Skip no-attack runs with GARs other than average/waverage
                for att in attacks_to_run:
                    gars_here = ["average", "waverage"] if att is None else gars_to_run
                    for gar in gars_here:
                        for lr in lrs:
                            for seed in seeds:
                                count += 1
                                label = f"[{count}/{total}] {ds}/{algo}/B={f_byz}/{att or 'none'}/{gar}/lr={lr}/seed={seed}"
                                if args.verbose:
                                    print(label)
                                
                                r = run_experiment(algo, ds, gar, att, f_byz, lr, seed,
                                                   device, verbose=args.verbose)
                                results.append(r)
                                
                                # Progress
                                if count % 10 == 0:
                                    elapsed = time.time() - start_time
                                    eta = elapsed / count * (total - count)
                                    print(f"[{count}/{total}] elapsed={elapsed:.0f}s, "
                                          f"ETA={eta:.0f}s, "
                                          f"last_acc={r['best_acc']:.4f}")
    
    # Save results
    with open(args.output, "w") as f:
        json.dump({
            "config": {
                "datasets": datasets,
                "algorithms": algos,
                "b_ratios": b_ratios,
                "gars": gars_to_run,
                "attacks": [a or "none" for a in attacks_to_run],
                "lrs": lrs,
                "seeds": seeds,
            },
            "results": results,
        }, f, indent=2)
    
    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.0f}s. Results saved to {args.output}")
    
    # Quick summary
    import json as _json
    print_summary(results)


def print_summary(results: list):
    """Print a quick summary table."""
    print("\n" + "=" * 60)
    print("SUMMARY (best accuracy per config, averaged over seeds)")
    print("=" * 60)
    
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        key = (r["dataset"], r["algo"], r["attack"], r["gar"], r["f_byz"], r["lr"])
        groups[key].append(r["best_acc"])
    
    print(f"{'Dataset':<14} {'Algo':<18} {'Attack':<8} {'GAR':<10} {'B':>3} {'LR':>5} {'BestAcc':>8}")
    print("-" * 70)
    for key in sorted(groups.keys()):
        ds, algo, att, gar, f_byz, lr = key
        accs = groups[key]
        avg = np.mean(accs)
        print(f"{ds:<14} {algo:<18} {att:<8} {gar:<10} {f_byz:>3} {lr:>5.2f} {avg:>8.4f}")


if __name__ == "__main__":
    main()