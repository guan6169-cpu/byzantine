"""Ablation study runner (R1C9 / R1C10): decompose Byz-WNSGDM and
Byz-EF21-WSGDM into their components and measure each module's contribution.

Byz-WNSGDM  = normalized SGD + momentum + WAggF weighting
Byz-EF21-WSGDM = SGD + momentum + EF21(compression+error feedback) + WAggF

Variants (8) x datasets(2) x lr(2) x attacks(2) x seeds(3) = 192 runs.
Sharded across GPUs via --shard / --nshards; checkpoint/resume on a JSON file.

Paper setting: W=25, T=300, beta=0.9, lr in {0.05, 0.2}.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
from models import make_model
from data_utils import get_data, make_worker_loaders
from train import train_one_run

# (algo, gar, beta, top_k_frac, label)
ABLATION = [
    ("byz-wnsgdm",     "trmean",  0.0, None, "W: NSGD (no mom, no w)"),
    ("byz-wnsgdm",     "trmean",  0.9, None, "W: + momentum"),
    ("byz-wnsgdm",     "wtrmean", 0.0, None, "W: + WAggF weight"),
    ("byz-wnsgdm",     "wtrmean", 0.9, None, "W: FULL Byz-WNSGDM"),
    ("byz-ef21-wsgdm", "trmean",  0.9, 1.0, "E: SGD+EF21 (no compr, no w)"),
    ("byz-ef21-wsgdm", "trmean",  0.9, 0.1, "E: + Top-K compression"),
    ("byz-ef21-wsgdm", "wtrmean", 0.9, 1.0, "E: + WAggF weight"),
    ("byz-ef21-wsgdm", "wtrmean", 0.9, 0.1, "E: FULL Byz-EF21-WSGDM"),
]
DATASETS = ["mnist", "fashionmnist"]
LRS = [0.05, 0.2]
ATTACKS = ["little", "empire"]
N_SEEDS = 3
T = 300
B = 5


def make_batch_sizes(f_byz):
    n_honest = 25 - f_byz
    per_group = max(1, n_honest // 5)
    honest = []
    for bs in [1, 2, 4, 8, 16]:
        honest.extend([bs] * min(per_group, n_honest - len(honest)))
    while len(honest) < n_honest:
        honest.append(4)
    return honest + [4] * f_byz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "results", f"ablation_{args.dataset}_s{args.shard}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    results = []
    done = set()
    if os.path.exists(out):
        results = json.load(open(out))
        for r in results:
            done.add((r["algo"], r["gar"], r["beta"], r["top_k_frac"],
                      r["attack"], r["lr"], r["seed"]))

    def save():
        json.dump(results, open(out, "w"), indent=2)

    trainset, test_loader, _, _ = get_data(args.dataset, "./data")
    bs = make_batch_sizes(B)

    # enumerate
    configs = []
    for (algo, gar, beta, tk, label) in ABLATION:
        for att in ATTACKS:
            for lr in LRS:
                for seed in range(N_SEEDS):
                    configs.append((algo, gar, beta, tk, label, att, lr, seed))
    mine = [c for i, c in enumerate(configs) if i % args.nshards == args.shard]
    print(f"[{args.dataset}] shard {args.shard}/{args.nshards} -> {len(mine)} runs", flush=True)

    t0 = time.time()
    for n, (algo, gar, beta, tk, label, att, lr, seed) in enumerate(mine):
        key = (algo, gar, beta, tk, att, lr, seed)
        if key in done:
            continue
        torch.manual_seed(seed); np.random.seed(seed)
        model = make_model(args.dataset, device); model.train()
        loaders = make_worker_loaders(trainset, bs, seed)
        kw = dict(beta=beta, top_k_frac=(tk if tk is not None else 0.1))
        accs = train_one_run(algo, model, loaders, bs, test_loader, gar, att, B, lr,
                             T=T, eval_every=10, device=device, **kw)
        r = {"algo": algo, "gar": gar, "beta": beta, "top_k_frac": tk,
             "label": label, "dataset": args.dataset, "attack": att, "lr": lr,
             "seed": seed, "best_acc": round(max(accs), 4),
             "final_acc": round(accs[-1], 4)}
        results.append(r); done.add(key); save()
        if (n + 1) % 4 == 0:
            el = (time.time() - t0) / 60
            print(f"  [{n+1}/{len(mine)}] {label} {att}/lr{lr}/s{seed} "
                  f"best={r['best_acc']:.4f} ({el:.1f}m)", flush=True)
    print(f"[{args.dataset}] shard {args.shard} DONE, total={len(results)}", flush=True)


if __name__ == "__main__":
    main()
