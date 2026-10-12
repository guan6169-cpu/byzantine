"""Extract complete Byzantine-fraction sweep (B=0,3,5,8,10) for the rebuttal table."""
import glob, json
from collections import defaultdict
import numpy as np

rows = []
for f in glob.glob("results/rebuttal_*.json"):
    rows.extend(json.load(open(f)))

g = defaultdict(list)
for r in rows:
    g[(r["algo"], r["dataset"], r["attack"], r["gar"], r["f_byz"], r["lr"])].append(r["best_acc"])

def m(key):
    v = g.get(key)
    return f"{np.mean(v):.4f}" if v else "-"

DS = {"mnist": "MNIST", "fashionmnist": "Fashion"}
for ds in ["mnist", "fashionmnist"]:
    for algo in ["byz-wnsgdm", "byz-ef21-wsgdm"]:
        for att in ["little", "empire"]:
            w = [m((algo, ds, att, "waverage", b, 0.05)) for b in [0,3,5,8,10]]
            u = [m((algo, ds, att, "average", b, 0.05)) for b in [0,3,5,8,10]]
            print(f"{DS[ds]:8s} {algo:16s} {att:6s} weighted:   " + " | ".join(w))
            print(f"{'':8s} {'':16s} {'':6s} unweighted: " + " | ".join(u))
    print()
