"""8-GAR weighted vs unweighted at B=5 (mean best_acc over 5 seeds) + lr=0.05/0.2."""
import glob, json
from collections import defaultdict
import numpy as np

rows = []
for f in glob.glob("results/rebuttal_*.json"):
    rows.extend(json.load(open(f)))

# filter main B=5, all GARs
g = defaultdict(list)
for r in rows:
    if r.get("f_byz") == 5:
        g[(r["algo"], r["dataset"], r["attack"], r["gar"], r["lr"])].append(r["best_acc"])

GARS = ["average", "waverage", "median", "wmedian", "trmean", "wtrmean", "krum", "wkrum"]
for ds in ["mnist", "fashionmnist"]:
    for att in ["little", "empire"]:
        print(f"=== {ds} / {att} / lr=0.05 ===")
        for gar in GARS:
            v = g.get(("byz-wnsgdm", ds, att, gar, 0.05))
            if v:
                print(f"  {gar:9s} {np.mean(v):.4f} ± {np.std(v):.4f}")
        print()
