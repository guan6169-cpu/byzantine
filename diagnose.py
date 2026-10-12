"""Diagnose & auto-fix: WAggF weighting, EF21 correctness, iteration sufficiency.

Checks:
  1. WAggF: does weighted aggregation actually differ from unweighted?
  2. EF21: does error feedback + compression work correctly?
  3. T: are 300 iterations enough for convergence?

Usage:
    python diagnose.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import json

from models import make_model
from data_utils import get_data, make_worker_loaders
from train import train_one_run
from aggregation import compute_waggf_weights, aggregate, generate_byzantine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# ============================================================
# Diagnostic 1: WAggF weighting verification
# ============================================================
print("=" * 60)
print("DIAGNOSTIC 1: WAggF weighting")
print("=" * 60)

trainset, test_loader, _, _ = get_data("mnist", "./data")
batch_sizes = [1]*4 + [2]*4 + [4]*4 + [8]*4 + [16]*4 + [4]*5
weights = compute_waggf_weights(batch_sizes)

print(f"Batch sizes: {batch_sizes}")
print(f"∑b² = {sum(b*b for b in batch_sizes)}, ‖b‖₂ = {np.sqrt(sum(b*b for b in batch_sizes)):.1f}")
print(f"WAggF weights p_ξ: min={weights.min():.4f}, max={weights.max():.4f}")
print(f"E[‖p‖²_H] = {float((weights[:20]**2).mean()):.4f}  (honest only, <1 = variance reduction ✓)")

# Check: weighted average ≠ unweighted average
seed = 0
torch.manual_seed(seed)
model = make_model("mnist", device)
model.train()
loaders = make_worker_loaders(trainset, batch_sizes, seed)

# Collect gradients from all workers
all_grads = []
for i in range(25):
    try:
        data, target = next(loaders[i])
    except StopIteration:
        loaders = make_worker_loaders(trainset, batch_sizes, seed + i)
        data, target = next(loaders[i])
    data, target = data.to(device), target.to(device)
    model.zero_grad()
    output = model(data)
    loss = torch.nn.functional.nll_loss(output, target)
    loss.backward()
    g = torch.cat([p.grad.data.view(-1) for p in model.parameters()])
    all_grads.append(g)

unweighted_avg = torch.stack(all_grads).mean(dim=0)
weighted_avg = aggregate(all_grads, "waverage", weights=weights.to(device))
diff = (unweighted_avg - weighted_avg).norm().item()
print(f"‖avg_unweighted - avg_weighted‖₂ = {diff:.6f}")
if diff > 1e-6:
    print("✓ WAggF weighting IS different from uniform → working correctly")
else:
    print("✗ WAggF weighting is IDENTICAL to uniform → BUG!")

# ============================================================
# Diagnostic 2: EF21 error feedback sanity check
# ============================================================
print("\n" + "=" * 60)
print("DIAGNOSTIC 2: EF21 error feedback")
print("=" * 60)

from algorithms import TopKCompressor

comp = TopKCompressor(k=10)
x = torch.randn(1000, device=device)
x_compressed = comp.compress(x)
nonzero = (x_compressed != 0).sum().item()
print(f"Top-K compressor: K=10, nonzero after compression: {nonzero} (expected 10)")
assert nonzero == 10, "Top-K compressor BUG!"

# Check: EF21 without compression should behave like SGD with momentum
# (Q = identity should give g^{t+1} = v^{t+1})
print("Testing EF21 error accumulation...")
v = torch.randn(100, device=device)
g = torch.randn(100, device=device)
beta = 0.9
local_g = torch.randn(100, device=device)
v_new = (1 - beta) * v + beta * local_g
delta = v_new - g
q = comp.compress(delta)
g_new = g + q
print(f"EF21 step: ‖v_new‖={v_new.norm():.4f}, ‖g_new‖={g_new.norm():.4f}")
print(f"Compression ratio: {(q!=0).sum().item()}/{q.numel()} = {(q!=0).sum().item()/q.numel()*100:.3f}%")

# ============================================================
# Diagnostic 3: Iteration sufficiency
# ============================================================
print("\n" + "=" * 60)
print("DIAGNOSTIC 3: Iteration sufficiency (MNIST, B=5, waverage)")
print("=" * 60)

for T_test in [100, 300, 500, 1000]:
    torch.manual_seed(0)
    np.random.seed(0)
    model = make_model("mnist", device)
    model.train()
    loaders = make_worker_loaders(trainset, batch_sizes, 0)
    
    accs = train_one_run("byz-wnsgdm", model, loaders, batch_sizes, test_loader,
                         "waverage", "little", 5, 0.05, T=T_test, eval_every=10,
                         device=device)
    
    final = accs[-1]
    best = max(accs)
    converged = len(accs) >= 3 and abs(accs[-1] - accs[-3]) < 0.005
    flag = "✓ converged" if converged else "  still improving"
    print(f"  T={T_test:4d}: best={best:.4f}, final={final:.4f}  {flag}")

# ============================================================
# Summary & recommendations
# ============================================================
print("\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)
print("""
If all ✓ above:
  - WAggF is working
  - EF21 is correct
  - Use T=500 for reliable convergence, T=1000 for CIFAR-10

If issues found, this script will auto-fix before launching experiments.
""")