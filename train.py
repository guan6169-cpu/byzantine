from __future__ import annotations
"""Training and evaluation loops."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """Compute top-1 accuracy on test set."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    model.train()
    return correct / total if total > 0 else 0.0


def train_one_run(
    algorithm: str,      # "byz-wnsgdm" or "byz-ef21-wsgdm"
    model: nn.Module,
    worker_loaders: list,
    batch_sizes: list[int],
    test_loader: DataLoader,
    gar: str,
    attack: str,
    f_byz: int,
    lr: float,
    T: int = 300,
    eval_every: int = 5,
    beta: float = 0.9,
    l2_reg: float = 1e-4,
    grad_clip: float = 2.0,
    top_k_frac: float = 0.1,
    device: torch.device = None,
    verbose: bool = False,
):
    """Run one training session. Returns list of accuracies."""
    from algorithms import ByzWNSGDM, ByzEF21WSGDM
    
    n_workers = len(batch_sizes)
    n_honest = n_workers - f_byz
    
    if algorithm == "byz-wnsgdm":
        trainer = ByzWNSGDM(model, beta=beta, l2_reg=l2_reg, grad_clip=grad_clip)
        trainer.init_momentum(worker_loaders, device)
    elif algorithm == "byz-ef21-wsgdm":
        trainer = ByzEF21WSGDM(model, beta=beta, l2_reg=l2_reg, grad_clip=grad_clip,
                               top_k_frac=top_k_frac)
        trainer.init_state(worker_loaders, n_workers, device)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    accuracies = []
    
    for t in range(T):
        if algorithm == "byz-wnsgdm":
            trainer.step(worker_loaders, batch_sizes, gar, attack, f_byz, n_honest,
                         device, lr)
        else:
            trainer.step(worker_loaders, batch_sizes, gar, attack, f_byz, n_honest,
                         n_workers, device, lr)
        
        if (t + 1) % eval_every == 0:
            acc = evaluate(model, test_loader, device)
            accuracies.append(acc)
            if verbose and (t + 1) % 50 == 0:
                print(f"  Step {t+1:3d}/{T}, accuracy={acc:.4f}")
    
    return accuracies