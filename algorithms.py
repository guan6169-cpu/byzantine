from __future__ import annotations
"""Byz-WNSGDM and Byz-EF21-WSGDM algorithm implementations."""

import torch
import torch.nn as nn
from aggregation import compute_waggf_weights, aggregate, generate_byzantine


# ============================================================
# Top-K compressor
# ============================================================

class TopKCompressor:
    """Contractive compressor: keeps top-K coordinates by magnitude, zeros the rest."""
    
    def __init__(self, k: int):
        self.k = k

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Return x with only top-K absolute values kept."""
        if self.k >= x.numel():
            return x.clone()
        flat = x.view(-1)
        _, idx = flat.abs().topk(self.k)
        result = torch.zeros_like(flat)
        result[idx] = flat[idx]
        return result.view_as(x)


# ============================================================
# Gradient helpers
# ============================================================

def compute_stochastic_gradient(model: nn.Module, data: torch.Tensor,
                                 target: torch.Tensor, l2_reg: float = 1e-4):
    """Compute stochastic gradient on one batch.
    
    Returns flat gradient vector.
    """
    model.zero_grad()
    output = model(data)
    loss = nn.functional.nll_loss(output, target)
    # ℓ₂ regularization
    if l2_reg > 0:
        l2_penalty = sum(p.pow(2).sum() for p in model.parameters())
        loss = loss + l2_reg * l2_penalty
    loss.backward()
    # Collect flat gradient
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.data.view(-1))
        else:
            grads.append(torch.zeros_like(p.data.view(-1)))
    return torch.cat(grads)


def gradient_clip(grad: torch.Tensor, max_norm: float):
    """Clip gradient by L2 norm."""
    norm = grad.norm()
    if norm > max_norm:
        return grad * (max_norm / norm)
    return grad


def set_model_grad(model: nn.Module, flat_grad: torch.Tensor):
    """Write flat gradient back to model parameters."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.grad = flat_grad[offset:offset + numel].view_as(p).clone()
        offset += numel


def apply_gradient(model: nn.Module, flat_grad: torch.Tensor, lr: float):
    """Apply SGD update: x = x - lr * grad."""
    with torch.no_grad():
        offset = 0
        for p in model.parameters():
            numel = p.numel()
            p.data -= lr * flat_grad[offset:offset + numel].view_as(p)
            offset += numel


# ============================================================
# Byz-WNSGDM (Algorithm 1)
# ============================================================

class ByzWNSGDM:
    """Byzantine-resilient Weighted Normalized SGD with Momentum."""
    
    def __init__(self, model: nn.Module, beta: float = 0.9, l2_reg: float = 1e-4,
                 grad_clip: float = 2.0):
        self.model = model
        self.beta = beta
        self.l2_reg = l2_reg
        self.grad_clip = grad_clip
        self.momentum = {}  # keyed by worker index
    
    def init_momentum(self, worker_loaders: list, device: torch.device):
        """Initialize momentum buffer for each worker."""
        from data_utils import sample_batch
        d = sum(p.numel() for p in self.model.parameters())
        self.momentum = {}
        for i, loader in enumerate(worker_loaders):
            data, target = sample_batch(loader, device)
            g = compute_stochastic_gradient(self.model, data, target, self.l2_reg)
            self.momentum[i] = gradient_clip(g.detach(), self.grad_clip)
    
    def step(self, worker_loaders: list, batch_sizes: list[int],
             gar: str, attack_name: str, f_byz: int,
             n_honest: int, device: torch.device, lr: float) -> torch.Tensor:
        """One training step. Returns the aggregated update direction."""
        from data_utils import sample_batch
        
        honest_grads_local = []  # local stochastic gradients (∇f)
        honest_grads = []        # momentum-mixed gradients (g_ξ)
        
        # Step 1: each honest worker computes local gradient and updates momentum
        for i in range(n_honest):
            data, target = sample_batch(worker_loaders[i], device)
            local_g = compute_stochastic_gradient(self.model, data, target, self.l2_reg)
            local_g = gradient_clip(local_g.detach(), self.grad_clip)
            honest_grads_local.append(local_g)
            
            # Update momentum: g_ξ^{t+1} = β·g_ξ^t + (1-β)·∇f_ξ(x^{t+1})
            self.momentum[i] = self.beta * self.momentum[i] + (1 - self.beta) * local_g
            honest_grads.append(self.momentum[i].clone())
        
        # Step 2: Byzantine workers generate malicious gradients
        byz_grads = generate_byzantine(honest_grads, attack_name, f_byz)
        all_grads = honest_grads + byz_grads
        
        # Step 3: WAggF-weighted aggregation
        weights = compute_waggf_weights(batch_sizes, device=device)
        g_agg = aggregate(all_grads, gar, weights=weights, f=f_byz)
        
        # Step 4: Normalized update: x^{t+1} = x^t - γ·g^t/‖g^t‖
        norm = g_agg.norm()
        update = lr * g_agg / norm if norm > 0 else torch.zeros_like(g_agg)
        apply_gradient(self.model, update, 1.0)  # update already scaled
        
        # Update Byzantine workers' momentum (with the aggregated gradient)
        for i in range(n_honest, len(batch_sizes)):
            if i not in self.momentum:
                self.momentum[i] = g_agg.clone()
            else:
                self.momentum[i] = self.beta * self.momentum[i] + (1 - self.beta) * g_agg
        
        return g_agg


# ============================================================
# Byz-EF21-WSGDM (Algorithm 2)
# ============================================================

class ByzEF21WSGDM:
    """Byzantine-resilient Weighted Error Feedback SGD with Momentum."""
    
    def __init__(self, model: nn.Module, beta: float = 0.9, l2_reg: float = 1e-4,
                 grad_clip: float = 2.0, top_k_frac: float = 0.1):
        self.model = model
        self.beta = beta
        self.l2_reg = l2_reg
        self.grad_clip = grad_clip
        # Top-K keeps the top-K coordinates of the gradient (Definition 2,
        # Q: R^d -> R^d). K is set as a FRACTION of the gradient dimension d
        # so it scales with model size (was a fixed 10, which is 0.013% of d
        # for the MLP and made EF21 barely move).
        d = sum(p.numel() for p in model.parameters())
        top_k = max(1, int(top_k_frac * d))
        self.top_k_frac = top_k_frac
        self.compressor = TopKCompressor(top_k)
        # Per-worker state
        self.v = {}   # local momentum v_ξ
        self.g = {}   # compressed gradient estimate g_ξ
    
    def init_state(self, worker_loaders: list, n_workers: int, device: torch.device):
        """Initialize v_ξ^0 = g_ξ^0 = ∇f_ξ(x^0)."""
        from data_utils import sample_batch
        d = sum(p.numel() for p in self.model.parameters())
        for i in range(n_workers):
            data, target = sample_batch(worker_loaders[i], device)
            g0 = compute_stochastic_gradient(self.model, data, target, self.l2_reg)
            g0 = gradient_clip(g0.detach(), self.grad_clip)
            self.v[i] = g0.clone()
            self.g[i] = g0.clone()
    
    def step(self, worker_loaders: list, batch_sizes: list[int],
             gar: str, attack_name: str, f_byz: int,
             n_honest: int, n_workers: int, device: torch.device,
             lr: float) -> torch.Tensor:
        """One training step."""
        from data_utils import sample_batch
        
        # Step 1: Server broadcasts x^{t+1} (model parameters are shared)
        
        # Step 2: Each honest worker updates local momentum and error feedback
        for i in range(n_honest):
            data, target = sample_batch(worker_loaders[i], device)
            local_g = compute_stochastic_gradient(self.model, data, target, self.l2_reg)
            local_g = gradient_clip(local_g.detach(), self.grad_clip)
            
            # v_ξ^{t+1} = (1-β)·v_ξ^t + β·∇f_ξ(x^{t+1})
            self.v[i] = (1 - self.beta) * self.v[i] + self.beta * local_g
            # q_ξ^{t+1} = Q(v_ξ^{t+1} - g_ξ^t)
            delta = self.v[i] - self.g[i]
            q = self.compressor.compress(delta)
            # g_ξ^{t+1} = g_ξ^t + q_ξ^{t+1}
            self.g[i] = self.g[i] + q
        
        # Step 3: Collect honest compressed estimates
        honest_g = [self.g[i].clone() for i in range(n_honest)]
        
        # Step 4: Byzantine workers generate malicious gradients
        byz_g = generate_byzantine(honest_g, attack_name, f_byz)
        # Assign Byzantine g values
        for j, bg in enumerate(byz_g):
            idx = n_honest + j
            if idx not in self.g:
                self.g[idx] = bg.clone()
            else:
                self.g[idx] = bg.clone()
            if idx not in self.v:
                self.v[idx] = bg.clone()
            else:
                self.v[idx] = bg.clone()
        
        all_g = honest_g + byz_g
        
        # Step 5: WAggF aggregation
        weights = compute_waggf_weights(batch_sizes, device=device)
        g_agg = aggregate(all_g, gar, weights=weights, f=f_byz)
        
        # Step 6: Update model: x^{t+1} = x^t - γ·g^t
        apply_gradient(self.model, lr * g_agg, 1.0)
        
        return g_agg