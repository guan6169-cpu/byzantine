"""Byzantine-resilient non-convex stochastic optimization with momentum.
Implements Byz-WNSGDM and Byz-EF21-WSGDM under WAggF.
Supports MNIST, Fashion-MNIST, and CIFAR-10.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Models
# ============================================================

class MLP(nn.Module):
    """Two-layer MLP: 784 -> 100 -> ReLU -> 10 -> log-softmax."""
    def __init__(self, input_dim=784, hidden_dim=100, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


class CNN(nn.Module):
    """Small CNN for CIFAR-10: 2 conv + 2 FC."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


def make_model(dataset: str, device: torch.device) -> nn.Module:
    if dataset in ("mnist", "fashionmnist"):
        return MLP().to(device)
    elif dataset == "cifar10":
        return CNN().to(device)
    elif dataset == "cifar100":
        return CNN(num_classes=100).to(device)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")