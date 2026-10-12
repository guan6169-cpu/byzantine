"""Data loading: MNIST, Fashion-MNIST, CIFAR-10."""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np


def get_data(dataset_name: str, data_dir: str = "./data"):
    """Load dataset, return (full_trainset, test_loader, input_dim, num_classes)."""
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        trainset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        testset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        input_dim, num_classes = 784, 10

    elif dataset_name == "fashionmnist":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        trainset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        testset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
        input_dim, num_classes = 784, 10

    elif dataset_name == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
        input_dim, num_classes = 3072, 10

    elif dataset_name == "cifar100":
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        trainset = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform_test)
        input_dim, num_classes = 3072, 100

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    test_loader = DataLoader(testset, batch_size=1000, shuffle=False)
    return trainset, test_loader, input_dim, num_classes


def make_worker_loaders(trainset, batch_sizes, seed):
    """Create one DataLoader per worker (i.i.d. sampling with replacement).

    Each worker samples independently from the full training set.
    batch_sizes[i] = batch size for worker i.

    Uses an infinite sampler so the iterator never exhausts (matters when a
    worker's batch size is large enough that num_batches < T).
    """
    loaders = []
    for bs in batch_sizes:
        s = seed + abs(bs) % 10000
        sampler = _InfiniteSampler(len(trainset), s)
        loader = DataLoader(trainset, batch_size=bs, sampler=sampler)
        loaders.append(iter(loader))
    return loaders


class _InfiniteSampler(torch.utils.data.Sampler):
    """Sampler that yields shuffled permutations of [0, n) forever."""

    def __init__(self, n, seed):
        self.n = n
        self.seed = seed

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            yield from torch.randperm(self.n, generator=g).tolist()

    def __len__(self):
        return 2 ** 31 - 1


def sample_batch(worker_iter, device):
    """Sample one batch from a worker's infinite loader."""
    try:
        data, target = next(worker_iter)
    except StopIteration:
        return None, None
    return data.to(device), target.to(device)