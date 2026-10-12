import torch, torch.nn as nn, torch.nn.functional as F, sys, time
sys.path.insert(0, '.')
from data_utils import get_data, make_worker_loaders
from train import train_one_run
import numpy as np


class CNNDeeper(nn.Module):
    """3 conv + 2 FC, ~4.6M params, for CIFAR-100."""
    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


device = torch.device('cuda:1')
tr, te, _, _ = get_data('cifar100', './data')

def make_bs(base):
    return [base[0]]*4 + [base[1]]*4 + [base[2]]*4 + [base[3]]*4 + [base[4]]*4 + [base[2]]*5

configs = [
    ('cifar100 DEEP batch{16,32,64,128,256} lr=0.05 T=500', make_bs([16,32,64,128,256]), 0.05, 500),
    ('cifar100 DEEP batch{16,32,64,128,256} lr=0.1  T=500', make_bs([16,32,64,128,256]), 0.1, 500),
]
for name, bs, lr, T in configs:
    torch.manual_seed(0); np.random.seed(0)
    m = CNNDeeper(num_classes=100).to(device); m.train()
    ld = make_worker_loaders(tr, bs, 0)
    t0 = time.time()
    accs = train_one_run('byz-wnsgdm', m, ld, bs, te, 'waverage', 'little', 5, lr,
                         T=T, eval_every=10, device=device)
    print(f'{name}: best={max(accs):.4f} final={accs[-1]:.4f}  ({time.time()-t0:.0f}s)', flush=True)
print('DONE', flush=True)
