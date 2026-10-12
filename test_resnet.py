import torch, sys, time
sys.path.insert(0, '.')
from torchvision.models import resnet18
import torch.nn as nn
from data_utils import get_data, make_worker_loaders
from train import train_one_run

device = torch.device('cuda:0')
tr, te, _, _ = get_data('cifar10', './data')

def make_resnet18_cifar():
    m = resnet18(num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m.to(device)

# Test 1: ResNet18 + Byz-WNSGDM, paper batch {1,2,4,8,16}, lr=0.2, T=300
bs = [1]*4 + [2]*4 + [4]*4 + [8]*4 + [16]*4 + [4]*5
torch.manual_seed(0); import numpy as np; np.random.seed(0)
m = make_resnet18_cifar(); m.train()
ld = make_worker_loaders(tr, bs, 0)
t0 = time.time()
accs = train_one_run('byz-wnsgdm', m, ld, bs, te, 'waverage', 'little', 5, 0.2, T=300, eval_every=10, device=device)
print(f'[1] ResNet18 + WNSGDM, batch1-16, lr=0.2: best={max(accs):.4f}  ({time.time()-t0:.0f}s)', flush=True)

# Test 2: ResNet18 + Byz-WNSGDM, larger batch {4,8,16,32,64}, lr=2.0, T=300
bs2 = [4]*4 + [8]*4 + [16]*4 + [32]*4 + [64]*4 + [16]*5
torch.manual_seed(0); np.random.seed(0)
m = make_resnet18_cifar(); m.train()
ld = make_worker_loaders(tr, bs2, 0)
t0 = time.time()
accs = train_one_run('byz-wnsgdm', m, ld, bs2, te, 'waverage', 'little', 5, 2.0, T=300, eval_every=10, device=device)
print(f'[2] ResNet18 + WNSGDM, batch4-64, lr=2.0: best={max(accs):.4f}  ({time.time()-t0:.0f}s)', flush=True)

# Test 3: ResNet18 + Byz-WNSGDM, larger batch, lr=0.2, T=500
torch.manual_seed(0); np.random.seed(0)
m = make_resnet18_cifar(); m.train()
ld = make_worker_loaders(tr, bs2, 0)
t0 = time.time()
accs = train_one_run('byz-wnsgdm', m, ld, bs2, te, 'waverage', 'little', 5, 0.2, T=500, eval_every=10, device=device)
print(f'[3] ResNet18 + WNSGDM, batch4-64, lr=0.2, T=500: best={max(accs):.4f}  ({time.time()-t0:.0f}s)', flush=True)

print('DONE', flush=True)
