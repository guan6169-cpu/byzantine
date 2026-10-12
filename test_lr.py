import torch, sys, time
sys.path.insert(0, '.')
from models import make_model
from data_utils import get_data, make_worker_loaders
from train import train_one_run
import numpy as np

device = torch.device('cuda:0')
tr, te, _, _ = get_data('cifar10', './data')
bs = [1]*4 + [2]*4 + [4]*4 + [8]*4 + [16]*4 + [4]*5

configs = [
    ('smallCNN lr=0.05 T=300', 0.05, 300),
    ('smallCNN lr=0.2  T=300', 0.2, 300),
    ('smallCNN lr=0.5  T=300', 0.5, 300),
    ('smallCNN lr=1.0  T=300', 1.0, 300),
    ('smallCNN lr=2.0  T=300', 2.0, 300),
    ('smallCNN lr=0.2  T=1000', 0.2, 1000),
    ('smallCNN lr=1.0  T=600', 1.0, 600),
]
for name, lr, T in configs:
    torch.manual_seed(0); np.random.seed(0)
    m = make_model('cifar10', device); m.train()
    ld = make_worker_loaders(tr, bs, 0)
    t0 = time.time()
    accs = train_one_run('byz-wnsgdm', m, ld, bs, te, 'waverage', 'little', 5, lr,
                         T=T, eval_every=10, device=device)
    print(f'{name}: best={max(accs):.4f} final={accs[-1]:.4f}  ({time.time()-t0:.0f}s)', flush=True)
print('DONE', flush=True)
