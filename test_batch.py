import torch, sys, time
sys.path.insert(0, '.')
from models import make_model
from data_utils import get_data, make_worker_loaders
from train import train_one_run
import numpy as np

device = torch.device('cuda:0')
tr, te, _, _ = get_data('cifar10', './data')

def make_bs(base):
    return [base[0]]*4 + [base[1]]*4 + [base[2]]*4 + [base[3]]*4 + [base[4]]*4 + [base[2]]*5

configs = [
    ('batch{16,32,64,128,256} lr=0.05 T=300', make_bs([16,32,64,128,256]), 0.05, 300),
    ('batch{16,32,64,128,256} lr=0.05 T=500', make_bs([16,32,64,128,256]), 0.05, 500),
    ('batch{16,32,64,128,256} lr=0.1  T=500', make_bs([16,32,64,128,256]), 0.1, 500),
    ('batch{8,16,32,64,128}   lr=0.05 T=500', make_bs([8,16,32,64,128]), 0.05, 500),
]
for name, bs, lr, T in configs:
    torch.manual_seed(0); np.random.seed(0)
    m = make_model('cifar10', device); m.train()
    ld = make_worker_loaders(tr, bs, 0)
    t0 = time.time()
    accs = train_one_run('byz-wnsgdm', m, ld, bs, te, 'waverage', 'little', 5, lr,
                         T=T, eval_every=10, device=device)
    print(f'{name}: best={max(accs):.4f} final={accs[-1]:.4f}  ({time.time()-t0:.0f}s)', flush=True)
print('DONE', flush=True)
