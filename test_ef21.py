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
    ('WNSGDM lr=0.05 T=1000', 'byz-wnsgdm', 0.05, 1000),
    ('WNSGDM lr=0.03 T=1000', 'byz-wnsgdm', 0.03, 1000),
    ('EF21 lr=0.05 T=300', 'byz-ef21-wsgdm', 0.05, 300),
    ('EF21 lr=0.01 T=300', 'byz-ef21-wsgdm', 0.01, 300),
    ('EF21 lr=0.005 T=300', 'byz-ef21-wsgdm', 0.005, 300),
]
for name, algo, lr, T in configs:
    torch.manual_seed(0); np.random.seed(0)
    m = make_model('cifar10', device); m.train()
    ld = make_worker_loaders(tr, bs, 0)
    t0 = time.time()
    accs = train_one_run(algo, m, ld, bs, te, 'waverage', 'little', 5, lr,
                         T=T, eval_every=10, device=device)
    print(f'{name}: best={max(accs):.4f} final={accs[-1]:.4f}  ({time.time()-t0:.0f}s)', flush=True)
print('DONE', flush=True)
