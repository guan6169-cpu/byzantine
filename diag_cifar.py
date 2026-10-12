import torch, sys, time
sys.path.insert(0, '.')
from models import make_model
from data_utils import get_data
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda:0')
tr, te, _, _ = get_data('cifar10', './data')
loader = torch.utils.data.DataLoader(tr, batch_size=128, shuffle=True)

model = make_model('cifar10', device)
opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
crit = nn.CrossEntropyLoss()

model.train()
for ep in range(5):
    total, correct = 0, 0
    t0 = time.time()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        out = model(data)
        loss = crit(out, target)
        loss.backward()
        opt.step()
        total += target.size(0)
        correct += (out.argmax(1) == target).sum().item()
    print(f'epoch {ep+1}: train acc={correct/total:.4f}  ({time.time()-t0:.1f}s)', flush=True)

# test acc
model.eval()
total, correct = 0, 0
with torch.no_grad():
    for data, target in te:
        data, target = data.to(device), target.to(device)
        out = model(data)
        total += target.size(0)
        correct += (out.argmax(1) == target).sum().item()
print(f'TEST acc={correct/total:.4f}', flush=True)
print('DONE', flush=True)
