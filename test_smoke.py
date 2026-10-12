"""Quick smoke test: verify algorithms run without errors."""
import torch
import sys
sys.path.insert(0, ".")

from models import make_model
from data_utils import get_data, make_worker_loaders
from train import train_one_run


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    for ds in ["mnist", "fashionmnist", "cifar10"]:
        print(f"\n--- {ds} ---")
        trainset, test_loader, _, _ = get_data(ds, data_dir="./data")
        
        # Simple config: W=25, B=5, waverage
        batch_sizes = [1]*4 + [2]*4 + [4]*4 + [8]*4 + [16]*4 + [4]*5
        
        for algo in ["byz-wnsgdm", "byz-ef21-wsgdm"]:
            for att in ["little", "empire"]:
                print(f"  {algo} + {att}...", end=" ", flush=True)
                model = make_model(ds, device)
                model.train()
                loaders = make_worker_loaders(trainset, batch_sizes, seed=0)
                
                accs = train_one_run(
                    algorithm=algo, model=model, worker_loaders=loaders,
                    batch_sizes=batch_sizes, test_loader=test_loader,
                    gar="waverage", attack=att, f_byz=5, lr=0.05,
                    T=20, eval_every=5, device=device, verbose=False,
                )
                print(f"done, final_acc={accs[-1]:.4f}")
    
    print("\n✅ All smoke tests passed.")


if __name__ == "__main__":
    test()