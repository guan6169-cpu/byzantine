# Byzantine-Resilient Non-Convex Stochastic Optimization with Momentum

Anonymous submission code. This repository implements the two algorithms studied in the paper:

- **Byz-WNSGDM** -- Byzantine-resilient Weighted Normalized SGD with Momentum
- **Byz-EF21-WSGDM** -- Byzantine-resilient Weighted EF21 SGD with Momentum (error feedback + contractive compression)

Both algorithms build on the Weighted Aggregation Framework (WAggF), which assigns larger aggregation weights to clients with larger batch sizes (lower-variance gradients), and plug into four base aggregators (average, median, trimmed mean, Krum).

## Requirements

Python 3.8+ with PyTorch 1.8+ (CUDA), torchvision, numpy. See `requirements.txt`.

## Quick Start

```bash
pip install -r requirements.txt

# Main experiments (Byzantine-fraction sweep + all configurations)
python run_final.py

# Non-IID (Dirichlet) evaluation -- weighted vs uniform aggregation
python run_non_iid.py --dataset mnist --gpu 0 --shard 0 --nshards 4

# Generate figures
python plot_final.py results_final.json --output-dir ./figures
```

## Datasets

MNIST, Fashion-MNIST, and CIFAR-10 are downloaded automatically by torchvision (`download=True` in `data_utils.py`); no manual download is required.

## Reproducing the Paper Results

All experiments use W=25 workers (B=5 Byzantine), batch sizes {1,2,4,8,16} in five groups of four workers, momentum 0.9, and LR in {0.05, 0.2}. The main results use 5 seeds; the ablation study uses 3 seeds.

| Experiment | Script |
|---|---|
| Main results (Figs. 1-4) | `run_final.py` |
| Byzantine-fraction sweep (B/W = 0-40%) | `run_final.py` |
| Ablation study | `run_ablation.py` |
| Non-IID (Dirichlet) evaluation | `run_non_iid.py` (partition code in `noniid_utils.py`) |
| CIFAR-10 evaluation | `cifar10_probe2.py` |

Pre-computed result JSONs are included under `results/`, so the figures and tables can be regenerated without re-running the full training.

## Files

| File | Purpose |
|---|---|
| `algorithms.py` | Byz-WNSGDM + Byz-EF21-WSGDM implementations |
| `aggregation.py` / `aggregators.py` | WAggF weights + 8 aggregation rules + Little/Empire attacks |
| `models.py` | MLP (MNIST/Fashion-MNIST) + CNN (CIFAR-10) |
| `data_utils.py` | Data loading (i.i.d. sampling) |
| `noniid_utils.py` | Dirichlet non-IID partition |
| `train.py` | Training / evaluation loop |
| `run_final.py` | Main experiment runner |
| `run_non_iid.py` | Non-IID evaluation runner |
| `summarize_non_iid.py` | Summarize weighted vs uniform under non-IID |
| `plot_*.py` | Figure generation |
