#!/bin/bash
# Finalize after phase 1 only (MNIST + FashionMNIST). No CIFAR-10.
cd "$HOME/byz-nonconvex-momentum" || exit 1
while [ "$(ps aux | grep run_rebuttal | grep -v grep | wc -l)" -gt 0 ]; do
  sleep 60
done
echo "[finalize] phase 1 done. generating figures + summary (no CIFAR)..."
/usr/bin/python3 plot_rebuttal.py results/rebuttal_*.json --output-dir ./figures_rebuttal > results/plot_log.txt 2>&1
/usr/bin/python3 summarize.py 'results/rebuttal_*_s*.json' > results/summary_log.txt 2>&1
echo "=== theoretical LR ===" > results/theoretical_lr.txt
CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 theoretical_lr.py mnist 300 >> results/theoretical_lr.txt 2>&1
CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 theoretical_lr.py fashionmnist 300 >> results/theoretical_lr.txt 2>&1
echo "DONE $(date +%H:%M:%S)" > results/FINALIZE_DONE
echo "[finalize] complete"
