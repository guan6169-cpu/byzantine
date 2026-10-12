#!/bin/bash
# Resume run (no clean-slate rm): MNIST+Fashion (resume, MARINA-only baselines)
# -> CIFAR-10 (8 shards, no baselines) -> finalize.
cd "$HOME/byz-nonconvex-momentum" || exit 1

echo "[master] Phase 1 (resume): MNIST (GPU 0-3) + FashionMNIST (GPU 4-7)"
for s in 0 1 2 3; do
  nohup /usr/bin/python3 run_rebuttal.py --dataset mnist --gpu $s --shard $s --nshards 4 > logs/mnist_s$s.log 2>&1 &
done
for s in 0 1 2 3; do
  g=$((s + 4))
  nohup /usr/bin/python3 run_rebuttal.py --dataset fashionmnist --gpu $g --shard $s --nshards 4 > logs/fashion_s$s.log 2>&1 &
done

while [ "$(ps aux | grep run_rebuttal | grep -v grep | wc -l)" -gt 0 ]; do
  sleep 60
done
echo "[master] Phase 1 done. Phase 2: CIFAR-10 (GPU 0-7, no baselines)"

for s in 0 1 2 3 4 5 6 7; do
  nohup /usr/bin/python3 run_rebuttal.py --dataset cifar10 --gpu $s --shard $s --nshards 8 --skip-baselines > logs/cifar_s$s.log 2>&1 &
done

while [ "$(ps aux | grep run_rebuttal | grep -v grep | wc -l)" -gt 0 ]; do
  sleep 60
done
echo "[master] Phase 2 done. Finalizing..."

/usr/bin/python3 plot_rebuttal.py results/rebuttal_*.json --output-dir ./figures_rebuttal > results/plot_log.txt 2>&1
/usr/bin/python3 summarize.py 'results/rebuttal_*_s*.json' > results/summary_log.txt 2>&1
echo "=== theoretical LR ===" > results/theoretical_lr.txt
CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 theoretical_lr.py mnist 300 >> results/theoretical_lr.txt 2>&1
CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 theoretical_lr.py fashionmnist 300 >> results/theoretical_lr.txt 2>&1
CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 theoretical_lr.py cifar10 300 >> results/theoretical_lr.txt 2>&1
echo "DONE $(date +%H:%M:%S)" > results/FINALIZE_DONE
echo "[master] ALL DONE"
