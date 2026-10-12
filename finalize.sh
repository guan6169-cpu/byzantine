#!/bin/bash
# Poll for all run_rebuttal shard processes to finish, then finalize.
cd "$HOME/byz-nonconvex-momentum" || exit 1
while true; do
  nproc=$(ps aux | grep run_rebuttal | grep -v grep | wc -l)
  total=$(/usr/bin/python3 -c "import glob,json;print(sum(len(json.load(open(f))) for f in glob.glob('results/rebuttal_*.json')))" 2>/dev/null || echo 0)
  echo "[finalize] $(date +%H:%M:%S) total=$total procs=$nproc"
  if [ "$nproc" -eq 0 ]; then
    break
  fi
  sleep 60
done

echo "[finalize] all shards done. generating figures + summary..."
/usr/bin/python3 plot_rebuttal.py results/rebuttal_*.json --output-dir ./figures_rebuttal > results/plot_log.txt 2>&1
/usr/bin/python3 summarize.py 'results/rebuttal_*_s*.json' > results/summary_log.txt 2>&1
echo "=== theoretical LR ===" > results/theoretical_lr.txt
CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 theoretical_lr.py mnist 300 >> results/theoretical_lr.txt 2>&1
CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 theoretical_lr.py fashionmnist 300 >> results/theoretical_lr.txt 2>&1
CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 theoretical_lr.py cifar10 300 >> results/theoretical_lr.txt 2>&1
echo "DONE $(date +%H:%M:%S)" > results/FINALIZE_DONE
echo "[finalize] complete"
