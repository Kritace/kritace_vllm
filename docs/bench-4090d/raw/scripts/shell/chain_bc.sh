#!/usr/bin/env bash
cd /root/dzyy-vllm/vllm-v5
export FLASHINFER_CUDA_ARCH_LIST=8.9 TORCH_CUDA_ARCH_LIST=8.9
PY=/root/miniconda3/bin/python
while pgrep -f "[b]ench_v4\.py" >/dev/null 2>&1; do sleep 10; done
echo "===== DP=2 已结束 $(date -Is) ====="
log(){ echo; echo "########## $* ##########"; }

log "B1 cutlass / tp2 eager"
timeout 900 env CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. $PY bench/moe_e2e.py --model moe \
  --devices 0,1 --tp 2 --eager --moe-backend cutlass --n 64 --warmup 2 --reps 3 --nccl-port 30110 2>&1 | tail -5
log "B2 triton / tp2 eager（同参数基准）"
timeout 900 env CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. $PY bench/moe_e2e.py --model moe \
  --devices 0,1 --tp 2 --eager --moe-backend triton --n 64 --warmup 2 --reps 3 --nccl-port 30126 2>&1 | tail -5

for N in 1 8 32 128; do
  log "C n=$N / tp2 cg"
  timeout 900 env CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. $PY bench/moe_e2e.py --model moe \
    --devices 0,1 --tp 2 --cg --moe-backend triton --n $N --warmup 1 --reps 2 --nccl-port $((30140+N)) 2>&1 | tail -4
done
echo "===== B+C 全部结束 $(date -Is) ====="
