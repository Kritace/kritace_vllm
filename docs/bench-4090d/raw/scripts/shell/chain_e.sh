#!/usr/bin/env bash
# E：补跑 cutlass。B 那次大概率被 timeout 900 掐断（编译 6+ 分钟 + benchmark）。
# 编译缓存已建立，这次应该快很多。
export PATH=/root/miniconda3/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST=8.9 TORCH_CUDA_ARCH_LIST=8.9
PY=/root/miniconda3/bin/python

while pgrep -f "[c]hain_(all|after|f)\.sh" >/dev/null 2>&1; do sleep 30; done
echo "队列已空 $(date -Is)"

cd /root/dzyy-vllm/vllm-v5
echo "########## E  cutlass / tp2 eager（补跑）##########"
timeout 1800 env CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. $PY bench/moe_e2e.py --model moe \
  --devices 0,1 --tp 2 --eager --moe-backend cutlass --n 64 --warmup 2 --reps 3 \
  --nccl-port 30210 2>&1 | tail -8

echo; echo "===== E DONE $(date -Is) ====="
