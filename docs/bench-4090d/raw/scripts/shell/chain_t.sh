#!/usr/bin/env bash
# T：只跑 K=2，步数少、打印够看
export PATH=/root/miniconda3/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST=8.9 TORCH_CUDA_ARCH_LIST=8.9
PY=/root/miniconda3/bin/python
for _ in $(seq 1 60); do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
  [ "$f" -gt 21500 ] && break
  sleep 5
done
cd /root/dzyy-vllm/vllm-v6
sed -i "s/^N = 16.*/N = 6/" /root/verify_spec_steps.py
timeout 600 $PY /root/verify_spec_steps.py 2 /tmp/s2.json 2>&1 | grep -E "k=|spec\]" | head -30
echo "===== T DONE $(date -Is) ====="
