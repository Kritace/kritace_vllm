#!/usr/bin/env bash
export PATH=/root/miniconda3/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST=8.9 TORCH_CUDA_ARCH_LIST=8.9
PY=/root/miniconda3/bin/python

echo "===== 等 chain_all 结束 ====="
while pgrep -f "[c]hain_all.sh" > /dev/null 2>&1; do sleep 20; done
echo "chain_all 已结束 $(date -Is)"

cd /root/dzyy-vllm/vllm-v4
echo
echo "===== DP 同口径单卡对照 (reqs=32, eager/cg) ====="
timeout 900 $PY /root/dp1_ref.py 2>&1 | tail -20

echo
echo "===== AFTER DONE $(date -Is) ====="
