#!/usr/bin/env bash
# I：n-gram 投机接受率的真机测量（快，约 2 分钟）
export PATH=/root/miniconda3/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST=8.9 TORCH_CUDA_ARCH_LIST=8.9
while pgrep -f "[c]hain_e\.sh" >/dev/null 2>&1; do sleep 20; done
echo "chain_e 已结束 $(date -Is)"
timeout 900 /root/miniconda3/bin/python /root/spec_acceptance_probe.py 2>&1 | tail -40
echo "===== I DONE $(date -Is) ====="
