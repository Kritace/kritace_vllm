#!/usr/bin/env bash
# K：n-gram 接受率测量（chain_i 曾因除零失败，已修复）
export PATH=/root/miniconda3/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST=8.9 TORCH_CUDA_ARCH_LIST=8.9
while pgrep -f "[c]hain_j\.sh" >/dev/null 2>&1; do sleep 15; done
echo "=== chain_j 已结束 $(date -Is) ==="
timeout 900 /root/miniconda3/bin/python /root/spec_acceptance_probe.py 2>&1 | tail -40
echo "===== K DONE $(date -Is) ====="
