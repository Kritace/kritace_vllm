#!/usr/bin/env bash
# N：K=5 单跑一次，带诊断
export PATH=/root/miniconda3/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST=8.9 TORCH_CUDA_ARCH_LIST=8.9
cd /root/dzyy-vllm/vllm-v6
timeout 600 /root/miniconda3/bin/python /root/verify_spec_correctness.py 5 /tmp/spec_k5.json 2>&1 | tail -30
echo "===== N DONE $(date -Is) ====="
