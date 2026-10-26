#!/usr/bin/env bash
cd /root/dzyy-vllm/vllm-v4
export FLASHINFER_CUDA_ARCH_LIST=8.9
echo "===== probe_server 开始 $(date -Is) ====="
/root/miniconda3/bin/python bench/probe_server.py --devices 0,1 --hidden 1024 --layers 28 --batch 1
echo
echo "===== bench_bandwidth 开始 $(date -Is) ====="
/root/miniconda3/bin/python bench/bench_bandwidth.py --devices 0,1
echo
echo "===== 全部结束 $(date -Is) ====="
