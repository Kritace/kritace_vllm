#!/usr/bin/env bash
while pgrep -f "[r]un_v4\.sh" >/dev/null 2>&1; do sleep 20; done
echo "===== v4 已结束 $(date -Is) ====="
while pgrep -f "[d]l_moe\.py" >/dev/null 2>&1; do sleep 20; done
echo "===== MoE 下载已结束 $(date -Is) ====="
D=/root/autodl-tmp/huggingface/Qwen1.5-MoE-A2.7B
sz=$(du -sb "$D" 2>/dev/null | cut -f1); sz=${sz:-0}
echo "MoE 体积 $((sz/1000000000)) GB（期望 ~28.6）"
ls "$D"/*.safetensors >/dev/null 2>&1 || { echo "!! 缺 safetensors，跳过 v5"; exit 1; }
cd /root/dzyy-vllm
export FLASHINFER_CUDA_ARCH_LIST=8.9
PY=/root/miniconda3/bin/python bash rental/run_v5.sh
echo "===== v5 实测结束 $(date -Is) ====="
