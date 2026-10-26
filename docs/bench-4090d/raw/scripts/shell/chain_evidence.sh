#!/usr/bin/env bash
# EVIDENCE：重跑 §九 9.4 的判定实验并把完整输出落盘，供归档溯源。
export PATH=/root/miniconda3/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST=8.9 TORCH_CUDA_ARCH_LIST=8.9
PY=/root/miniconda3/bin/python
wait_gpu() { for _ in $(seq 1 60); do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
  [ "$f" -gt 21500 ] && return; sleep 5; done; }
cd /root

for spec in "01-numerical-tie:compare_step.py:" \
            "02-canonical-ref:decide_v6.py:" \
            "03-clean-history:isolate_v6.py:V6_SPEC_DEBUG=1" \
            "04-poison-kvpool:poison_v6.py:V6_SPEC_DEBUG=1" \
            "05-no-corrupt:trigger_v6.py:V6_SPEC_DEBUG=1"; do
  name=$(echo "$spec" | cut -d: -f1)
  script=$(echo "$spec" | cut -d: -f2)
  envs=$(echo "$spec" | cut -d: -f3)
  wait_gpu
  echo "########## $name ($script) ##########"
  env $envs timeout 600 $PY /root/$script 2>&1 \
    | grep -vE "it/s\]|Loading weights|KV pool|NCCL|Warning|warn|torch_dtype" \
    > /root/evidence_${name}.log
  tail -3 /root/evidence_${name}.log
  echo "  -> /root/evidence_${name}.log ($(wc -l < /root/evidence_${name}.log) 行)"
  echo
done
echo "===== EVIDENCE DONE $(date -Is) ====="
