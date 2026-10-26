#!/usr/bin/env bash
# V：打开逐序列诊断，跑 K=2 短用例（步数少，容易对齐）
export PATH=/root/miniconda3/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST=8.9 TORCH_CUDA_ARCH_LIST=8.9 V6_SPEC_DEBUG=1
PY=/root/miniconda3/bin/python
for _ in $(seq 1 60); do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
  [ "$f" -gt 21500 ] && break
  sleep 5
done
cd /root/dzyy-vllm/vllm-v6
sed -i "s/^N = .*/N = 5/" /root/verify_spec_steps.py

echo "########## K=0 基线（无草稿，不应有 spec 打印）##########"
timeout 600 $PY /root/verify_spec_steps.py 0 /tmp/v0.json 2>&1 | grep -E "k=|\[sched\]|\[spec\]" | head -20

for _ in $(seq 1 40); do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
  [ "$f" -gt 21500 ] && break
  sleep 5
done
echo
echo "########## K=2 投机（看 draft / seg / j）##########"
timeout 600 $PY /root/verify_spec_steps.py 2 /tmp/v2.json 2>&1 | grep -E "k=|\[sched\]|\[spec\]" | head -30

echo
echo "########## 序列对比 ##########"
$PY - <<'PYJ'
import json
a = json.load(open('/tmp/v0.json')); b = json.load(open('/tmp/v2.json'))
pl = a['prompt_len']
ga, gb = a['token_ids'][pl:], b['token_ids'][pl:]
print(f"  K=0 生成 {ga}")
print(f"  K=2 生成 {gb}")
print(f"  一致: {ga == gb}")
PYJ
echo "===== V DONE $(date -Is) ====="
