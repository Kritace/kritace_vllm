#!/usr/bin/env bash
# U：验证 read_samples 修复（先短用例，再完整对比）
export PATH=/root/miniconda3/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST=8.9 TORCH_CUDA_ARCH_LIST=8.9
PY=/root/miniconda3/bin/python
for _ in $(seq 1 60); do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
  [ "$f" -gt 21500 ] && break
  sleep 5
done
cd /root/dzyy-vllm/vllm-v6

echo "########## 短用例 K=2（看诊断）##########"
sed -i "s/^N = .*/N = 6/" /root/verify_spec_steps.py
timeout 600 $PY /root/verify_spec_steps.py 2 /tmp/s2.json 2>&1 | grep -E "k=|\[spec\]" | head -20

echo
echo "########## 完整对比 K=0 vs K=5 ##########"
sed -i "s/^N = .*/N = 48/" /root/verify_spec_steps.py
for _ in $(seq 1 40); do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
  [ "$f" -gt 21500 ] && break
  sleep 5
done
timeout 600 $PY /root/verify_spec_steps.py 0 /tmp/s0.json 2>&1 | grep -E "k=|\[spec\]" | tail -3
for _ in $(seq 1 40); do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
  [ "$f" -gt 21500 ] && break
  sleep 5
done
timeout 600 $PY /root/verify_spec_steps.py 5 /tmp/s5.json 2>&1 | grep -E "k=|\[spec\]" | tail -6

echo
echo "########## 判定 ##########"
$PY - <<'PYJ'
import json, os
if not (os.path.exists('/tmp/s0.json') and os.path.exists('/tmp/s5.json')):
    print("  ⚠️ 文件缺失"); raise SystemExit
a = json.load(open('/tmp/s0.json')); b = json.load(open('/tmp/s5.json'))
pl = a['prompt_len']
ga, gb = a['token_ids'][pl:], b['token_ids'][pl:]
print(f"  prompt={pl}  K=0 步数={len(a['trace'])}  K=5 步数={len(b['trace'])}")
print(f"  token 完全一致: {ga == gb}")
if ga == gb:
    print(f"  PASS 步数加速比 = {len(a['trace'])/len(b['trace']):.2f}x")
else:
    print(f"  首个不同(生成段): {next((i for i,(x,y) in enumerate(zip(ga,gb)) if x!=y), -1)}")
print(f"  K=5 接受的草稿总数: {sum(t['accepted'] for t in b['trace'])}")
PYJ
echo "===== U DONE $(date -Is) ====="
