#!/usr/bin/env bash
# X：验证投机的正确性（组包修复 + 扁平布局修复）
export PATH=/root/miniconda3/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST=8.9 TORCH_CUDA_ARCH_LIST=8.9 V6_SPEC_DEBUG=1
PY=/root/miniconda3/bin/python
wait_gpu() { for _ in $(seq 1 60); do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
  [ "$f" -gt 21500 ] && return; sleep 5; done; }
cd /root/dzyy-vllm/vllm-v6

wait_gpu
sed -i "s/^N = .*/N = 4/" /root/verify_spec_steps.py
echo "########## K=2 短用例 ##########"
timeout 600 $PY /root/verify_spec_steps.py 2 /tmp/x2.json 2>&1 | grep -E "k=|\[spec\]|Error|error|assert|Traceback" | head -12

wait_gpu
sed -i "s/^N = .*/N = 48/" /root/verify_spec_steps.py
echo
echo "########## K=0 基线 ##########"
timeout 900 $PY /root/verify_spec_steps.py 0 /tmp/x0.json 2>&1 | grep -E "^  k=|Error|Traceback" | tail -3

wait_gpu
echo
echo "########## K=5 投机 ##########"
timeout 900 $PY /root/verify_spec_steps.py 5 /tmp/x5.json 2>&1 | grep -E "^  k=|Error|Traceback" | tail -3

echo
echo "########## 判定 ##########"
$PY - <<'PYJ'
import json, os
if not (os.path.exists('/tmp/x0.json') and os.path.exists('/tmp/x5.json')):
    print("  ⚠️ 文件缺失"); raise SystemExit
a = json.load(open('/tmp/x0.json')); b = json.load(open('/tmp/x5.json'))
pl = a['prompt_len']
ga, gb = a['token_ids'][pl:], b['token_ids'][pl:]
print(f"  prompt={pl}  K=0 步数={len(a['trace'])}  K=5 步数={len(b['trace'])}")
print(f"  token 完全一致: {ga == gb}")
if ga == gb:
    print(f"  ✅ PASS  步数加速比 = {len(a['trace'])/len(b['trace']):.2f}x")
else:
    i = next((i for i,(x,y) in enumerate(zip(ga,gb)) if x!=y), -1)
    print(f"  ❌ 首个不同(生成段): {i}")
acc = sum(t['accepted'] for t in b['trace'])
print(f"  K=5 累计接受草稿数: {acc}")
PYJ
echo "===== X DONE $(date -Is) ====="
