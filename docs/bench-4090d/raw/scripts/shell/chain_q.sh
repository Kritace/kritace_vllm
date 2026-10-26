#!/usr/bin/env bash
# Q：逐步对比，定位投机偏离点
export PATH=/root/miniconda3/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST=8.9 TORCH_CUDA_ARCH_LIST=8.9
PY=/root/miniconda3/bin/python
wait_gpu() {
  for _ in $(seq 1 40); do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
    [ "$f" -gt 20000 ] && return 0
    sleep 5
  done
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p"; done
  sleep 10
}
# 等 chain_p（官方 vLLM 对照）结束，避免抢卡
while pgrep -f "[c]hain_(p|r)\.sh" >/dev/null 2>&1; do sleep 20; done
echo "chain_p 已结束 $(date -Is)"
cd /root/dzyy-vllm/vllm-v6
wait_gpu; timeout 600 $PY /root/verify_spec_steps.py 0 /tmp/s0.json 2>&1 | tail -3
wait_gpu; timeout 600 $PY /root/verify_spec_steps.py 5 /tmp/s5.json 2>&1 | tail -3
wait_gpu
$PY - <<'PYJ'
import json
a = json.load(open("/tmp/s0.json")); b = json.load(open("/tmp/s5.json"))
pl = a["prompt_len"]
print(f"\nprompt_len={pl}")
print(f"基线步数={len(a['trace'])}  投机步数={len(b['trace'])}")
print(f"生成序列一致: {a['token_ids'][pl:] == b['token_ids'][pl:]}")
print("\n前 8 步对比（基线 1 token/步 vs 投机）:")
print(f"  {'step':>4} {'base':>14} {'draft':>28} {'j':>2} {'spec产出':>22}")
for i in range(max(len(a['trace']), len(b['trace']))):
    ba = a['trace'][i]['new_ids'] if i < len(a['trace']) else []
    bb = b['trace'][i] if i < len(b['trace']) else {'draft': [], 'accepted': 0, 'new_ids': []}
    if i < 8 or ba != bb['new_ids'][:1]:
        print(f"  {i:>4} {str(ba):>14} {str(bb['draft']):>28} {bb['accepted']:>2} {str(bb['new_ids']):>22}")
PYJ
echo "===== Q DONE $(date -Is) ====="
