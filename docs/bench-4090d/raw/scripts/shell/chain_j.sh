#!/usr/bin/env bash
# J：v6 投机解码的真机验收（等 GPU 空出来）
export PATH=/root/miniconda3/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST=8.9 TORCH_CUDA_ARCH_LIST=8.9
PY=/root/miniconda3/bin/python

while pgrep -f "[c]hain_i\.sh" >/dev/null 2>&1; do sleep 15; done
echo "=== chain_i 已结束 $(date -Is) ==="

cd /root/dzyy-vllm/vllm-v6
echo "--- 基线 K=0 ---"
timeout 600 $PY /root/verify_spec_correctness.py 0 /tmp/spec_k0.json 2>&1 | tail -12
echo "--- 投机 K=5 ---"
timeout 600 $PY /root/verify_spec_correctness.py 5 /tmp/spec_k5.json 2>&1 | tail -12

echo
echo "=== 判定 ==="
$PY - <<'PY'
import json
a = json.load(open("/tmp/spec_k0.json"))
b = json.load(open("/tmp/spec_k5.json"))
ta, tb = a["token_ids"], b["token_ids"]
first = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
print(f"  K=0 : steps={a['steps']:4d} tokens={len(ta)}")
print(f"  K=5 : steps={b['steps']:4d} tokens={len(tb)}")
print(f"  首个不一致位置: {first}")
print(f"  token 完全一致: {ta == tb}")
if ta == tb:
    print(f"  ✅ 通过 —— 加速比 = {a['steps'] / b['steps']:.2f}x (步数比)")
else:
    print("  ❌ 失败 —— 投机改变了输出，说明接受判定或 KV 账本有 bug")
print(f"  K=5 最后一步接受数: {b['last_accepted']}")
PY
echo "===== J DONE $(date -Is) ====="
