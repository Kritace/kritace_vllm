#!/usr/bin/env bash
# M：v6 真机验收（K=0 vs K=5）+ n-gram 接受率
# GPU 已清空，立即跑。两次运行之间留出显存归还的时间 —— 上一个进程退出后
# NCCL 进程组与 KV 池不会瞬间释放，紧接着启动会 OOM（已踩过一次）。
export PATH=/root/miniconda3/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST=8.9 TORCH_CUDA_ARCH_LIST=8.9
PY=/root/miniconda3/bin/python

echo "########## 1. 投机正确性验收 ##########"
cd /root/dzyy-vllm/vllm-v6
echo "--- 基线 K=0 ---"
timeout 600 $PY /root/verify_spec_correctness.py 0 /tmp/spec_k0.json 2>&1 | tail -15
sleep 15
echo "--- 投机 K=5 ---"
timeout 600 $PY /root/verify_spec_correctness.py 5 /tmp/spec_k5.json 2>&1 | tail -15

echo
echo "=== 判定 ==="
$PY - <<'PY'
import json, os
if not (os.path.exists('/tmp/spec_k0.json') and os.path.exists('/tmp/spec_k5.json')):
    print("  ⚠️ 结果文件缺失，说明某次运行失败")
else:
    a = json.load(open('/tmp/spec_k0.json')); b = json.load(open('/tmp/spec_k5.json'))
    ta, tb = a['token_ids'], b['token_ids']
    first = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
    print(f"  K=0 : steps={a['steps']:4d} tokens={len(ta)}")
    print(f"  K=5 : steps={b['steps']:4d} tokens={len(tb)}")
    print(f"  首个不一致位置: {first}")
    print(f"  token 完全一致: {ta == tb}")
    if ta == tb:
        print(f"  PASS 步数加速比 = {a['steps'] / b['steps']:.2f}x")
    else:
        print("  FAIL 投机改变了输出 => 接受判定或 KV 账本有 bug")
PY

echo
echo "########## 2. n-gram 接受率 ##########"
sleep 15
timeout 900 $PY /root/spec_acceptance_probe.py 2>&1 | tail -35

echo "===== M DONE $(date -Is) ====="
