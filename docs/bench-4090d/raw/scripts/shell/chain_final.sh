#!/usr/bin/env bash
# FINAL：单一串行队列。每个 GPU 步骤之间调用 wait_gpu()，
# 因为上一个进程退出后 NCCL 进程组与 KV 池不会瞬间释放 —— 紧接着启动会 OOM。
export PATH=/root/miniconda3/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST=8.9 TORCH_CUDA_ARCH_LIST=8.9
PY=/root/miniconda3/bin/python

wait_gpu() {
  for _ in $(seq 1 40); do
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
    if [ "$free_mb" -gt 20000 ]; then return 0; fi
    sleep 5
  done
  echo "  [warn] 显存未归还，强制清理残留进程"
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    kill -9 "$p" 2>/dev/null
  done
  sleep 10
}

cd /root/dzyy-vllm/vllm-v6

echo "########## 1. K=0 基线 ##########"
wait_gpu
timeout 600 $PY /root/verify_spec_correctness.py 0 /tmp/k0.json 2>&1 | tail -8

echo
echo "########## 2. K=5 投机（带形状诊断）##########"
wait_gpu
timeout 600 $PY /root/verify_spec_correctness.py 5 /tmp/k5.json 2>&1 | tail -22

echo
echo "########## 3. 判定 ##########"
wait_gpu
$PY - <<'PY'
import json, os
if not (os.path.exists('/tmp/k0.json') and os.path.exists('/tmp/k5.json')):
    print("  ⚠️ 结果文件缺失")
else:
    a = json.load(open('/tmp/k0.json')); b = json.load(open('/tmp/k5.json'))
    ta, tb = a['token_ids'], b['token_ids']
    first = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
    print(f"  K=0 : steps={a['steps']:4d} tokens={len(ta)}")
    print(f"  K=5 : steps={b['steps']:4d} tokens={len(tb)}")
    print(f"  首个不一致位置: {first}")
    print(f"  token 完全一致: {ta == tb}")
    if ta == tb:
        print(f"  PASS 步数加速比 = {a['steps'] / b['steps']:.2f}x")
    else:
        print(f"  FAIL 前 8 个: K0={ta[:8]} K5={tb[:8]}")
PY

echo
echo "########## 4. n-gram 接受率 ##########"
wait_gpu
timeout 900 $PY /root/spec_acceptance_probe.py 2>&1 | tail -32

echo "===== FINAL DONE $(date -Is) ====="
