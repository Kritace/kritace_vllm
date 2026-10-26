#!/usr/bin/env bash
# P：① 重跑 v6 验收（read_samples 切片已修）② 官方 vLLM 量化+投机对照
export PATH=/root/miniconda3/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST=8.9 TORCH_CUDA_ARCH_LIST=8.9
export NCCL_SOCKET_IFNAME=lo
PY=/root/miniconda3/bin/python
VLLM_PY=/root/venvs/vllm-official/bin/python
M8B=/root/autodl-tmp/huggingface/Qwen3-8B
MAWQ=/root/autodl-tmp/huggingface/Qwen3-8B-AWQ
DRAFT=/root/huggingface/Qwen3-0.6B

wait_gpu() {
  for _ in $(seq 1 40); do
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
    [ "$free_mb" -gt 20000 ] && return 0
    sleep 5
  done
  echo "  [warn] 强制清理"
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  sleep 10
}

echo "########## ① v6 验收：K=0 vs K=5 ##########"
cd /root/dzyy-vllm/vllm-v6
wait_gpu; timeout 600 $PY /root/verify_spec_correctness.py 0 /tmp/k0.json 2>&1 | tail -4
wait_gpu; timeout 600 $PY /root/verify_spec_correctness.py 5 /tmp/k5.json 2>&1 | tail -6
wait_gpu
$PY - <<'PYJ'
import json, os
if os.path.exists('/tmp/k0.json') and os.path.exists('/tmp/k5.json'):
    a = json.load(open('/tmp/k0.json')); b = json.load(open('/tmp/k5.json'))
    ta, tb = a['token_ids'], b['token_ids']
    print(f"  K=0 steps={a['steps']}  K=5 steps={b['steps']}")
    print(f"  token 完全一致: {ta == tb}")
    if ta == tb:
        print(f"  PASS 步数加速比 = {a['steps'] / b['steps']:.2f}x")
    else:
        print(f"  FAIL 首个不一致: {next((i for i,(x,y) in enumerate(zip(ta,tb)) if x!=y), -1)}")
PYJ

echo
echo "########## ② 官方 vLLM：量化 + 投机对照 ##########"
X=/root/bench_vllm_extra.py
run(){ tag="$1"; shift; echo; echo "--- $tag ---"
  timeout 2400 $VLLM_PY $X --tag "$tag" --tp 1 --cg --cuda-devices 0 "$@" 2>&1 | tail -6; }

wait_gpu; run 8b-vllm-bf16-cg --model "$M8B"
wait_gpu; run 8b-vllm-awq-cg  --model "$MAWQ"
wait_gpu; run 8b-vllm-spec-real  --model "$M8B" --draft "$DRAFT" --num-spec 5 --prompt real
wait_gpu; run 8b-vllm-spec-dummy --model "$M8B" --draft "$DRAFT" --num-spec 5 --prompt dummy

echo "===== P DONE $(date -Is) ====="
