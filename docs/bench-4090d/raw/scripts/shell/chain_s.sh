#!/usr/bin/env bash
# S：一个脚本串行跑完，不再用多脚本互相等待（那一晚反复出问题都是因为依赖成环/残留）
export PATH=/root/miniconda3/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST=8.9 TORCH_CUDA_ARCH_LIST=8.9
export NCCL_SOCKET_IFNAME=lo
PY=/root/miniconda3/bin/python
VLLM_PY=/root/venvs/vllm-official/bin/python
M8B=/root/autodl-tmp/huggingface/Qwen3-8B
DRAFT=/root/huggingface/Qwen3-0.6B

wait_gpu() {
  for _ in $(seq 1 60); do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
    if [ "$f" -gt 21500 ]; then sleep 3; return 0; fi
    sleep 5
  done
  echo "  [warn] 显存未归还，强制清理"
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  sleep 10
}

echo "########## ① 官方 vLLM 投机（真实文本）##########"
wait_gpu
timeout 2400 $VLLM_PY /root/bench_vllm_extra.py --tag 8b-vllm-spec-real \
  --model "$M8B" --draft "$DRAFT" --num-spec 5 --tp 1 --cg --cuda-devices 0 \
  --max-model-len 2048 --gpu-mem 0.92 --prompt real 2>&1 | tail -10

echo
echo "########## ② v6 逐步对比：K=0 ##########"
wait_gpu
cd /root/dzyy-vllm/vllm-v6
timeout 600 $PY /root/verify_spec_steps.py 0 /tmp/s0.json 2>&1 | tail -4

echo
echo "########## ③ v6 逐步对比：K=5 ##########"
wait_gpu
timeout 600 $PY /root/verify_spec_steps.py 5 /tmp/s5.json 2>&1 | tail -20

echo
echo "########## ④ 逐步差异分析 ##########"
wait_gpu
$PY - <<'PYJ'
import json, os
if not (os.path.exists('/tmp/s0.json') and os.path.exists('/tmp/s5.json')):
    print("  ⚠️ 文件缺失"); raise SystemExit
a = json.load(open('/tmp/s0.json')); b = json.load(open('/tmp/s5.json'))
pl = a['prompt_len']
print(f"  prompt_len={pl}  基线步数={len(a['trace'])}  投机步数={len(b['trace'])}")
ga, gb = a['token_ids'][pl:], b['token_ids'][pl:]
print(f"  生成序列一致: {ga == gb}")
if ga != gb:
    print(f"  首个不同位置(生成段): {next((i for i,(x,y) in enumerate(zip(ga,gb)) if x!=y), -1)}")
print(f"\n  前 10 步逐步对比:")
print(f"  {'i':>3} {'基线产出':>12} {'草稿':>26} {'j':>2} {'投机产出':>20} {'一致':>4}")
for i in range(max(len(a['trace']), len(b['trace']))):
    ba = a['trace'][i]['new_ids'] if i < len(a['trace']) else []
    bb = b['trace'][i] if i < len(b['trace']) else {'draft': [], 'accepted': 0, 'new_ids': []}
    ok = "OK" if ba == bb['new_ids'] else "DIFF"
    if i < 10 or ok == "DIFF":
        print(f"  {i:>3} {str(ba):>12} {str(bb['draft'][:6]):>26} {bb['accepted']:>2} {str(bb['new_ids']):>20} {ok:>4}")
PYJ
echo "===== S DONE $(date -Is) ====="
