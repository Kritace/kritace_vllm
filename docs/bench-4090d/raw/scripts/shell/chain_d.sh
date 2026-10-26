#!/usr/bin/env bash
# D：官方 vLLM 对照。必须用它自己的 venv（项目 README 明确警告过）
export PATH=/root/miniconda3/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST=8.9 TORCH_CUDA_ARCH_LIST=8.9
export NCCL_SOCKET_IFNAME=lo
VLLM_PY=/root/venvs/vllm-official/bin/python
M=/root/huggingface/Qwen3-0.6B

echo "===== 等 GPU 队列清空 ====="
while pgrep -f "[c]hain_(all|after|f|e|i|j|k|m)\.sh" >/dev/null 2>&1; do sleep 30; done
echo "队列已空 $(date -Is)"

echo "===== 等官方 vLLM 可导入（最多 20 分钟）====="
for i in $(seq 1 60); do
  if $VLLM_PY -c "import vllm" >/dev/null 2>&1; then echo "vllm 就绪 $(date -Is)"; break; fi
  sleep 20
done
if ! $VLLM_PY -c "import vllm" >/dev/null 2>&1; then echo "官方 vLLM 不可用，跳过 D"; exit 1; fi

cd /root/dzyy-vllm/vllm-v4
run(){ tag="$1"; shift; echo; echo "########## D  $tag ##########"
  timeout 1200 $VLLM_PY bench/bench_vllm.py --model "$M" --tag "vllm-0.6b-$tag" "$@" 2>&1 | tail -6; }

run tp1-eager --tp 1 --cuda-devices 0
run tp1-cg    --tp 1 --cg --cuda-devices 0
run tp2-eager --tp 2 --cuda-devices 0,1
run tp2-cg    --tp 2 --cg --cuda-devices 0,1

echo; echo "===== D DONE $(date -Is) ====="
