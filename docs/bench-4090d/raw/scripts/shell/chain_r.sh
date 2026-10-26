#!/usr/bin/env bash
# R：官方 vLLM 的投机对照（修掉显存配置后重跑）
export PATH=/root/miniconda3/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST=8.9 TORCH_CUDA_ARCH_LIST=8.9
export NCCL_SOCKET_IFNAME=lo
VLLM_PY=/root/venvs/vllm-official/bin/python
X=/root/bench_vllm_extra.py
M8B=/root/autodl-tmp/huggingface/Qwen3-8B
DRAFT=/root/huggingface/Qwen3-0.6B

while pgrep -f "[c]hain_p\.sh" >/dev/null 2>&1; do sleep 20; done
echo "前置队列已结束 $(date -Is)"
for _ in $(seq 1 40); do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
  [ "$f" -gt 20000 ] && break
  sleep 5
done

run(){ tag="$1"; shift; echo; echo "--- $tag ---"
  timeout 2400 $VLLM_PY $X --tag "$tag" --tp 1 --cg --cuda-devices 0 \
    --max-model-len 2048 --gpu-mem 0.92 "$@" 2>&1 | tail -8; }

# 真实文本 vs 伪随机：后者用来证明"用随机 token 测投机会严重低估收益"
run 8b-vllm-spec-real  --model "$M8B" --draft "$DRAFT" --num-spec 5 --prompt real
run 8b-vllm-spec-dummy --model "$M8B" --draft "$DRAFT" --num-spec 5 --prompt dummy

echo "===== R DONE $(date -Is) ====="
