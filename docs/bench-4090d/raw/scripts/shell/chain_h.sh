#!/usr/bin/env bash
# H：官方 vLLM 的量化 + 投机对照 —— 给 v6/v7 建立"目标值"
#   这是今晚唯一能拿到的、关于 v6/v7 收益的**实测**数据（不需要集成 v6/v7）。
export PATH=/root/miniconda3/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST=8.9 TORCH_CUDA_ARCH_LIST=8.9
export NCCL_SOCKET_IFNAME=lo
VLLM_PY=/root/venvs/vllm-official/bin/python
X=/root/bench_vllm_extra.py
M8B=/root/autodl-tmp/huggingface/Qwen3-8B
MAWQ=/root/autodl-tmp/huggingface/Qwen3-8B-AWQ
DRAFT=/root/huggingface/Qwen3-0.6B

while pgrep -f "[c]hain_(d|g)\.sh" >/dev/null 2>&1; do sleep 30; done
echo "前置完成 $(date -Is)"
for i in $(seq 1 60); do $VLLM_PY -c "import vllm" >/dev/null 2>&1 && break; sleep 20; done
$VLLM_PY -c "import vllm" >/dev/null 2>&1 || { echo "官方 vLLM 不可用"; exit 1; }

run(){ tag="$1"; shift; echo; echo "########## H  $tag ##########"
  timeout 2400 $VLLM_PY $X --tag "$tag" "$@" 2>&1 | tail -8; }

# ① 量化：同规模只改精度（AWQ 的量化方式由模型 config 自动识别，不显式传 --quant）
run 8b-vllm-bf16-cg --model "$M8B"  --tp 1 --cg --cuda-devices 0
run 8b-vllm-awq-cg  --model "$MAWQ" --tp 1 --cg --cuda-devices 0
# ② 投机：真实文本 vs 伪随机 —— 后者用来证明"dummy prompt 会低估收益"
run 8b-vllm-spec-real  --model "$M8B" --draft "$DRAFT" --num-spec 5 --cg --cuda-devices 0 --prompt real
run 8b-vllm-spec-dummy --model "$M8B" --draft "$DRAFT" --num-spec 5 --cg --cuda-devices 0 --prompt dummy

echo; echo "===== H DONE $(date -Is) ====="
