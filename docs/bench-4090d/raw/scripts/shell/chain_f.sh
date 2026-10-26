#!/usr/bin/env bash
# F：8B 标度律验证 —— 直接检验 A11 的预测
#   0.6B（d=1024）：TP=2 eager 37.9 vs 单卡 49.5 → 输 23%
#   预测：比值 计算/通信 ∝ hidden 维 → 8B（d=4096）上 TP=2 应追平甚至反超
export PATH=/root/miniconda3/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST=8.9 TORCH_CUDA_ARCH_LIST=8.9
PY=/root/miniconda3/bin/python
M8B=/root/autodl-tmp/huggingface/Qwen3-8B

echo "===== 等前置队列 ====="
while pgrep -f "[c]hain_(all|after)\.sh" >/dev/null 2>&1; do sleep 30; done
echo "队列已空 $(date -Is)"

echo "===== 等 8B 下载完 ====="
while pgrep -f "[d]l_8b\.py" >/dev/null 2>&1; do sleep 30; done
sleep 10
if [ ! -f "$M8B/config.json" ]; then echo "8B 未就绪，跳过 F"; exit 1; fi
echo "8B 大小: $(du -sh $M8B 2>/dev/null | cut -f1)"

cd /root/dzyy-vllm/vllm-v4
run(){ tag="$1"; shift; echo; echo "########## F  $tag ##########"
  timeout 900 $PY bench/bench_v4.py --model "$M8B" --tag "8b-$tag" "$@" 2>&1 | tail -5; }

run single-eager --tp 1 --pp 1 --dp 1 --device-ids 0
run single-cg    --tp 1 --pp 1 --dp 1 --cg --device-ids 0
run tp2-eager    --tp 2 --pp 1 --dp 1 --device-ids 0,1
run tp2-cg       --tp 2 --pp 1 --dp 1 --cg --device-ids 0,1

echo; echo "===== F DONE $(date -Is) ====="
