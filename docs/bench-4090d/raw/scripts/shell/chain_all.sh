#!/usr/bin/env bash
# ⚠️ PATH 必须在这里设：FlashInfer/triton 的 JIT 要 ninja（这条我踩了三次）
export PATH=/root/miniconda3/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST=8.9 TORCH_CUDA_ARCH_LIST=8.9
PY=/root/miniconda3/bin/python
log(){ echo; echo "########## $* ##########"; }

# ── A: DP=2 对照 ──
cd /root/dzyy-vllm/vllm-v4
for m in "" "--cg"; do
  if [ -n "$m" ]; then tag="0.6b-dp2-cg"; else tag="0.6b-dp2-eager"; fi
  log "A  $tag"
  timeout 600 $PY bench/bench_v4.py --model /root/huggingface/Qwen3-0.6B \
    --tp 1 --pp 1 --dp 2 --n-reqs 32 --tag "$tag" $m 2>&1 | tail -6
done

# ── B: MoE 后端对比（cutlass vs triton）──
cd /root/dzyy-vllm/vllm-v5
for be in cutlass triton; do
  log "B  $be / tp2 eager"
  timeout 900 env CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. $PY bench/moe_e2e.py --model moe \
    --devices 0,1 --tp 2 --eager --moe-backend $be --n 64 --warmup 2 --reps 3 \
    --nccl-port $([ "$be" = cutlass ] && echo 30110 || echo 30126) 2>&1 | tail -5
done

# ── C: 并发扫描 ──
for N in 1 8 32 128; do
  log "C  n=$N / tp2 cg"
  timeout 900 env CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. $PY bench/moe_e2e.py --model moe \
    --devices 0,1 --tp 2 --cg --moe-backend triton --n $N --warmup 1 --reps 2 \
    --nccl-port $((30140+N)) 2>&1 | tail -4
done
echo
echo "===== ALL DONE $(date -Is) ====="
