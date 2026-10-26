#!/usr/bin/env bash
# 等 setup.sh 结束 → 校验冒烟是否成功 → 自动接上 v4 实测（只跑 0.6B）
while pgrep -f "rental/[s]etup\.sh" >/dev/null 2>&1; do sleep 15; done
echo "===== setup 已结束 $(date -Is) ====="
if grep -q "全部完成" /root/setup_06b.log; then
  echo "冒烟通过，开始 v4 实测"
  cd /root/dzyy-vllm
  export FLASHINFER_CUDA_ARCH_LIST=8.9
  ONLY=0.6b PY=/root/miniconda3/bin/python bash rental/run_v4.sh
  echo "===== v4 实测结束 $(date -Is) ====="
else
  echo "!! setup 未成功完成，跳过 v4 实测。日志末尾："
  tail -20 /root/setup_06b.log
fi
