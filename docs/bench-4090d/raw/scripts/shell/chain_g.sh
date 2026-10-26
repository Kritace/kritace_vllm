#!/usr/bin/env bash
# G：腾磁盘 + 下载 AWQ-INT4（纯磁盘/网络，不与 GPU 任务冲突）
#   删 MoE 的前提：chain_e 是最后一个还在用它的任务；v5 的五个用例均已跑完、
#   数据已落盘并拉回本地（rental/out/fetched/）。
export PATH=/root/miniconda3/bin:$PATH
PY=/root/miniconda3/bin/python

while pgrep -f "[c]hain_e\.sh" >/dev/null 2>&1; do sleep 20; done
echo "chain_e 已结束 $(date -Is)"

MOE=/root/autodl-tmp/huggingface/Qwen1.5-MoE-A2.7B
if [ -d "$MOE" ]; then
  echo "释放 MoE（27G）—— v5 数据已归档，重跑需重新下载"
  rm -rf "$MOE" /root/shared/models/Qwen1.5-MoE-A2.7B
fi
df -h /root/autodl-tmp | tail -1

if [ -d /root/autodl-tmp/huggingface/Qwen3-8B-AWQ ]; then
  echo "AWQ 已存在，跳过下载"
else
  echo "下载 Qwen3-8B-AWQ ..."
  $PY -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen3-8B-AWQ', local_dir='/root/autodl-tmp/huggingface/Qwen3-8B-AWQ')
"
fi
ln -sfn /root/autodl-tmp/huggingface/Qwen3-8B-AWQ /root/shared/models/Qwen3-8B-AWQ 2>/dev/null
du -sh /root/autodl-tmp/huggingface/Qwen3-8B-AWQ
echo "AWQ 就绪 $(date -Is)"
