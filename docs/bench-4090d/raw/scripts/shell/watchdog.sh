#!/usr/bin/env bash
# 看门狗 v3：认识 v4 和 v5 两套任务，避免把正在跑的 v5 误判成残留
LOG=/root/watchdog.log
STATUS=/root/watchdog_status.txt
IDLE=0; SAME=0; LASTTAG=""
: > "$LOG"
while true; do
  TS=$(date +%H:%M:%S)
  U0=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
  U1=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 1 2>/dev/null | tr -d ' ')
  UMAX=$(( ${U0:-0} > ${U1:-0} ? ${U0:-0} : ${U1:-0} ))
  GPUPIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  CPUTOP=$(ps -eo pcpu --sort=-pcpu --no-headers 2>/dev/null | head -1 | tr -d ' ' | cut -d. -f1); CPUTOP=${CPUTOP:-0}
  NCOMPILE=$(pgrep -cf "[c]c1plus|[g]\+\+|[n]vcc" 2>/dev/null); NCOMPILE=${NCOMPILE:-0}

  P_V4=$(pgrep -cf "[r]un_v4\.sh"); P_V4=${P_V4:-0}
  P_MT=$(pgrep -cf "[r]un_matrix\.py"); P_MT=${P_MT:-0}
  P_BN=$(pgrep -cf "[b]ench_v4\.py"); P_BN=${P_BN:-0}
  P_V5=$(pgrep -cf "[r]un_v5\.sh"); P_V5=${P_V5:-0}
  P_MOE=$(pgrep -cf "[m]oe_e2e\.py"); P_MOE=${P_MOE:-0}
  P_DL=$(pgrep -cf "[d]l_moe\.py"); P_DL=${P_DL:-0}
  P_CH=$(pgrep -cf "[c]hain_v5\.sh"); P_CH=${P_CH:-0}

  # 正在跑的 v4 用例
  L4=$(ls -t /root/dzyy-vllm/rental/out/run_v4_*.log 2>/dev/null | head -1)
  T4=$(grep -E "^>>> " "$L4" 2>/dev/null | tail -1 | grep -oE '\-\-tag [^ ]+' | head -1 | sed 's/--tag //')
  F4=$(grep -c "^FAIL" "$L4" 2>/dev/null); F4=${F4:-0}
  # 正在跑的 v5 用例
  L5=$(ls -t /root/dzyy-vllm/rental/out/run_v5_*.log 2>/dev/null | head -1)
  T5=$(grep -oE "model=[a-z]+ tp=[0-9] ep=[A-Za-z]+ mode=[a-z]+" "$L5" 2>/dev/null | tail -1)
  DLSZ=$(du -sh /root/autodl-tmp/huggingface/Qwen1.5-MoE-A2.7B 2>/dev/null | cut -f1)

  LINE="$TS | GPU ${UMAX}% CPU ${CPUTOP}% | v4=$P_V4 mx=$P_MT bn=$P_BN | v5=$P_V5 moe=$P_MOE dl=$P_DL ch=$P_CH | v4:${T4:-完} v5:${T5:-无} FAIL=$F4 | MoE=$DLSZ"

  ANYTASK=$(( P_MT + P_BN + P_MOE + P_V5 + P_DL ))
  TAG="$T4|$T5"
  if [ -n "$TAG" ] && [ "$TAG" = "$LASTTAG" ]; then SAME=$((SAME+1)); else SAME=0; LASTTAG="$TAG"; fi

  # 可疑：有任务在跑，但 GPU/CPU 都空且无编译器
  if [ "$ANYTASK" -gt 0 ] && [ "$UMAX" -lt 5 ] && [ "$CPUTOP" -lt 50 ] && [ "$NCOMPILE" -eq 0 ]; then
    IDLE=$((IDLE+1))
    [ "$IDLE" -ge 3 ] && LINE="$LINE  ⚠️ ALARM 任务在跑但 GPU+CPU 都空"
  else IDLE=0; fi

  # 同一用例 > 12 分钟且 GPU 空闲（v5 加载权重较久，阈值放宽）
  [ "$SAME" -ge 24 ] && [ "$UMAX" -lt 5 ] && LINE="$LINE  ⚠️ ALARM 同一用例已 $((SAME/2)) 分钟无进展"

  # 孤儿：有进程占 GPU，但**两套任务都不在跑**（v4/v5/下载都算）
  if [ "$ANYTASK" -eq 0 ] && [ "$GPUPIDS" -gt 0 ]; then
    LINE="$LINE  ⚠️ ALARM 有残留进程占 GPU（无任务在跑）"
  fi

  echo "$LINE" >> "$LOG"
  [ "$(wc -l < "$LOG")" -gt 1200 ] && { tail -600 "$LOG" > "$LOG.tmp"; mv "$LOG.tmp" "$LOG"; }
  { echo "更新时间: $(date -Is)"; echo "$LINE"
    echo "--- v4 结果 ---"; grep -E "ttft=|^FAIL" "$L4" 2>/dev/null | tail -14
    echo "--- v5 结果 ---"; grep -E "ttft_ms=|model=" "$L5" 2>/dev/null | tail -14
    echo "--- chain_v5 ---"; tail -8 /root/chain_v5.log 2>/dev/null; } > "$STATUS" 2>/dev/null
  sleep 30
done
