#!/usr/bin/env bash
# bench/3050test/profile.sh —— 采集一份 nsys 报告（CPU 线程 + NVTX 区间 + CUDA kernel），
#                              可选：直接出报表、直接开 Nsight Systems 界面。
#
# 为什么要脚本而不是长命令：
#   ① 环境变量（KRITACE_NVTX / KRITACE_ZOS / PYTHONPATH）必须在 python 启动前设置；
#   ② 参数一多就记不住，且每次都要拼文件名；
#   ③ 产物统一落在 bench/3050test/{nsys,logs}/，文件名能反推出参数。
#
# 用法（在仓库任意位置均可）：
#   ./bench/3050test/profile.sh                     # ZOS=0，1 并发，32 token
#   ./bench/3050test/profile.sh -m 1 -t zos1        # ZOS=1，输出名带 zos1
#   ./bench/3050test/profile.sh -m 1 -n 8 --stats   # 采完顺便打 nvtx_sum / kernel 汇总
#   ./bench/3050test/profile.sh -m 1 --ui           # 采完直接开 Nsight Systems
#
# 产物：
#   bench/3050test/nsys/<tag>.nsys-rep      ← 用 nsys-ui 打开这个
#   bench/3050test/nsys/<tag>.sqlite        ← 给自定义 SQL / timeline.py 用
#   bench/3050test/logs/<tag>_run.log       ← bench 脚本自己的输出（step_ms / sha 等）
set -euo pipefail

cd "$(dirname "$0")/../.."          # 仓库根

PY=${PYTHON:-/home/Kritace/miniconda3/envs/kritace_vllm/bin/python}
NSYS=$(command -v nsys || echo /usr/local/cuda-13.0/bin/nsys)

MODE=0 N=1 K=32 CTX=512 BLOCKS=4 TAG="" DO_STATS=0 DO_UI=0 DO_EXPORT=1

usage() {
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--mode)        MODE="$2";   shift 2 ;;
        -n)               N="$2";      shift 2 ;;
        -k|--max-tokens)  K="$2";      shift 2 ;;
        -c|--context-len) CTX="$2";    shift 2 ;;
        -b|--blocks)      BLOCKS="$2"; shift 2 ;;
        -t|--tag)         TAG="$2";    shift 2 ;;
        --stats)          DO_STATS=1;  shift ;;
        --ui)             DO_UI=1;     shift ;;
        --no-export)      DO_EXPORT=0; shift ;;
        -h|--help)        usage; exit 0 ;;
        *) echo "未知参数: $1" >&2; usage; exit 1 ;;
    esac
done

TAG=${TAG:-zos${MODE}}
OUT="bench/3050test/nsys/${TAG}"
LOG="bench/3050test/logs/${TAG}"
mkdir -p bench/3050test/nsys bench/3050test/logs

# 解释器自检：base 环境没有 transformers，子进程会静默失败。
"$PY" -c "import torch, transformers" 2>/dev/null || {
    echo "❌ $PY 缺 torch/transformers —— 先 conda activate kritace_vllm，或用 PYTHON=... 指定" >&2
    exit 1
}

echo "▶ 采集 KRITACE_ZOS=$MODE  n=$N  k=$K  ctx=$CTX  blocks=$BLOCKS"
echo "  解释器：$PY"

PYTHONPATH=. KRITACE_NVTX=1 KRITACE_ZOS="$MODE" \
    "$NSYS" profile \
        -t cuda,nvtx,osrt \
        --sample=none \
        --force-overwrite true \
        -o "$OUT" \
        "$PY" bench/zos_compare.py \
            --n "$N" --max-tokens "$K" --context-len "$CTX" --blocks "$BLOCKS" \
    2>&1 | tee "${LOG}_run.log"

if [[ $DO_EXPORT -eq 1 ]]; then
    "$NSYS" export --type sqlite --force-overwrite true \
        -o "${OUT}.sqlite" "${OUT}.nsys-rep" >/dev/null 2>&1 || true
    echo "▶ sqlite：${OUT}.sqlite"
fi

if [[ $DO_STATS -eq 1 ]]; then
    for rep in nvtx_sum cuda_gpu_kern_sum; do
        echo
        echo "▶ 报表 ${rep}"
        "$NSYS" stats --report "$rep" --force-export=true "${OUT}.nsys-rep" \
            > "${LOG}_${rep}.txt" 2>&1 || true
        sed -n '1,22p' "${LOG}_${rep}.txt"
    done
fi

if [[ $DO_UI -eq 1 ]]; then
    echo
    echo "▶ 打开 Nsight Systems（WSL 走软件渲染，首次加载可能要十几秒）"
    "$(dirname "$NSYS")/nsys-ui" "${OUT}.nsys-rep" >/dev/null 2>&1 &
fi

cat <<EOF

完成：${OUT}.nsys-rep

EOF
