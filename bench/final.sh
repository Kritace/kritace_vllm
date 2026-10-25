#!/usr/bin/env bash
# 最终回归：三组场景 × (KRITACE_ZOS=0/1)，逐 token 对拍。
set -u
out=${1:-/tmp/final.txt}
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=.
PY=${PY:-python3}
: > "$out"

run2() {
  local row
  row=$(KRITACE_ZOS=0 timeout 900 "$PY" bench/zos_matrix.py "$@" 2>&1 | grep -E '^CASE|^TOKENS')
  echo "$row" >> "$out"
  row=$(KRITACE_ZOS=1 timeout 900 "$PY" bench/zos_matrix.py "$@" 2>&1 | grep -E '^CASE|^TOKENS')
  echo "$row" >> "$out"
}

run2 --case short --n 4 --max-tokens 16 --blocks 32
run2 --case chunk --n 1 --max-tokens 8 --max-batched-tokens 64 --long-repeat 4 --blocks 48
run2 --case short --n 8 --max-tokens 8 --blocks 24 --prefix-backend hash
echo done >> "$out"
