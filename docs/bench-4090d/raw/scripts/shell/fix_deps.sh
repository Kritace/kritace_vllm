#!/usr/bin/env bash
PY=/root/miniconda3/bin/python
PKGS=(flashinfer-python msgspec pyzmq triton transformers safetensors
      "huggingface_hub>=0.23.0" "Brotli>=1.1.0" "zstandard>=0.24.0"
      xxhash pyyaml numpy tqdm protobuf tabulate)

# 1) 暂停 MoE 下载，让出带宽（SIGSTOP 保留进度）
MOE_PID=$(pgrep -f "[d]l_moe.py" | head -1)
if [ -n "$MOE_PID" ]; then kill -STOP "$MOE_PID" && echo "[1] 已暂停 MoE (pid $MOE_PID)，让出带宽"; fi

# 2) 清 pip 缓存，避免复用刚才那个损坏的 wheel
$PY -m pip cache purge 2>/dev/null | tail -1 || true

# 3) 多镜像重试安装
ok=0
for M in https://pypi.tuna.tsinghua.edu.cn/simple \
         https://mirrors.aliyun.com/pypi/simple \
         https://pypi.org/simple ; do
  echo "[2] 尝试镜像 $M"
  if $PY -m pip install -q --timeout 180 --retries 10 -i "$M" "${PKGS[@]}"; then
    echo "[2] 安装成功（$M）"; ok=1; break
  fi
  echo "[2] $M 失败，换下一个"
done

# 4) 恢复 MoE
if [ -n "$MOE_PID" ]; then kill -CONT "$MOE_PID" && echo "[4] 已恢复 MoE"; fi
[ "$ok" = 1 ] || { echo "!! 依赖安装全部失败"; exit 1; }

# 5) 验证算子栈
echo "[5] 验证 import"
$PY - <<'PYEOF'
import flash_attn, flashinfer, transformers, msgspec, zmq, triton
from flashinfer import apply_rope_with_cos_sin_cache
from flashinfer.activation import silu_and_mul
from flashinfer.norm import fused_add_rmsnorm, rmsnorm
print("  flash_attn", flash_attn.__version__)
print("  flashinfer", flashinfer.__version__)
print("  transformers", transformers.__version__)
print("  OK: v3/v4/v5 需要的算子 API 全部可用")
PYEOF
rc=$?
[ $rc -eq 0 ] || { echo "!! import 验证失败"; exit 1; }

# 6) 接上 v4 实测
echo "[6] 开始 v4 实测（只 0.6B）"
cd /root/dzyy-vllm
export FLASHINFER_CUDA_ARCH_LIST=8.9
ONLY=0.6b PY=$PY bash rental/run_v4.sh
echo "[6] v4 实测结束 $(date -Is)"
