# ============================================================================
# 【本文件是做什么的】三类"横切"工具的入口：路径解析、可选性能计时、CUDA 环境修正。
#
# 【为什么强调 "no Engine dependency"】这三个模块要在**引擎还没起来**的时候就能用：
#     paths       —— 找模型目录（引擎初始化之前就要知道模型在哪）
#     cuda_env    —— 在 import flashinfer 之前把架构列表钉住（错过时机就晚了）
#     profiler    —— 引擎内部各处埋点，但它们不能反向依赖引擎（否则循环 import）
# 所以它们被单独放一层，谁都可以 import，它们不 import 谁。
#
# ⚠️ profiler 默认是**关闭**的（靠环境变量打开）：埋点本身也有开销，
#   常开会让性能数字失真 —— 计时工具污染被测对象是这类工具的经典陷阱。
# ============================================================================
"""Cross-cutting helpers with no Engine dependency.

  paths.py      model / HF cache resolution
  profiler.py   optional full-stack timers (``KRITACE_SYS_PROF``)
  cuda_env.py   FlashInfer arch pinning for mixed-GPU hosts
"""

from __future__ import annotations

from .cuda_env import apply_flashinfer_arch_list
from .paths import resolve_hf_home, resolve_model_path
from .profiler import CudaTimer, Timer, enabled, reset, rollup, summary

__all__ = [
    "CudaTimer",
    "Timer",
    "apply_flashinfer_arch_list",
    "enabled",
    "reset",
    "resolve_hf_home",
    "resolve_model_path",
    "rollup",
    "summary",
]
