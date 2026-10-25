# ============================================================================
# 【本文件是做什么的】一行转发：把仓库根 `utils/sampling_params.py` 的实现
# **导出到本模块**，让引擎侧统一写 `from ..sampling_params import SamplingParams`，
# 不用关心实现放在哪。
#
# 【为什么要先插 sys.path】实现放在仓库根的 `utils/` 下，而本文件在 `src/` 里，
# 属于跨目录复用 —— 所以先把它加进 path 再 import。
#
# 【⚠️ 一个容易踩的点】`parents[2]` 数的是**目录层级**，文件挪个位置就会数错；
# 数错不会报"路径不对"，只会 import 到别的东西（或直接 ImportError）。
# ============================================================================

"""Re-export utils.sampling_params."""

from __future__ import annotations

import sys
from pathlib import Path

# 仓库根目录：本文件在 src/ 下，所以往上数 2 层。
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils._loader import export_module  # noqa: E402

# 导出到本模块的命名空间（depth=2 用来修正模块名）。
export_module(globals(), Path(__file__), "sampling_params", depth=2)
