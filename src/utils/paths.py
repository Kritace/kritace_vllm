# ============================================================================
# 【本文件是做什么的】一层**转发**：把仓库根 `utils/model_hub.py` 的实现搬到
# `src/utils/paths.py` 这个名字下，让引擎侧统一写 `from ..utils import resolve_model_path`。
#
# 【为什么要这么做】实现放在仓库根（那样才能与其它版本的引擎共用同一套模型解析逻辑），
# 而引擎代码在 `src/` 里 —— 直接用相对路径 import 会一路冒到包外。
# 这里用"一行转发"把位置问题隔离掉：以后实现搬家，只改这一个文件。
#
# ⚠️ `HUGGINGFACE_HUB_CACHE` 是**模块级常量**（不是函数）：
#   老代码里有 `from ...paths import HUGGINGFACE_HUB_CACHE` 这种写法，
#   所以它必须是一个可导入的名字，不能改成函数调用。
# ============================================================================
"""Backward-compatible path helpers for v5 internals/tests."""

from __future__ import annotations

from ..model_hub import DEFAULT_CACHE_DIR, is_local_model, resolve_model_path

__all__ = [
    "DEFAULT_CACHE_DIR",
    "HUGGINGFACE_HUB_CACHE",
    "is_local_model",
    "resolve_hf_home",
    "resolve_model_path",
]


# 兼容旧名字：老代码用的是 resolve_hf_home()，新代码用 DEFAULT_CACHE_DIR。
def resolve_hf_home():
    return DEFAULT_CACHE_DIR


# 常量形式的别名 —— 有的调用方直接拿它当默认值参数用，必须保持可导入。
HUGGINGFACE_HUB_CACHE = DEFAULT_CACHE_DIR
