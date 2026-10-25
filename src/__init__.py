# ============================================================================
# 【本文件是做什么的】包的公开入口：`from src import Engine, EngineConfig, SamplingParams`。
# 三个名字都通过 file-level 的 `__getattr__`（下面的 importlib 逻辑）延迟导入 ——
# 这样 `import src` 不会顺带把 torch / 引擎整条链拉起来（看 help 或做工具脚本时很省时间）。
#
# 【注意】上面的模块 docstring 里写的是 "kritace_vllm public API"，那是从 v4 原样带过来的
# （v5 的公开 API 完全一致，没有变化），不是笔误 —— 但读到时不必怀疑版本。
# ============================================================================

"""kritace_vllm public API: ``from src import Engine, EngineConfig, SamplingParams``."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# ============================================================================
# 【本文件是做什么的】包的公开入口：`from src import Engine, EngineConfig, SamplingParams`。
# 三个名字都通过 file-level 的 `__getattr__`（下面的 importlib 逻辑）延迟导入 ——
# 这样 `import src` 不会顺带把 torch / 引擎整条链拉起来（看 help 或做工具脚本时很省时间）。
#
# 【注意】上面的模块 docstring 里写的是 "kritace_vllm public API"，那是从 v4 原样带过来的
# （v5 的公开 API 完全一致，没有变化），不是笔误 —— 但读到时不必怀疑版本。
# ============================================================================
__all__ = ["Engine", "EngineConfig", "SamplingParams"]

if TYPE_CHECKING:
    from .config import EngineConfig
    from .control.engine import Engine
    from .sampling_params import SamplingParams

# 名字 → (模块路径, 属性名) 的映射表。改模块位置只要改这里。
_LAZY = {
    "Engine": (".control.engine", "Engine"),
    "EngineConfig": (".config", "EngineConfig"),
    "SamplingParams": (".sampling_params", "SamplingParams"),
}


# 只有当前面所有属性查找都失败时，Python 才会调到这里。
def __getattr__(name: str) -> Any:
    """懒加载：避免 ``import src`` 时立即拉起 torch/tokenizer 等重依赖。"""
    try:
        module, attr = _LAZY[name]
    except KeyError:
        # from None 是为了不让 KeyError 出现在 traceback 里 ——
        # 否则报错信息会变成一段关于 dict 的噪声，掩盖真正的问题。
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(importlib.import_module(module, __name__), attr)
