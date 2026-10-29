"""把 utils.<name> 的实现导出到 src/ 下的转发 shim。

三个 shim（`src/model_hub.py`、`src/sampling_params.py`、`src/compute/models/loader.py`）
都调这里的 `export_module()`：把仓库根 `utils/<name>.py` 的 `__all__` 搬进调用方的 globals，
所以实现只有一份，shim 只是桥。
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def repo_root(shim_file: Path, *, depth: int) -> Path:
    return shim_file.resolve().parents[depth]


def export_module(
    globals_dict: dict,
    shim_file: Path,
    name: str,
    *,
    depth: int,
) -> None:
    root = repo_root(shim_file, depth=depth)
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    mod = importlib.import_module(f"utils.{name}")
    names = list(mod.__all__)
    globals_dict.update({n: getattr(mod, n) for n in names})
    globals_dict["__all__"] = names
