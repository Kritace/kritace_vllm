# ============================================================================
# 【本文件是做什么的】把仓库根目录的 `utils/model_hub.py` 整个导出到 `src.model_hub`。
#
# 【为什么需要它】模型路径解析（HF 缓存目录、本地路径判断、模型下载）是所有版本
# 共用的逻辑，实现放在仓库根的 `utils/` 下；而引擎代码在 `src/` 包里，
# 需要 `from src import model_hub` 这样的入口。这个文件就是那座桥。
#
# 【三步走】① 先把仓库根塞进 sys.path（否则 `import utils` 找不到）；
#           ② 用 `_loader.export_module` 按"同名映射"把目标模块的公开名字搬过来；
#           ③ 不复制实现 —— 所以仓库根那份改了，这里自动跟着改。
#
# ⚠️ `sys.path` 的插入是有判断的（不在才插）：重复插入会让 sys.path 越滚越长，
#   排查导入问题时非常难看清真正的解析顺序。
# ============================================================================
"""Re-export utils.model_hub."""

from __future__ import annotations

import sys
from pathlib import Path

# parents[2]：本文件在 src/ 下 ⇒ 上两级才是仓库根。
# 用 resolve() 取绝对路径，避免相对路径在不同工作目录下解析成不同位置。
_ROOT = Path(__file__).resolve().parents[1]
# 只在缺失时插入（幂等），见文件头对 sys.path 的提醒。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils._loader import export_module  # noqa: E402

# depth=2：从本文件往上两级找 `.` 之外的包边界，从而定位到根目录那份 model_hub。
export_module(globals(), Path(__file__), "model_hub", depth=2)
