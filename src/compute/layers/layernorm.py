# ============================================================================
# 【本文件是做什么的】RMSNorm，以及它的融合残差版本。
#
# 【RMSNorm 与 LayerNorm 的差别】只有缩放系数 γ，没有 bias、也不做均值中心化：
#     RMSNorm(x) = x / sqrt(mean(x²) + eps) * γ
# 少一次求均值与一次减均值，decode 阶段（每步都要过一遍全部归一化层）省下来的访存很实在。
#
# 【为什么 forward 永远返回二元组】(hidden, residual) —— 即使某些层用不到 residual。
#     统一返回类型让调用方少写分支；残差在这里是一路带着走的，
#     而不是像 HF 那样每层做一次 `h = h + shortcut`。
#
# 【融合版本做了什么】`_fused_add_rmsnorm(x, residual, weight, eps)` 一步完成
#     「先把 residual 加进 x，再对结果做归一化」，并且是原地改写这两个张量。
#     省掉一次中间张量分配与一次全量读写 —— 这在每步都要跑的路径上是可见的开销。
#
# ⚠️ 原地改写是个坑：调用处不能写成 `x = _fused_add_rmsnorm(...)` 之类的赋值形式
#   （那会丢掉原地写入的结果，看起来还能跑，只是数值不对）。
#   正确的调用是"不接返回值、直接把 x 与 residual 传进去"（见下面 forward）。
# ============================================================================

from __future__ import annotations

import torch
from torch import nn

from .ops import fused_add_rmsnorm as _fused_add_rmsnorm
from .ops import rmsnorm as _rmsnorm


# 只有一个可学习参数（缩放系数 γ），没有 bias、没有均值中心化 —— 这就是 RMSNorm 与 LayerNorm 的差别。
class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        # eps 防止 x 全零时除零。
        self.eps = eps
        # torch.ones 初始化：RMSNorm 的 γ 默认全 1（保持原尺度）。
        self.weight = nn.Parameter(torch.ones(hidden_size))

    # 返回类型永远是二元组，即使只在普通模式下用 —— 接口统一，调用方少写分支。
    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # 普通模式：没有残差基线（第一次调用时就是这种）。
        # 返回的第二个值是 x 本身 —— 它将成为后续的残差基线。
        if residual is None:
            # 融合模式：一步完成"相加 + 归一化"。
            return _rmsnorm(x, self.weight, self.eps), x
        # 注意这里没有赋值语句：两个张量是被原地改写的，
        # x 现在是归一化结果、residual 现在是 (旧 x + 旧 residual)。
        # 写成 x = ... 反而是错的（那会丢掉原地写的结果）。
        _fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
