# ============================================================================
# 【本文件是做什么的】moe 包的门面：把 MoE 的四个部分在这里导出，外部只认这一个入口。
#
#      block.py      SparseMoeBlock（一层 MoE 的前向）、route_experts（路由）
#      experts.py    专家权重容器 + EP 的 expert_map（谁在本卡）
#      parallel.py   TP/EP 布局 + fused 后端选择
#      fused.py      cutlass / triton 两条 kernel 路径的分发
#      kernels/      真正的 kernel 实现（CUDA 辅助核 + Triton grouped GEMM）
#
# 【为什么把 local_expert_range / make_expert_map 也导出】它们不只是内部工具：
#     · 测试要用它们构造「按 ep 切开分别算、再相加，等于单卡全专家一次算完」的对照
#       （tests/gpu/test_moe_kernel.py 的核心断言）；
#     · 排查 EP 问题时，第一个想看的就是「这张卡到底负责哪几个专家」。
#
# ⚠️ fused 路径是真·延迟 import（在 fused.py 里按需 import flashinfer），
#   所以哪怕机器上没装 FlashInfer，这个包也 import 得动 —— 后端探测失败会自动退 Triton。
# ============================================================================
"""Fused MoE: Triton / FlashInfer cutlass + optional CUDA aux kernels.

Layout::

    block.py      SparseMoeBlock + route_experts
    experts.py    weights, EP map, Experts.forward
    parallel.py   TP/EP config + resolve_moe_backend
    fused.py      cutlass / triton dispatch
    kernels/      moe_kernels.cu + cuda.py + triton.py
"""

from __future__ import annotations

from .block import SparseMoeBlock, route_experts
from .experts import local_expert_range, make_expert_map
from .fused import clear_fused_state
from .parallel import (
    MoEParallelConfig,
    cutlass_moe_available,
    resolve_moe_backend,
    triton_moe_available,
)

__all__ = [
    "SparseMoeBlock",
    "MoEParallelConfig",
    "route_experts",
    "clear_fused_state",
    "resolve_moe_backend",
    "cutlass_moe_available",
    "triton_moe_available",
    "local_expert_range",
    "make_expert_map",
]
