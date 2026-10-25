# ============================================================================
# 【本文件是做什么的】moe.kernels 包的门面：三组 kernel 的入口。
#
#     cuda.py    —— 手写 CUDA 辅助核：①按专家重排 token（align block size）
#                   ②融合 SwiGLU ③按路由权重求和；编不出来时用 available() 探测
#     triton.py  —— Triton 写的 grouped GEMM，不依赖预编译产物，任何 SM 都能跑
#
# 【为什么两类 kernel 并存】它们解决的问题不一样：
#     grouped GEMM（专家 GEMM）—— Triton 版本自带 JIT、通用性好，是 cutlass 不可用时的主力；
#     辅助核（重排 / silu / 求和）—— 这三件事本身是逐元素或小规模操作，用 CUDA 手写更直接，
#       走 Triton 反而要绕一圈（还要自己管网格与分块）。
#
# ⚠️ 所以「MoE 后端是 cutlass 还是 triton」只决定专家 GEMM走谁；
#   重排、激活、求和这些辅助步骤在两条路径里都可能落到 CUDA 那套手上。
# ============================================================================
"""MoE kernels: CUDA aux (align/silu/sum) + Triton grouped GEMM."""

from .cuda import (
    align_capacity,
    clear_state as clear_cuda_state,
    cuda_available,
    moe_align_block_size,
    moe_sum,
    silu_and_mul,
)
from . import triton

__all__ = [
    "align_capacity",
    "clear_cuda_state",
    "cuda_available",
    "moe_align_block_size",
    "moe_sum",
    "silu_and_mul",
    "triton",
]
