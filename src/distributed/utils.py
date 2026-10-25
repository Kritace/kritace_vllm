# ============================================================================
# 【本文件是做什么的】三件小事，都与 PP（流水线并行）有关：
#     ① 把 80 层切成 pp_size 段，回答"本进程负责第几层到第几层"
#     ② 跨 stage 传的中间激活用什么装（IntermediateTensors）
#     ③ 没分到本进程的那几层用什么顶替（PPMissingLayer）
#
# 【要解决什么问题】PP 的核心是把层切开，但切完之后每个进程都会遇到两个新问题：
#     ① 模型代码里写的是 for layer in range(num_hidden_layers)，
#        而本进程只该跑其中一段 —— 所以要有一张"层号 → 归谁"的表。
#     ② 本进程没有的层不能直接删掉，否则层号与实际代码位置对不上，
#        而且权重加载/状态字典会被打乱 —— 所以要用占位层顶住位置。
#
# 【为什么要支持"不按层数等分"】层的显存占用与计算量不完全相同，
#   异构卡（例如 12G + 24G）或"第一段多分几层省通信"这类场景都需要手动指定。
#   所以这里同时提供等分与显式分层表两条路径，默认等分。
#
# 【⚠️ 一个容易踩的点】`_PP_LAYER_RANGES` 是**进程级全局状态**：
#   一旦 set 过，之后 get_pp_indices 都按它走。多引擎/多模型共存时要记得
#   显式传 None 恢复等分，否则会静默沿用上一次的切法。
# ============================================================================

from __future__ import annotations

import torch

# ==============================================================================
# PP 层区间、中间激活、占位层。
#
# get_pp_indices 给 Qwen3 决定本 rank 加载哪些层。
# IntermediateTensors / PPMissingLayer 给跨 stage 的 hidden 和没分到的层。
# ==============================================================================

# 全局的 PP 分层表。None = 按层数等分；否则按显式给定的每段层数走。
_PP_LAYER_RANGES: list[tuple[int, int]] | None = None


# 断言式校验：TP/PP 除不尽是最常见的配置错误，早炸比后面形状对不上好查得多。
def ensure_divisibility(numerator: int, denominator: int) -> None:
    assert numerator % denominator == 0, (
        f"{numerator} is not divisible by {denominator}"
    )


# divide = 校验 + 整除。它让「这里必须能除尽」这件事写在调用点上，而不是散在注释里。
def divide(numerator: int, denominator: int) -> int:
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


# 等分：每段 n 层。除不尽时余数全部压给最后一段（而不是前面每段多一层）。
def _equal_split(
    num_hidden_layers: int, pp_rank: int, pp_size: int
) -> tuple[int, int]:
    # 每段固定 n 层
    n = num_hidden_layers // pp_size
    # 本段起始层号
    start = pp_rank * n
    # 最后一段吃掉剩余全部：num_hidden_layers 除不尽时，余数给最后一段
    end = num_hidden_layers if pp_rank == pp_size - 1 else start + n
    return start, end


# 取本 rank 的 [start, end)：有显式分层表就按表走，否则等分。
def get_pp_indices(
    num_hidden_layers: int, pp_rank: int, pp_size: int
) -> tuple[int, int]:
    """本 rank 的 [start, end)。set_pp_layer_ranges 之后走不均分。"""
    # 走显式分层表（例如 12G + 24G 的异构切法）。
    if _PP_LAYER_RANGES is not None:
        return _PP_LAYER_RANGES[pp_rank]
    # 没有表 → 等分。
    return _equal_split(num_hidden_layers, pp_rank, pp_size)


# 设置显式分层表。传 None 恢复等分 —— 这让「试不同切法」变成一次调用。
def set_pp_layer_ranges(layer_counts: list[int] | None) -> None:
    """按每 stage 层数设区间，例如 12G+24G 的 12/28。None 回到等分。"""
    # 进程级全局状态：一个进程只有一套切法。
    global _PP_LAYER_RANGES
    # 传 None = 恢复等分。
    if layer_counts is None:
        _PP_LAYER_RANGES = None
        return
    # 逐段累加起点，把「每段层数」转成 [start, end) 区间列表。
    ranges: list[tuple[int, int]] = []
    start = 0
    # c = 这一段要吃多少层
    for c in layer_counts:
        ranges.append((start, start + c))
        start += c
    _PP_LAYER_RANGES = ranges


# 跨 PP stage 传的中间激活。用 dict 而不是 tuple —— 只有 hidden 与 residual 两项，按名字取比按下标清楚。
class IntermediateTensors(dict[str, torch.Tensor]):
    """PP 边界上的 hidden_states / residual。"""


# 本 rank 没有的那几层用占位层顶上：forward 原样返回，让层号与实际代码位置保持一一对应。
class PPMissingLayer(torch.nn.Identity):
    """本 rank 没有的层，forward 原样返回第一个参数。"""

    # 原样返回第一个位置参数；只有关键字实参时退而取第一个 kwarg 的值。
    def forward(self, *args, **kwargs):
        return args[0] if args else next(iter(kwargs.values()))
