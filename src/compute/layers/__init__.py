# ============================================================================
# 【本文件是做什么的】layers/ 的对外接口：把这一层的模块名统一从这里导出，
# 上层（模型文件）只 import 这一个入口。
#
# 【为什么第一行要 import ops】`ops` 模块的作用是**注册自定义算子**（副作用），
# 它自己不需要被任何人引用 —— 但必须被 import 一次，注册才会生效。
# 所以这一行是刻意的（带 noqa 标记就是为了说明"看起来没用，其实必须留"）。
#
# 【这一层包含什么】
#     linear.py         —— TP 的四种线性层（列 / 行 / QKV 融合 / 多段融合）
#     vocab_parallel.py —— 词表并行（Embedding / LM Head，含哑元行与分布式 argmax）
#     attention.py      —— 真正的注意力（写 KV + 调 flash-attn）
#     layernorm.py      —— RMSNorm（融合残差相加）
#     rotary.py         —— RoPE 查找表
#     activation.py     —— SwiGLU
#
# 【建议的阅读顺序】先 linear.py（权重怎么切到各卡上），
# 再看 vocab_parallel.py（词表这么大，为什么也要切），最后 attention.py（怎么用切好的权重）。
# ============================================================================

"""Reusable nn modules (Attention, norms, RoPE, activations)."""

from __future__ import annotations

from . import ops  # noqa: F401  注册 vllm::silu_and_mul / rotary_embedding custom op
from .activation import SiluAndMul
from .attention import Attention
from .layernorm import RMSNorm
from .linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    # v5 新增：整份复制、不做任何切分与通信的线性层。
    # 目前两处用到 —— MoE 的 router（每卡都要打出**完全一致**的分数）与
    # EP 下的共享专家（每卡一份完整权重，归约顺序由 MoE 层自己安排）。
    ReplicatedLinear,
    RowParallelLinear,
)
from .rotary import RotaryEmbedding
from .vocab_parallel import ParallelLMHead, VocabParallelEmbedding

# 对外契约：这一层能用的全部模块。
__all__ = [
    "Attention",
    "RMSNorm",
    "RotaryEmbedding",
    "SiluAndMul",
    "ColumnParallelLinear",
    "MergedColumnParallelLinear",
    "QKVParallelLinear",
    "ReplicatedLinear",
    "RowParallelLinear",
    "ParallelLMHead",
    "VocabParallelEmbedding",
]
