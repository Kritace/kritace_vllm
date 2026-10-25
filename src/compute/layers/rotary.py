# ============================================================================
# 【本文件是做什么的】RoPE（旋转位置编码）：预计算一张 cos/sin 表，前向时查表旋转 q/k。
#
# 【为什么预计算表】每个位置的旋转角只与"位置 × 频率"有关，与输入无关 ⇒
#     启动时算一次 `[max_position, head_dim]` 的表，前向只做查表与逐元素乘加。
#     表是按 head_dim 建的，所有层共用同一张（层与层之间的 rope 参数相同）。
#
# 【为什么强制 fp32】FlashInfer 的 rope 算子要求 cos/sin 表是 float32：
#     位置编码的相位对精度敏感，bf16 表会在长上下文下明显劣化。
#     所以 _apply 里挂了一个钩子 —— 模型 `.to(dtype)` / `.cuda()` 之后
#     强制把这张表转回 fp32（否则会被一起转成 bf16，且不报错）。
#
# 【NeoX half-split 约定】频率只取偶数下标（head_dim/2 个），旋转时把向量前后两半配对，
#     这是 NeoX 风格；与 GPT-J 的"相邻两两配对"不同。混用会导致输出轻微错误（不易察觉）。
#
# ⚠️ 目前只支持全维旋转（rotary_dim == head_dim）。部分旋转的模型需要在这里扩展，
#   否则会静默地旋转了不该旋转的维度。
# ============================================================================

from __future__ import annotations

import torch
from torch import nn


def build_cos_sin_cache(
    head_dim: int,
    max_position_embeddings: int,
    rope_theta: float,
) -> torch.Tensor:
    """构建 [max_pos, head_dim] 查找表：前 head_dim/2 列为 cos、后为 sin。"""
    # 只取偶数下标 → head_dim/2 个逆频率（NeoX half-split 约定）
    # inv_freq 是 RoPE 的核心：第 i 对维度对应的角频率是 base^(-2i/d)。
    # arange 用步长 2（只取偶数维），因为 RoPE 是成对旋转的。
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
    )
    # 位置 × 频率 外积 → [max_pos, head_dim/2] 的相位角
    positions = torch.arange(max_position_embeddings, dtype=torch.float)
    angles = positions[:, None] * inv_freq[None, :]
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


# 构造期做两件事：算频段、算表。表是 register_buffer，会跟着 .to(device) 一起搬。
class RotaryEmbedding(nn.Module):
    """持有 cos/sin 查找表，forward 转交 vllm::rotary_embedding 算子。"""

    # head_size 与 rotary_dim 分开传但这里要求相等 —— 保留参数是为了将来支持
    # "只在部分维度上做旋转"（如 GLM 的 partial RoPE）。
    def __init__(
        self,
        head_dim: int,
        rotary_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        # 仅支持全维旋转；部分旋转（rotary_dim < head_dim）需扩展此处
        assert rotary_dim == head_dim, "rotary_dim must equal head_dim (full rotation)"
        cache = build_cos_sin_cache(head_dim, max_position_embeddings, rope_theta)
        # persistent=False 表示该缓存实时计算，在保存权重时不会被保存
        # persistent=False 表示"不进 state_dict"—— 它是算出来的，不需要存进权重文件。
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    # 拦一层设备/精度转换：FlashInfer 要求 cos/sin 表始终是 fp32。
    def _apply(self, fn, recurse: bool = True):
        """PyTorch在每次设备或者类型转换时自动调用的内部函数
        保证cos_sin_cache始终fp32，这是flashInfer的要求
        """

        # 父类照常执行转换操作
        # 父类照常做搬运（搬设备、转 dtype 都靠它）。
        out = super()._apply(fn, recurse) 
        # 如果缓存类型不是float32，则转换为float32
        # 搬完立刻把 dtype 拉回 fp32 —— 否则 .to(bf16) 会把它一起降精度。
        if self.cos_sin_cache.dtype != torch.float32:
            self.cos_sin_cache = self.cos_sin_cache.to(torch.float32)
        return out

    # 前向只做转发。注意它接收 positions 而不是 int ——
    # 因为前缀命中时位置不从 0 开始（要从命中的长度接着数）。
    # 前向只做转发。接收的是 positions 张量而不是单个整数：
    def forward(
        self,
        # 因为前缀命中时位置不从 0 开始，要从命中的长度接着数。
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 调用vllm的rotary_embedding算子
        # cos_sin_cache 已在 executor 加载模型后固定为 fp32（FlashInfer 要求）。
        # 不能在 forward 里惰性 .float()：torch.compile 的 CUDAGraph 会复用该临时内存，
        # 下次重放时覆盖 → "overwritten by a subsequent run"。
        # 走自定义算子（见 ops.py）：这样 torch.compile 把它当黑盒，
        # 不会尝试 trace 进 FlashInfer 的 kernel 内部而断图。
        return torch.ops.vllm.rotary_embedding(
            query, key, positions, self.cos_sin_cache, self.head_dim
        )
