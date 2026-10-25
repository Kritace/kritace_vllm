"""torch.compile 自定义算子注册。

该模块将 flashinfer 的底层 kernel 包装为 torch.library 自定义算子，
避免编译器尝试 trace 进这些 kernel 内部。

设计要点：
    - 每个算子通过 torch.library.custom_op 注册，编译时作为黑盒处理。
    - 每个算子注册 fake 实现，供 torch.compile 进行形状和 dtype 推导。
    - 底层 kernel 由 flashinfer 提供，本模块仅做形状适配与接口封装。
"""

from __future__ import annotations
import torch
from flashinfer import apply_rope_with_cos_sin_cache as _flashinfer_rope
from flashinfer.activation import silu_and_mul as _flashinfer_silu_and_mul
from flashinfer.norm import fused_add_rmsnorm as _flashinfer_fused_rmsnorm
from flashinfer.norm import rmsnorm as _flashinfer_rmsnorm


# mutates_args=() 表示这是个纯函数：不改写任何入参，只靠返回值传结果。
@torch.library.custom_op("vllm::silu_and_mul", mutates_args=())
def silu_and_mul(x: torch.Tensor) -> torch.Tensor:

    # 入参最后一维是 2×中间维：gate 支路与 up 支路拼在一起（一次读进来，省一趟访存），
    # 输出是二者逐元素乘之后的结果，维数减半。形状推导见下面的 fake 实现。
    return _flashinfer_silu_and_mul(x)


# 编译期用的假实现：只推形状和 dtype，绝不碰数据、绝不跑 kernel。
@silu_and_mul.register_fake
def _silu_and_mul_fake(x: torch.Tensor) -> torch.Tensor:
    # 最后一维 //2 是关键：门控支路和上投影支路在这里"乘"成一份。
    return torch.empty((x.shape[0], x.shape[1] // 2), dtype=x.dtype, device=x.device)


@torch.library.custom_op("vllm::rotary_embedding", mutates_args=())
def rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    # 每个 token 的绝对位置（前缀命中时不是从 0 开始，而是从命中的长度接着数）
    positions: torch.Tensor,
    # 预先算好的 cos/sin 表：按位置查表，省掉每个 token 重算三角函数
    cos_sin_cache: torch.Tensor,
    # 每个头的维度（Qwen3-0.6B 是 128）
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """执行 RoPE 旋转。"""
    # n = 本步 token 数；n_qh = Q 头数
    n, n_qh = q.shape[:2]
    # n_kh = K 头数，GQA 下比 Q 头数少（Qwen3-0.6B：16 个 Q 头对 8 个 K 头）
    n_kh = k.shape[1]

    # flashinfer 要求输入为 [tokens, num_heads * head_size] 的连续张量。
    # 这里的 reshape 是把 [n, 头, 维] 拉平成 [n, 头×维]：头是主序、维是次序。
    # 不能先 transpose 再 reshape —— 那会把维度顺序交叉错位，而且不会报错，
    # 只会让 RoPE 旋错配对、输出整体变差（静默错误）。
    qf = q.reshape(n, n_qh * head_size).contiguous()
    kf = k.reshape(n, n_kh * head_size).contiguous()

    # 使用 neox 风格的 RoPE 实现。
    # is_neox 决定"哪两个维度配对旋转"：
    #     neox 风格：前半段与后半段配对 —— (x0, x_{d/2}), (x1, x_{d/2+1}), …
    #     交错 风格：相邻两个配对     —— (x0, x1), (x2, x3), …
    #   两种数学上都能表达相对位置，但必须与权重训练时用的那一种一致。
    #   选错同样不报错，但结果全错。
    qo, ko = _flashinfer_rope(positions, qf, kf, head_size, cos_sin_cache, is_neox=True)

    # 恢复原始形状后返回。
    # view_as 是零拷贝的（只改元信息），前提是内存布局仍然匹配 ——
    # 上面 reshape 之后接的 .contiguous() 就是为这一步铺路的。
    return qo.view_as(q), ko.view_as(k)


@rotary_embedding.register_fake
def _rotary_embedding_fake(q, k, positions, cos_sin_cache, head_size):
    # RoPE 是逐元素旋转，形状完全不变 → 原样返回即可，连空张量都不用建。
    return q, k


@torch.library.custom_op("vllm::rmsnorm", mutates_args=())
def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    orig = x.shape
    x2 = x.reshape(-1, weight.shape[0])
    out = _flashinfer_rmsnorm(x2, weight, eps, enable_pdl=False)
    return out.view(orig)


@rmsnorm.register_fake
def _rmsnorm_fake(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # RMSNorm 形状不变 → empty_like 就是正确的形状推导。
    return torch.empty_like(x)


# 唯一有副作用的算子：声明 mutates_args 告诉编译器这两个入参会被原地改写。
@torch.library.custom_op("vllm::fused_add_rmsnorm", mutates_args=("x", "residual"))
def fused_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    # flashinfer fused 是 inplace：x 变为 norm 输出，residual 变为残差和。
    # 声明 mutates_args 让 torch.compile 正确做别名推断（否则 fake 认为输出是新的，
    # 与 CUDA Graph 捕获的实际写入冲突）。
    _flashinfer_fused_rmsnorm(x, residual, weight, eps, enable_pdl=False)


@fused_add_rmsnorm.register_fake
def _fused_add_rmsnorm_fake(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    pass