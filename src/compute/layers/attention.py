# ============================================================================
# 【本文件是做什么的】注意力层的三段：输入打包 → 算子调用 → 模块封装。
#     AttnInputs      传给算子的**全部**输入（NamedTuple，防止传参错位）
#     AttnMetadata    本次前向的注意力元信息（块表、长度等），用 context manager 装好
#     attention       `torch.library.custom_op` 注册的自定义算子（含 fake 实现）
#     Attention       模型侧使用的 nn.Module 壳
#
# 【为什么把输入做成 NamedTuple】attention 要传十几个张量（q/k/v、块表、seqlens、
#     位置、缩放、是否 causal ……）。位置传参极易错位且形状相容时不会报错，
#     用具名元组后"名字对不上"会在构造时就暴露。
#
# 【为什么要有 fake 实现】`torch.compile` 在追踪阶段不跑真实 kernel，
#     只调用 register_fake 注册的形状推导函数。没有它就无法编译包含 attention 的图。
#
# 【custom_op 的意义】把这条调用标成"对编译器不透明的边界"：
#     compile 不会试图把内部展开、也不会做跨边界的算子融合 ——
#     换来的是"能正确编译 + 图切分点明确"。
#
# ⚠️ attention 同时服务 prefill 与 decode 两条路径（前者 varlen、后者走 paged KV），
#   两条路径的输入约定不同（见 AttnInputs 各字段的注释）。
# ============================================================================

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
# 本机没有 flash-attn 的可用 wheel 时，退到纯 torch 的等价实现（只影响速度，不影响语义）。
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

    _USING_FALLBACK = False
except ImportError:  # pragma: no cover - 取决于机器
    from ._flash_attn_fallback import (
        flash_attn_varlen_func,
        flash_attn_with_kvcache,
    )

    _USING_FALLBACK = True

from ..kv_ops import store_kvcache

# 传给 attention 算子的**全部**输入。做成 NamedTuple 是为了传参时不容易写错顺序。
class AttnInputs(NamedTuple):
    # 本步 token 的 KV 该写进哪个物理位置。-1 表示跳过（padding 行）。
    slot_mapping: torch.Tensor
    is_prefill: bool
    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_k: Optional[torch.Tensor] = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    block_tables: Optional[torch.Tensor] = None
    cache_seqlens: Optional[torch.Tensor] = None


# 全局不变的信息：整个引擎运行期间只会有一个 KV cache 池，所有层共用它。
# dataclass 只是懒得写 __init__ —— 里面就一个字段。
@dataclass
class AttnMetadata:
    kv_cache: torch.Tensor


_ATTN_META: Optional[AttnMetadata] = None


# 用 contextmanager 而不是手动赋值解开：这样即使 forward 中途抛异常，
# finally 也保证把全局变量恢复回去。
@contextmanager
def attention_context(meta: AttnMetadata):
    global _ATTN_META
    prev = _ATTN_META
    _ATTN_META = meta
    try:
        yield
    finally:
        _ATTN_META = prev


@torch.library.custom_op("vllm::attention", mutates_args=())
# 注册torch库自定义算子，mutates_args，声明本函数没有直接修改输入参数
# compile捕获计算图遇到if会graph break，无法生成完整的triton图
# 用custom_op包装后，compile把整个vllm:attention当作单个黑盒算子节点
# 代价是算子内部的python代码不会被compile优化，真正加速来自内部调用flash-attention CUDA内核
# 自定义算子本体。下面这张参数表虽然长，但每一列都对应 AttnInputs 里的一个字段，
# 顺序必须与 Attention.forward 的调用顺序严格一致 —— 因为全部按位置传。
def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    slot_mapping: torch.Tensor,
    is_prefill: bool,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_tables: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    layer_id: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
# 返回的是**打平**后的输出 [总token数, 头数×头维]，与输入 q 的语义对应。
) -> torch.Tensor:
    meta = _ATTN_META
    assert meta is not None, "attention_context() not active"
    # 池子的形状约定为 [2, 层数, 页数, 块大小, KV头数, 头维]：
    # 0 号是 K、1 号是 V；按 layer_id 切出本层的那一份。
    pool = meta.kv_cache
    # 切出来的是「本层」的视图，不是拷贝 —— 后面的写入会直接落到池子里。
    k_cache = pool[0, layer_id]
    v_cache = pool[1, layer_id]
    # 第一步永远是先把本步新算出的 K/V 写进池子（按 slot_mapping 逐 token 落位）。
    # 必须先写再算：decode 的 attention 要读的正是刚写进去的这一个 token。
    store_kvcache(
        k,
        v,
        k_cache.reshape(-1, num_kv_heads, head_dim),
        v_cache.reshape(-1, num_kv_heads, head_dim),
        slot_mapping,
    )
    # 统一成 flash-attn 要求的 [总token数, 头数, 头维]。contiguous 不能省，
    # 否则 varlen 接口拿到的是错的步长。
    q = q.contiguous().view(-1, num_qo_heads, head_dim)
    # ================= prefill 分支 =================
    # prefill 用 varlen 接口：一批里每条序列长度不同，靠 cu_seqlens 表达边界。
    if is_prefill:
        # 有页表 = 有前缀命中 → K/V 要从池子里取（池子里已经包含了历史 token）。
        if block_tables is not None:
            # 直接把整个池子当 K/V 传给 flash-attn，靠 block_table 寻址。
            k, v = k_cache, v_cache
        else:
            # 不采用 paged kv，直接用新生成kv计算
            # 没有页表 = 这批请求都没命中前缀 → 池子里没有历史内容，
            # 直接用本步新算出的 K/V 就行，省掉一次间接寻址。
            k = k.contiguous().view(-1, num_kv_heads, head_dim)
            v = v.contiguous().view(-1, num_kv_heads, head_dim)
        # 注意这里给的 K/V 可能是「池子视图」也可能是「本步新张量」，
        # 由上面那个 if block_tables 决定 —— 两种情况下 flash-attn 的用法一致。
        o = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            # causal=True 是必须的：prefill 时每个 token 只能看到它自己及之前的 token。
            # 漏掉它会让模型「看到未来」，输出看起来正常但完全错误，且极难发现。
            causal=True,
            block_table=block_tables,
        )
    else:
        # decode阶段
        # decode 走 with_kvcache 接口：它自己处理「读历史 + 拼接当前」，
        # 不需要 cu_seqlens（每序列固定 1 个 query）。
        # q 要 unsqueeze 出一个「序列长度=1」的维度：[bs,1,头数,头维]。
        # 回退实现需要这个上界才能在设备上批量 gather；真 flash-attn 不接受该参数。
        extra = {}
        if _USING_FALLBACK:
            extra["max_seqlen_k"] = max_seqlen_k
        o = flash_attn_with_kvcache(
            q.unsqueeze(1),
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_tables,
            causal=True,
            **extra,
        )
    return o.view(-1, num_qo_heads * head_dim)


@attention.register_fake
# 给自定义算子提供一个纯 Python 的形状模拟实现，只算输出 shape/dtype/device，不跑真实 GPU 计算
# 编译期用的假实现：只推形状，不跑真实 kernel（同 ops.py 里的那些 fake）。
def _attention_fake(
    q, k, v, slot_mapping, is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
    max_seqlen_k, block_tables, cache_seqlens, layer_id, num_qo_heads,
    num_kv_heads, head_dim,
):
    # 返回宽度是 num_qo_heads * head_dim，与上面 view 的结果一致；
    # 两边不一致会让 compile 推出错误的形状，报错位置往往离这里很远。
    return torch.empty(
        (q.shape[0], num_qo_heads * head_dim), dtype=q.dtype, device=q.device
    )


# 模型里真正被调用的那一层。它自己不做任何计算，只是把参数转交给自定义算子。
class Attention(nn.Module):
    # 构造时只存元信息。layer_id 决定去池子的哪一层读写，传错了不会报错、只会算错。
    def __init__(
        self, num_heads: int, head_dim: int, num_kv_heads: int, layer_id: int = 0
    ) -> None:
        # 刻意不建任何参数：Q/K/V 的投影权重在模型那边，本层只负责 attention 本身。
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.layer_id = layer_id

    def forward(
        # 转发到自定义算子。参数顺序必须与注册的函数签名严格一致。
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn: AttnInputs
    ) -> torch.Tensor:
        return torch.ops.vllm.attention(
            q,
            k,
            v,
            attn.slot_mapping,
            attn.is_prefill,
            attn.cu_seqlens_q,
            attn.cu_seqlens_k,
            attn.max_seqlen_q,
            attn.max_seqlen_k,
            attn.block_tables,
            attn.cache_seqlens,
            self.layer_id,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        )
