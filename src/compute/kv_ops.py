# ============================================================================
# 【本文件是做什么的】一个 Triton 内核 + 一层薄封装：把这一步算出的 K / V
# 按 `slot_mapping` 写进分页池。
#
# 【为什么不直接用通用的索引写算子】通用的 scatter 算子要照顾任意形状、任意索引模式，
# 于是它自带一串与数据量无关的固定开销（参数校验、索引边界检查、通用线程调度）。
# 在 decode 场景下这一步只写 1 个 token、真正的搬运只有几 KB，
# 那些固定开销会完全盖过搬运本身。自研版省在两处：
#     ① **K 与 V 在同一个 kernel 里写** —— 少一次发射、少一次参数打包；
#     ② 访存按行连续、宽度是 d，便于向量化。
#
# 【为什么槽位是 -1 就要直接返回】CUDA Graph 按档位录图，真实并发不足一档时
# 会多出几个空槽。这些空槽的槽位被写成 -1，内核碰到它必须跳过 ——
# 否则会按 -1 算出地址、写到别的序列正在用的页上，**静默污染**别人的 KV。
#
# 【为什么一个 program 处理一个 token】写 KV 是纯访存、没有归约需求，
# 「一个 token 一行」是最直观也最容易向量化的映射方式。
# 宽度 d 未必是 2 的幂，所以用 mask 处理尾巴（这也是要传 BLOCK_SIZE 的原因）。
#
# 【⚠️ 一个容易踩的点】`tl.load` 的 `other=0.0` 不能省。
# mask 掉的元素如果没有给定值，读出来是**不确定内容**，而它会被原样写进池子里。
#
# 【读这个文件的顺序】
#     _store_kvcache_kernel（寻址：源侧按 token 行，目标侧按槽位）
#       → store_kvcache（形状与连续性的前置处理，再启动内核）
# ============================================================================

from __future__ import annotations

import torch
import triton
import triton.language as tl


# Triton 内核：D 与 BLOCK_SIZE 在编译期就是常量，便于寄存器分配与向量化。
@triton.jit
def _store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 一个 program 负责一个 token。
    idx = tl.program_id(0)
    # 这个 token 该写到哪个物理槽位。
    slot = tl.load(slot_mapping_ptr + idx)
    # 槽位 -1 = 空槽（补档位padding），必须直接返回。
    if slot == -1:
        return
    # 偏移向量 + mask：d 未必是 2 的幂，尾巴用 mask 挡住。
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < D
    # 源侧：按 token 行取，行内是连续的 [KV头 × head_dim]。
    key_offsets = idx * key_stride + offsets
    value_offsets = idx * value_stride + offsets
    # 取出这个 token 的 K 与 V 两段。
    key = tl.load(key_ptr + key_offsets, mask=mask, other=0.0)
    # other=0.0 必须给：否则 mask 掉的元素是不确定值，会被写进池子。
    value = tl.load(value_ptr + value_offsets, mask=mask, other=0.0)
    # 目标侧：槽位 × 每槽宽度 —— 这一步就是分页布局的寻址。
    cache_offsets = slot * D + offsets
    # K 与 V 在同一个 kernel 里各写一次，省掉一次发射。
    tl.store(k_cache_ptr + cache_offsets, key, mask=mask)
    tl.store(v_cache_ptr + cache_offsets, value, mask=mask)


# 把 [n, KV头, head_dim] 的 K / V 按 slot_mapping 写进池。
def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    # 行内必须连续才能向量化：不连续就先拷一份。
    if not key.is_contiguous():
        key = key.contiguous()
    # 同上，V 也要连续。
    if not value.is_contiguous():
        value = value.contiguous()

    # d = 一个 token 的 KV 宽度（KV 头数 × head_dim）。
    n, num_heads, head_dim = key.shape
    d = num_heads * head_dim
    # 槽位数必须与 token 数一致 —— 错位会静默写串。
    assert slot_mapping.numel() == n

    # 块宽取 d 的上取整 2 次幂，配合内核里的 mask 使用。
    block = triton.next_power_of_2(d)
    # 启动 n 个 program（每 token 一个），把 stride 与 d 一起传进去。
    _store_kvcache_kernel[(n,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        d,
        block,
    )
