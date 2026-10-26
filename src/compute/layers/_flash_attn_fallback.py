"""flash-attn 的纯 PyTorch 回退实现。

为什么有这个文件：本机（RTX 3050 Laptop, CUDA 13）没有 flash-attn 的可用 wheel，
装不上就整个引擎都 import 不了。这里用 `torch.nn.functional.scaled_dot_product_attention`
实现与 attention.py 用到的两个接口同名的函数：

- `flash_attn_varlen_func`：prefill 用，变长打包 + 可选 paged KV
- `flash_attn_with_kvcache`：decode 用，q 长度恒为 1，K/V 从页表里取

语义与 flash-attn 一致（因果掩码、GQA 广播、paged 寻址），常数因子差很多，
只用于功能验证与对拍，**不要**用它跑性能结论。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _gqa_expand(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """把 KV 的头数广播到 Q 的头数（GQA/MQA）。x: [..., h_kv, d]。"""
    h_kv = x.shape[-2]
    if h_kv == n_heads:
        return x
    return x.repeat_interleave(n_heads // h_kv, dim=-2)


def _gather_pages(pool: torch.Tensor, pages: torch.Tensor, length: int) -> torch.Tensor:
    """从 paged 池子里按 block_table 取出前 length 行。

    pool: [num_blocks, block_size, h, d]；pages: [max_blocks]。
    """
    block_size = pool.shape[1]
    used = (length + block_size - 1) // block_size
    idx = pages[:used].long()
    flat = pool.index_select(0, idx).reshape(-1, *pool.shape[2:])
    return flat[:length]


def _sdpa(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
          q_len: int, k_len: int) -> torch.Tensor:
    """单条序列的因果注意力。queries: [q_len, h, d] → [q_len, h, d]。

    q 的每个位置只能看到「它之前」的 k；q 与 k 尾部对齐（decode 时 q_len=1）。
    """
    h = queries.shape[1]
    q = queries.transpose(0, 1).unsqueeze(0)          # [1, h, q_len, d]
    k = _gqa_expand(keys, h).transpose(0, 1).unsqueeze(0)   # [1, h, k_len, d]
    v = _gqa_expand(values, h).transpose(0, 1).unsqueeze(0)
    mask = torch.ones(q_len, k_len, dtype=torch.bool, device=queries.device)
    mask = torch.tril(mask, diagonal=k_len - q_len)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return out.squeeze(0).transpose(0, 1).contiguous()


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    causal: bool = False,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    **_ignored,
) -> torch.Tensor:
    """变长打包的因果注意力。q: [total_q, h_q, d]；k/v: [total_k, h_kv, d]
    或 paged 池子 [num_blocks, block_size, h_kv, d]（给了 block_table 时）。
    """
    n_seq = cu_seqlens_q.numel() - 1
    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()
    out = torch.empty_like(q)
    for i in range(n_seq):
        q0, q1 = cu_q[i], cu_q[i + 1]
        if q0 == q1:
            continue
        k0, k1 = cu_k[i], cu_k[i + 1]
        if block_table is not None:
            keys = _gather_pages(k, block_table[i], k1)
            vals = _gather_pages(v, block_table[i], k1)
        else:
            keys = k[k0:k1]
            vals = v[k0:k1]
        # 注意 k_len 必须取 keys 的实际行数：带前缀缓存时 cu_seqlens_k 是"累计"值，
        # 而 k0 可能不为 0（k[k0:k1] 只有 k1-k0 行），两者不能混用。
        out[q0:q1] = _sdpa(q[q0:q1], keys, vals, q1 - q0, keys.shape[0])
    return out


def _with_kvcache_batched(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    """一次把整批的 KV 从页表里 gather 出来，用布尔掩码裁掉无效位。

    与逐序列循环的区别：**不读 cache_seqlens 的数值**（不做 .tolist()），
    长度信息只以"上界 + 掩码"的形式参与计算，因此 CPU 不会被同步打断。
    """
    bs, q_len, n_heads, head_dim = q.shape
    block_size = k_cache.shape[1]
    n_blocks = (max_len + block_size - 1) // block_size
    pages = block_table[:, :n_blocks].long()
    neg = pages < 0
    offsets = torch.arange(block_size, device=q.device)
    flat = (pages.clamp(min=0).unsqueeze(-1) * block_size + offsets).reshape(
        bs, n_blocks * block_size
    )
    pad = neg.repeat_interleave(block_size, dim=1)
    # 页是分配单位（这里 256 行），但本批最长也就 max_len 行：
    # 不截断的话要按整页算，白算十几倍（大显存卡上不明显，4 GB 卡上非常明显）。
    flat = flat[:, :max_len]
    pad = pad[:, :max_len]

    k_flat = k_cache.reshape(-1, *k_cache.shape[2:])
    v_flat = v_cache.reshape(-1, *v_cache.shape[2:])
    idx = flat.reshape(-1)
    keys = k_flat.index_select(0, idx).view(bs, max_len, -1, head_dim)
    vals = v_flat.index_select(0, idx).view(bs, max_len, -1, head_dim)

    pos = torch.arange(max_len, device=q.device)
    valid = pos.unsqueeze(0) < cache_seqlens.view(-1, 1)
    # padding 行的 cache_seqlens=0：给它留一个位置，避免整行全掩码出 NaN。
    valid = (valid & ~pad) | (pos.unsqueeze(0) == 0)
    mask = valid.view(bs, 1, 1, -1)

    keys = _gqa_expand(keys, n_heads).transpose(1, 2)
    vals = _gqa_expand(vals, n_heads).transpose(1, 2)
    out = F.scaled_dot_product_attention(q.transpose(1, 2), keys, vals, attn_mask=mask)
    return out.transpose(1, 2).contiguous()


def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    causal: bool = False,
    max_seqlen_k: int = 0,
    **_ignored,
) -> torch.Tensor:
    """decode：q [bs, 1, h, d]，K/V 从 paged 池子里取（含刚写进去的这一格）。

    返回 [bs, 1, h, d]，与 flash-attn 的返回形状一致。
    """
    bs, q_len = q.shape[0], q.shape[1]
    # 有上界且是标准 decode（每序列 1 个 query）时走无同步的批量路径。
    if max_seqlen_k > 0 and q_len == 1:
        return _with_kvcache_batched(
            q, k_cache, v_cache, cache_seqlens, block_table, int(max_seqlen_k)
        )
    # 旧路径：逐序列 gather，需要 .tolist() —— 会打断流水，只在没法给上界时使用。
    lens = cache_seqlens.tolist()
    out = torch.empty_like(q)
    for i in range(bs):
        length = int(lens[i])
        if length <= 0:
            out[i].zero_()
            continue
        keys = _gather_pages(k_cache, block_table[i], length)
        vals = _gather_pages(v_cache, block_table[i], length)
        q_i = q[i, :q_len]
        out[i, :q_len] = _sdpa(q_i, keys, vals, q_len, length)
    return out
