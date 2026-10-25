# ============================================================================
# 【本文件是做什么的】MoE 的三个辅助 kernel（JIT 编的 CUDA 扩展）+ 纯 torch 回退。
#
#     align              按专家把 token 重排成"每个专家一整块"，块数向上取整（grouped GEMM 的前置）
#     silu_and_mul       融合 SwiGLU：up * silu(gate)，一步出结果
#     moe_sum            把 [M*top_k, H] 按 top_k 归约回 [M, H]
#     主 GEMM（专家自己的两段矩阵乘）不在这里 —— 交给 Triton / cutlass（见同目录 triton.py）
#
# 【为什么用 JIT 现场编译】源码就在同目录的 `moe_kernels.cu`，用
# `torch.utils.cpp_extension.load` 现编。好处是仓库不用分发二进制 wheel、
# 也不受预编译 wheel 覆盖不到新架构的限制；代价是第一次调用要编译几十秒，
# 之后命中 torch 的编译缓存。
#
# 【为什么每个辅助步骤都要带一个纯 torch 回退】这台机器能编扩展不代表别处能编。
# 回退路径让"没有 CUDA / 编译失败"的环境照样能跑通测试与 CPU 对照，
# 只是慢一些 —— 数值结果必须一致（`_align_torch` 与 kernel 是同语义的两份实现）。
#
# ⚠️ 所有临时张量都按形状池化，不是随手 torch.empty。
#   目的不是"省一点分配开销"，而是让 decode CUDA Graph 捕获时每步的分配次数固定 ——
#   CG 重放要求地址稳定，vLLM 用的是同一个做法。
# ============================================================================
"""JIT CUDA aux kernels for MoE: align / silu / moe_sum.

Source lives next to this file (``moe_kernels.cu``). Main GEMMs stay Triton/cutlass.

Buffers are pooled by shape so decode CUDA Graph capture sees a fixed
allocation count (same approach as vLLM: no fresh ``cudaMalloc`` per step).
"""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F

# CUDA 源码就在本文件旁边（同目录），不进任何 build 系统，靠 torch 运行时现编。
_CSRC = Path(__file__).resolve().parent / "moe_kernels.cu"

# 三组池化缓存，各自对应一个 kernel 的输出/工作区：
#     _ALIGN —— (sorted_ids, expert_ids, num_post) 三元组
#     _SILU  —— SwiGLU 的输出
#     _SUM   —— 按 top_k 归约的输出
# 键里都带设备号：同进程多卡时缓存不能混用。
_ALIGN: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_SILU: dict[tuple, torch.Tensor] = {}
_SUM: dict[tuple, torch.Tensor] = {}


def clear_state() -> None:
    _ALIGN.clear()
    _SILU.clear()
    _SUM.clear()


# 【为什么要算"容量上界"而不是精确长度】CUDA Graph 要求形状固定，
# 但每步重排后有真实 token 的块数会变（路由不均导致的填充量不同）。
# 于是约定：缓冲长度按上界分配、真实长度由 num_post 带回 ——
# 形状固定（利于 CG）、内容长度可变（保证正确）。
def align_capacity(n: int, num_experts: int, block_size: int) -> int:
    """Upper bound on ``sorted_ids`` length for fixed ``n = M * top_k``."""
    if n <= 0:
        return 0
    # 上界推导：n 个 token 每个都可能落在自己的专家上，
    # 每个专家的 token 数都要向上取整到 block_size 的倍数 ⇒ 最坏多出 E*(block_size-1)。
    # 取 int 是为了不把 numpy / torch 的标量类型漏出去（它不是张量，只是个数）。
    cap = n + num_experts * (block_size - 1)
    if n < num_experts:
        cap = min(n * block_size, cap)
    # 返回上界本身；真实长度在 num_post 里，别混用这两个数。
    return int(cap)


# 【池化取块】命中条件比"键存在"更严：形状 / dtype / 设备三者都要一致。
# 建在 `_pool` 里而不是各函数各写一遍，是为了三处缓存的语义完全一样。
def _pool(cache: dict, key: tuple, *, shape, dtype, device) -> torch.Tensor:
    buf = cache.get(key)
    if buf is None or buf.shape != shape or buf.dtype != dtype or buf.device != device:
        buf = torch.empty(shape, dtype=dtype, device=device)
        cache[key] = buf
    return buf


# 【JIT 编译入口】同样只编一次。这个函数里几乎每一行都是"让它在各种机器上真的能编出来"
# 的工程细节，而不是算法 —— 下面逐段说明。
@lru_cache(maxsize=1)
# 【能不能用 CUDA 扩展】只探测一次（lru_cache），结果缓存；
# 环境变量 KRITACE_MOE_CUDA=0 / false 可强制关掉 —— 用来做"扩展 vs 纯 torch"的对照。
def cuda_available() -> bool:
    if os.environ.get("KRITACE_MOE_CUDA", "1") in ("0", "false", "False"):
        return False
    try:
        _ext()
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _ext():
    from torch.utils.cpp_extension import load

    # torch 找不到 nvcc 时的兜底：常见安装路径挨个试。
    # 本机装了 CUDA 13，但 torch 的扩展编译默认仍会去找 CUDA_HOME —— 不设就会报
    # "No CUDA runtime is found" 这类与真实原因无关的错。
    if not (os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")):
        for cand in ("/usr/local/cuda-12.8", "/usr/local/cuda"):
            if Path(cand).exists():
                os.environ["CUDA_HOME"] = cand
                break
    # 把 nvcc 所在目录补进 PATH，否则 nvcc 的子进程（cicc / ptxas）找不到。
    for cand in ("/usr/local/cuda-12.8/bin", "/usr/local/cuda/bin"):
        if Path(cand, "nvcc").exists():
            path = os.environ.get("PATH", "")
            if cand not in path.split(os.pathsep):
                os.environ["PATH"] = cand + os.pathsep + path
            break

    # 【为什么要绕开 ccache】有的环境里 torch 会用 ccache 包装编译器，
    # 而 ccache 与 nvcc 的组合会缓存出错误产物（编译"成功"但 kernel 不对）。
    # 这里临时把 shutil.which 换掉，让 torch 找不到 ccache，从而退化成直接调 nvcc。
    # 注意 finally 里必须还原 —— 不然会影响同进程后续的所有 which 查询。
    _which = shutil.which

    def _no_ccache(cmd, *args, **kwargs):
        if cmd == "ccache":
            return None
        return _which(cmd, *args, **kwargs)

    # 临时替换，作用域只包住下面这次 load。
    shutil.which = _no_ccache  # type: ignore[assignment]
    try:
        return load(
            # 扩展名固定，配合 torch 的缓存目录：同名同源码时第二次启动直接复用。
            name="kritace_moe_cuda",
            sources=[str(_CSRC)],
            # 编译选项分三类：
            #   -O3 / --use_fast_math        —— 常规优化
            #   -U__CUDA_NO_HALF_* 等         —— torch 默认定义这些宏来关掉 half 的隐式转换与运算符重载，
            #                                    -U 掉它们，.cu 里才能直接写 half/bf16 的算术
            #   -gencode=...                 —— 目标架构；不指定的话只能跑当前卡，换卡就要重编
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                # 四档架构一次编全：80 / 86 / 89 / 120a。
                # 89 是本机与服务器（4090D）；120a 是新架构带 accelerated 特性。
                # 多编几档会让首次编译更慢、产物更大 —— 换来的是"换机器不用改代码"。
                "-gencode=arch=compute_80,code=sm_80",
                "-gencode=arch=compute_86,code=sm_86",
                "-gencode=arch=compute_89,code=sm_89",
                "-gencode=arch=compute_120a,code=sm_120a",
            ],
            verbose=bool(os.environ.get("KRITACE_MOE_CUDA_VERBOSE")),
            with_cuda=True,
        )
    finally:
        shutil.which = _which  # type: ignore[assignment]


# 【融合 SwiGLU】输入最后一维是 up 与 gate 拼在一起，输出是 up * silu(gate)。
# 融合的意义：原本要 chunk 出两块、算 silu、再乘，是 3 次访存 + 3 个 kernel；
# 这里一次读、一次写。decode 阶段是访存瓶颈，这一步省的是显存带宽。
def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """``[up|gate]`` → ``up * silu(gate)``."""
    out_shape = (*x.shape[:-1], x.shape[-1] // 2)
    if x.is_cuda and cuda_available():
        # 池化键里带设备号与输出形状：不同 batch 各有一块，形状一致时复用。
        key = (x.device.index or 0, *out_shape, x.dtype)
        out = _pool(_SILU, key, shape=out_shape, dtype=x.dtype, device=x.device)
        _ext().mul_and_silu(out, x.contiguous())
        return out
    up, gate = x.chunk(2, dim=-1)
    return up * F.silu(gate)


# 【按 top_k 归约】融合核的输出是 [M*top_k, H]（每个 token 的 k 条路由各一行），
# 要按行成组相加成 [M, H]。这一步看起来很简单，但它是"每步必做"的，
# 所以给一个专用 kernel，而不是让上层写 .view().sum() 生成临时张量。
def moe_sum(mid: torch.Tensor, M: int, top_k: int) -> torch.Tensor:
    """Reduce ``[M*top_k, H]`` → ``[M, H]``."""
    H = mid.shape[-1]
    if M == 0:
        return mid.new_empty(0, H)
    key = (mid.device.index or 0, M, H, mid.dtype)
    # 先取池化输出（键里含 M —— 不同 batch 各一块）。
    out = _pool(_SUM, key, shape=(M, H), dtype=mid.dtype, device=mid.device)
    # 有 CUDA 才走核；否则下面用 sum(dim=1, out=out) 写进同一块池化缓冲。
    if mid.is_cuda and cuda_available():
        _ext().moe_sum(mid.view(M, top_k, H), out)
        return out
    # 纯 torch 回退：维度含义是「每个 token 的 k 条路由相加」，out= 复用缓冲避免再分配。
    return mid.view(M, top_k, H).sum(dim=1, out=out)


# 【最重要的一步：按专家重排 token】grouped GEMM 之前必须先做这件事。
#
# 为什么必须重排：所有专家共用一次 GEMM 调用（这是"fused"的核心），
# 而 GEMM 是按"每个专家一整块连续行"来切 tile 的。
# 如果 token 顺序还是原始的路由顺序，tile 边界就会落在别人的专家上。
#
# 三个返回值：
#     sorted_ids  —— 重排后每个位置放"原来的第几个 token"，长度是容量
#     expert_ids  —— 每个 block 属于哪个专家（-1 表示这个 block 是空块）
#     num_post    —— 重排后真实用到的长度（容量 ≥ num_post）
# 容量固定 + 长度可变，正是为了兼容 CUDA Graph（见文件头）。
def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack tokens per expert. Returns ``(sorted_ids, expert_ids, num_post)``.

    ``sorted_ids`` / ``expert_ids`` lengths are the *capacity* for this
    ``(n, E, block_size)``, not the live ``num_post``. Capacity is fixed when
    decode pads to a captured batch size, which is what CUDA Graph needs.
    """
    # 展平成一维：[M, top_k] → [M*top_k]，每个元素是一路路由的专家号。
    # 之后所有"排序 / 分组"都在这一维上做，因为它才是真正的行数。
    flat = topk_ids.reshape(-1)
    n = flat.numel()
    device = topk_ids.device
    if n == 0:
        # 空输入直接返回三个空/零张量：不然后面的 kernel 会拿到非法尺寸。
        z = torch.zeros(0, dtype=torch.int32, device=device)
        return z, z, torch.zeros(1, dtype=torch.int32, device=device)

    # 有 CUDA 扩展就走 kernel；否则落到下面的纯 torch 等价实现。
    if device.type == "cuda" and cuda_available():
        # 容量按上界算（见 align_capacity），块数 = 容量 / block_size 向上取整。
        cap = align_capacity(n, num_experts, block_size)
        n_blocks = (cap + block_size - 1) // block_size
        key = (device.index or 0, n, num_experts, block_size)
        bufs = _ALIGN.get(key)
        if (
            bufs is None
            or bufs[0].numel() != cap
            or bufs[1].numel() != n_blocks
            or bufs[0].device != device
        ):
            sorted_ids = torch.empty(cap, dtype=torch.int32, device=device)
            expert_ids = torch.empty(n_blocks, dtype=torch.int32, device=device)
            num_post = torch.empty(1, dtype=torch.int32, device=device)
            _ALIGN[key] = (sorted_ids, expert_ids, num_post)
        else:
            sorted_ids, expert_ids, num_post = bufs
        # 专家号统一成 int32：kernel 的签名要求 int32，而路由那边可能给 int64。
        # 已经是 int32/int64 时尽量不复制（flat 本身就可能已经是 int32）。
        ids = flat if flat.dtype in (torch.int32, torch.int64) else flat.to(torch.int32)
        _ext().moe_align_block_size(
            ids.contiguous(),
            int(num_experts),
            int(block_size),
            sorted_ids,
            expert_ids,
            num_post,
        )
        return sorted_ids, expert_ids, num_post

    # 没有 CUDA 扩展（或强制关闭）时的回退路径。
    return _align_torch(flat, block_size, num_experts)


# 【纯 torch 实现的 align】与 kernel 逐语义对应，只是用张量运算表达。
# 它的存在价值：①无 CUDA 环境也能跑 MoE 的正确性测试；②作为 kernel 的对照实现。
def _align_torch(
    flat: torch.Tensor, block_size: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-torch fallback when the CUDA extension is unavailable."""
    device = flat.device
    flat = flat.to(torch.int32).contiguous()
    n = flat.numel()
    # 只统计合法专家号（0..E-1）。-1 是融合核传来的"不是我的专家"哨兵，
    # 参与分组会把它们塞进一个不存在的专家里，所以先过滤掉。
    valid = (flat >= 0) & (flat < num_experts)
    counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    if bool(valid.any()):
        counts = torch.bincount(flat[valid], minlength=num_experts).to(torch.int32)
    # 每个专家的长度向上取整到 block_size 的倍数 —— 这就是"填充"的来源。
    # 填充的位置在 sorted_ids 里放一个越界下标（见下），GEMM 算出来的行会被丢弃。
    padded = ((counts + block_size - 1) // block_size) * block_size
    starts = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    starts[0] = 0
    # starts[e] = 专家 e 在重排后数组里的起始位置（前缀和）。
    # 有了它，后面判断"某个位置属于哪个专家"只需要一次 searchsorted。
    # cumsum 写进 starts[1:] 是为了给第 0 位留 0 —— 前缀和的经典写法。
    torch.cumsum(padded, dim=0, out=starts[1:])
    num_post = starts[-1:].contiguous()
    cap = align_capacity(n, num_experts, block_size)
    # 先整体填成"越界下标 n"：这些位置上的行会被 GEMM 跳过（配合 expert_ids=-1）。
    # 用 n 而不是 -1 是因为这个值后面要拿去当索引取 hidden（索引必须非负）；
    # -1 只出现在 expert_ids 里那种"按专家查表"的位置。
    sorted_ids = torch.full((cap,), n, dtype=torch.int32, device=device)
    n_blocks = (cap + block_size - 1) // block_size
    if bool(valid.any()):
        # stable=True：同一个专家内部的 token 保持原始相对顺序 ——
        # 这样同一批输入的输出顺序可复现，便于与单卡结果逐步对拍。
        # 排序键就是专家号本身，所以排完就是"按专家分好组的顺序"。
        order = torch.argsort(flat, stable=True)
        keep = (flat[order] >= 0) & (flat[order] < num_experts)
        order = order[keep]
        experts = flat[order]
        compact = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        compact[0] = 0
        torch.cumsum(counts, dim=0, out=compact[1:])
        arange = torch.arange(order.numel(), device=device, dtype=torch.int32)
        # 段内偏移的算法：dest = 该专家段的起点 + 这个 token 在段内的序号。
        # 段内序号 = 排完全局的序号 arange − 这个专家之前已经累计了多少个 token（compact）。
        # 减去的这一项保证了"同一专家的 token 从 starts[e] 开始紧挨着放"。
        dest = starts[experts] + (arange - compact[experts])
        sorted_ids[dest.long()] = order.to(torch.int32)
    # 每个 block 的首个 token 位置（0, block_size, 2*block_size, ...）。
    token_pos = torch.arange(n_blocks, device=device, dtype=torch.int32) * block_size
    # 用 searchsorted 反查这些位置落在哪个专家段里（right=True 取右边界，
    # 与"前缀和区间左闭右开"的约定配套）。不需要遍历，一次二分就够。
    expert_ids = torch.searchsorted(starts[1:].contiguous(), token_pos, right=True).to(
        torch.int32
    )
    # 超出真实长度的 block 标成 -1：融合核看到 -1 就知道这是填充块，整块跳过。
    expert_ids = torch.where(token_pos < num_post, expert_ids, expert_ids.new_full((), -1))
    return sorted_ids, expert_ids, num_post
