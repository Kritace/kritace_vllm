# ============================================================================
# 【本文件是做什么的】把 torch.distributed 的通信原语**包一层**，统一三件事：
#     rank 口径、单卡短路、以及"只在某一维拼接"的维度搬运。
#
# 【为什么不能直接调 torch.distributed】
#     ① 单卡的短路：size == 1 时不该建组、也不该调用集合通信（部分后端会报错）。
#     ② all_gather / reduce_scatter 的底层只认**第 0 维**：想沿别的维拼，
#        得自己把维度搬来搬去。这层就是把这件事**收口到一个地方**。
#     ③ 调用方不该关心"我到底属于哪个组"：TP 专用的包装在这里一次性提供。
#
# 【两条通信路径，别搞混】
#     TP（层内）：all_reduce / all_gather / reduce_scatter —— 集合通信，走 NCCL
#     PP（层间）：send_tensors / recv_tensors        —— 点对点，跨 stage 传 hidden
#   TP 是"权重切分 + 每层一次 all-reduce"；PP 是"层切分 + stage 边界传激活"。
#   两者的通信量量级完全不同（前者与 batch×hidden 成正比、每层一次；
#   后者每步只有 hidden 大小的一两次），这也是实测里 TP 与 PP 表现相反的根因之一。
#
# 【⚠️ 两个容易踩的点】
#     ① 点对点的 peer 参数是**组内** rank（PP 场景下用 next_group_rank / prev_group_rank），
#        不是 WORLD rank。传错不会报错，只会让对端一直等（看起来像死锁）。
#     ② 零长张量要跳过：numel() == 0 的张量在部分后端上会出错，而且它本来也没有传输意义。
#
# 【读这个文件的顺序】
#     _resolve（组为空时按 WORLD 处理）
#       → all_gather / all_reduce / reduce_scatter（三个集合通信原语 + 维度搬运）
#       → tensor_model_parallel_*（TP 专用包装）
#       → send_tensors / recv_tensors / _p2p（PP 点对点）
# ============================================================================

from __future__ import annotations

import torch
import torch.distributed as dist

from .parallel_state import Group, get_tp_group, get_world_group

# ==============================================================================
# NCCL 集合通信与 PP 点对点。
#
# Linear / vocab 走 tensor_model_parallel_*（TP 组）。
# ModelRunner 图外 PP 走 send_tensors / recv_tensors。
# size==1 直接返回。
# ==============================================================================


# 组参数留空时一律理解为 WORLD：调用方不传组，就按全局通信处理。
def _resolve(group: Group | None) -> Group:
    return get_world_group() if group is None else group


# all_gather：把各 rank 的分片沿某一维拼起来。难点全在「沿哪一维」上。
def all_gather(
    input: torch.Tensor, dim: int = -1, group: Group | None = None
) -> torch.Tensor:
    """沿 dim 拼接各 rank 分片。NCCL 只在 dim0 拼接，其它维先按 rank 堆再 permute。"""
    # size == 1 直接返回：单卡不需要通信，也不建组。
    group = _resolve(group)
    if group.size == 1:
        return input
    # 允许传负数维（Python 习惯写法），这里统一换算成正数。
    if dim < 0:
        dim += input.dim()
    # contiguous 是必须的：NCCL 只认连续内存。
    src = input.contiguous()
    # 先把输出按「rank 维放在最前」的形状开出来，之后再决定要不要搬维。
    out = torch.empty(
        (src.shape[0] * group.size, *src.shape[1:]),
        dtype=src.dtype,
        device=src.device,
    )
    # all_gather_single 是单张量版本，避开了 all_gather 的 list 打包开销。
    dist.all_gather_single(out, src, group=group.device_group)
    # 拼接维正好是第 0 维，直接用，不用搬。
    if dim == 0:
        return out
    # 其它维的走法：先把 rank 维单独 view 出来 → movedim 搬到目标维 → reshape 压平。
    return (
        out.view(group.size, *src.shape)
        .movedim(0, dim)
        .reshape(src.shape[:dim] + (src.shape[dim] * group.size,) + src.shape[dim + 1 :])
    )


# all_reduce：**原地**累加，所以没有额外的输出分配。
def all_reduce(
    input: torch.Tensor, op=dist.ReduceOp.SUM, group: Group | None = None
) -> torch.Tensor:
    # size == 1 短路。
    group = _resolve(group)
    if group.size == 1:
        return input
    # 原地改输入张量 —— 也就意味着调用方传进来的那块内存会被就地改写。
    dist.all_reduce(input, op, group=group.device_group)
    return input


# reduce_scatter：all-reduce 的省流量变体 —— 先归约，再把结果按 rank 切开各留一份。
def reduce_scatter(
    input: torch.Tensor,
    dim: int = -1,
    op=dist.ReduceOp.SUM,
    group: Group | None = None,
) -> torch.Tensor:
    # size == 1 短路。
    group = _resolve(group)
    if group.size == 1:
        return input
    # 负数维归一化，同上。
    if dim < 0:
        dim += input.dim()
    # reduce_scatter 只能沿第 0 维切，所以先把目标维搬到 0。
    src = input.movedim(dim, 0).contiguous()
    # 第 0 维必须能被组大小整除，否则切不匀 —— 这里直接 assert，而不是悄悄补零。
    assert src.shape[0] % group.size == 0
    # 输出只有输入的 1/size 大，这正是它比 all-reduce 省流量的地方。
    out = torch.empty(
        (src.shape[0] // group.size,) + src.shape[1:],
        dtype=src.dtype,
        device=src.device,
    )
    # reduce_scatter_single：同样避开了 list 版本的一次打包。
    dist.reduce_scatter_single(out, src, op=op, group=group.device_group)
    # 结果搬回原来所在的维度位置；dim != 0 时会多一次 contiguous。
    return out if dim == 0 else out.movedim(0, dim).contiguous()


# 下面三个是 TP 专用包装：组从全局拿，调用方不用自己传组。
def tensor_model_parallel_all_reduce(
    input: torch.Tensor, op=dist.ReduceOp.SUM
) -> torch.Tensor:
    return all_reduce(input, op, group=get_tp_group())


def tensor_model_parallel_all_gather(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return all_gather(input, dim=dim, group=get_tp_group())


def tensor_model_parallel_reduce_scatter(
    input: torch.Tensor, dim: int = -1, op=dist.ReduceOp.SUM
) -> torch.Tensor:
    return reduce_scatter(input, dim=dim, op=op, group=get_tp_group())


# send：PP 边界传 hidden / residual。**阻塞**语义 —— 所有 work.wait() 走完才算发完。
def send_tensors(
    tensors: tuple[torch.Tensor, ...] | list[torch.Tensor],
    dst: int,
    group: Group | None = None,
) -> None:
    """阻塞发送。dst 是组内 rank（PP 用 next_group_rank），不是 WORLD rank。"""
    # _p2p 返回一批 work，逐个 wait 才是真正完成。
    for work in _p2p(dist.isend, tensors, dst, group):
        work.wait()


# recv 与 send 完全对称，只差方向。
def recv_tensors(
    tensors: tuple[torch.Tensor, ...] | list[torch.Tensor],
    src: int,
    group: Group | None = None,
) -> None:
    """阻塞接收。src 是组内 rank（PP 用 prev_group_rank）。"""
    for work in _p2p(dist.irecv, tensors, src, group):
        work.wait()


# 统一的点对点实现：send 与 recv 只差传进来的 op（isend / irecv）。
def _p2p(
    op,
    tensors: tuple[torch.Tensor, ...] | list[torch.Tensor],
    peer: int,
    group: Group | None,
) -> list:
    # 组大小为 1 时没有通信可做，返回空列表 —— 调用方的 for 循环自然空转。
    group = _resolve(group)
    if group.size == 1:
        return []
    # peer 是**组内**序号，越界要立刻报错：越界不会崩，只会让对端一直等。
    if not 0 <= peer < group.size:
        raise ValueError(f"peer={peer} not in [0, {group.size})")
    ops = [
        dist.P2POp(op, t, group=group.device_group, group_peer=peer)
        for t in tensors
        # 跳过零长张量：没有传输意义，且部分后端会报错。
        if t.numel() > 0
    ]
    # ops 为空时不要调用 batch_isend_irecv —— 空列表会让部分后端报错。
    return list(dist.batch_isend_irecv(ops)) if ops else []
