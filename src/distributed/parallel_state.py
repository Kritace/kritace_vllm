# ============================================================================
# 【本文件是做什么的】建"进程组"—— 把 N 个进程按 TP / PP 两个维度分好组，
# 之后所有集合通信（all-reduce / all-gather / send / recv）都从这里的组拿句柄。
#
# 【要解决什么问题】单进程单卡时不需要它；要跑 TP / PP，就必须先回答两件事：
#     ① 我这个进程是谁？   → global_rank 由 (pp_rank, tp_rank) 唯一确定
#     ② 我该跟谁通信？     → TP 组 = 同一 PP 段内连续的 tp_size 个进程
#                            PP 组 = 每个 TP 位置各出一个，串成一条流水线
#   本文件用全局 rank 的算术回答这两问，不依赖任何外部布局描述：
#       global_rank = pp_rank * tp_size + tp_rank
#       TP 组 = [pp_rank * tp_size + k for k in range(tp_size)]
#       PP 组 = [k * tp_size + tp_rank for k in range(pp_size)]
#   这跟真实 vLLM 的做法一致 —— 进程拓扑是算出来的，不是配置文件里写出来的。
#
# 【为什么每个组要带两个 backend】device_group 走 NCCL（GPU 张量），
#   cpu_group 走 Gloo（CPU 张量 / 控制消息）。NCCL 传不了 CPU 张量，
#   所以凡是要传字节或小标量的地方，都必须另建一个 Gloo 组。
#
# 【DP 为什么不在这个文件里】DP 的每个副本是一个独立的 Engine，
#   各自有自己的一套 WORLD / TP / PP（见 control/mp_client.py）。
#   本文件管的是"一个 Engine 内部"的切分，所以只有 TP 与 PP 两维。
#
# 【size == 1 时不建组】单进程时所有通信函数直接短路返回：
#   既不建 NCCL clique（省启动时的建链与握手），也避免出现"1 个成员的组"。
#
# 【⚠️ 三个容易踩的点】
#     ① NCCL 的 new_group 是集合调用：组内所有 rank 都必须执行到，
#        否则没走到的那一侧会一直等 —— 表现为启动就卡住，且不报错。
#     ② rank 有两种口径：WORLD rank 与组内 rank。
#        NCCL 点对点（group_peer=）要组内序号；Gloo 的 send/recv 要 WORLD rank。
#        混用会"发到错误的进程"或直接挂住。
#     ③ size == 1 的组一律不建，靠通信函数里的短路兜住。
#        所以"只有一张卡，随便调通信也没事"是被短路救了，不是真的没事。
#
# 【读这个文件的顺序】
#     Group（组的数据结构，以及谁拥有它）
#       → WORLD / TP / PP 三个全局单例与 getter
#       → initialize_model_parallel（按算术规则切出 TP/PP）
#       → init_distributed_environment（NCCL/Gloo 的真正初始化）
#       → graph_capture（录图前后的全局对齐）
#       → set_nccl_env（NCCL 调优项）
# ============================================================================

from __future__ import annotations

import datetime
import logging
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

# 本模块的 logger：建组与初始化过程中的告警都从这里出去。
logger = logging.getLogger(__name__)

# ==============================================================================
# 单 Engine 内的 TP × PP 进程组。
#
# Worker.init_environment 调 init_distributed_environment + initialize_model_parallel。
# global_rank = pp_rank * tp_size + tp_rank。
# DP 不在这里建组（MPClient 每个副本各自一套 WORLD）。
# ==============================================================================


# 一个进程组：记下成员、自己的角色，以及 NCCL / Gloo 两个通信句柄。
class Group:
    """一组 rank：device_group 走 NCCL，cpu_group 走 Gloo。size==1 不建组。"""

    # ranks 是本组的 WORLD rank 列表（顺序即组内 rank 顺序）；rank 是本进程的 WORLD rank。
    def __init__(
        self,
        ranks: list[int],
        rank: int,
        device_group: ProcessGroup | None = None,
        cpu_group: ProcessGroup | None = None,
    ):
        # 组内成员，顺序就是组内 rank 顺序
        self.ranks = ranks
        # 自己（WORLD rank）
        self.rank = rank
        # 组大小。下面每个通信函数的第一件事都是判它是否等于 1
        self.size = len(ranks)
        # 自己的组内序号（0..size-1）—— 跟 NCCL 打交道要用它，不要用 WORLD rank
        self.group_rank = ranks.index(rank)
        # 这两个标志位决定 destroy 时该销毁谁：自己新建的才拥有
        self._owns_device_group = False
        self._owns_cpu_group = False
        # 外部传进来的句柄（例如 WORLD 复用 torch 的默认组）只是托管，不拥有它
        if device_group is not None:
            self.device_group = device_group
        # 否则自己建一个 NCCL 组。注意 new_group 是集合调用，组内所有 rank 都要走到这里
        elif self.size > 1:
            self.device_group = dist.new_group(ranks=ranks)
            self._owns_device_group = True
        else:
            self.device_group = None
        # Gloo 组同理。它专门服务 CPU 张量与控制消息（NCCL 传不了 CPU 张量）
        if cpu_group is not None:
            self.cpu_group = cpu_group
        elif self.size > 1:
            self.cpu_group = dist.new_group(ranks=ranks, backend="gloo")
            self._owns_cpu_group = True
        else:
            self.cpu_group = None

    # 下面四个 property 把「我在流水线的哪一段」翻译成不同口径的 rank
    @property
    def is_first_rank(self) -> bool:
        return self.group_rank == 0

    # 上游 / 下游的组内序号：NCCL 点对点（group_peer=）认的是这个口径
    @property
    def is_last_rank(self) -> bool:
        return self.group_rank == self.size - 1

    # 下游同理：走到最后一段就没有下一个了
    @property
    def prev_group_rank(self) -> int:
        """PP 上游，给 NCCL group_peer=。线性流水线，第一段没有前驱。"""
        # 已经是第一段还想要上游 → 报错，而不是悄悄绕到另一端去
        if self.group_rank <= 0:
            raise RuntimeError("first PP stage has no previous peer")
        return self.group_rank - 1

    # 再翻译成 WORLD rank：Gloo 的 send/recv 要全局 rank，NCCL 要组内序号 —— 两者别混用
    @property
    def next_group_rank(self) -> int:
        """PP 下游，给 NCCL group_peer=。"""
        if self.group_rank >= self.size - 1:
            raise RuntimeError("last PP stage has no next peer")
        return self.group_rank + 1

    @property
    def prev_rank(self) -> int:
        """上游的 WORLD rank，给 Gloo send/recv。NCCL 不要用这个。"""
        return self.ranks[self.prev_group_rank]

    @property
    def next_rank(self) -> int:
        return self.ranks[self.next_group_rank]

    # 组内同步点。size==1 或没有 Gloo 组时是空操作（不是错误）
    def barrier(self) -> None:
        # 用 cpu_group 做 barrier：控制面的同步不该占 GPU 的通信资源
        if self.size > 1 and self.cpu_group is not None:
            dist.barrier(self.cpu_group)

    # 只销毁自己建的组。销毁复用来的（WORLD 的默认组）会连带把 torch 的默认组打掉
    def destroy(self) -> None:
        if self._owns_device_group and self.device_group is not None:
            dist.destroy_process_group(self.device_group)
        if self._owns_cpu_group and self.cpu_group is not None:
            dist.destroy_process_group(self.cpu_group)
        self.device_group = None
        self.cpu_group = None


# 三个全局单例。用「模块级变量 + getter」是刻意的：任何模块 import 后拿到的都是同一份组
WORLD: Group | None = None
# TP 组：同一个 PP 段内连续的 tp_size 个进程，all-reduce 走它
TP: Group | None = None
# PP 组：每个 tp_rank 位置各出一个，跨 stage 传 hidden 走它
PP: Group | None = None
_ENABLE_EP = False


# ============================================================================
# 【EP 是"翻一个布尔"，不是新建通信组】
# 关键在于 MoE 的 EP 复用现有的 TP 组：Attention 仍在那组 rank 上做 all-reduce，
# MoE 只是把同一组 rank 的切法从"切中间维"换成"切专家"。
# 因此这里不调 new_group、world_size 也不变 —— 只翻一个模块级标志。
# ============================================================================
def set_expert_parallel(enabled: bool) -> None:
    """``--ep``：MoE 在现有 TP ranks 上切 expert，不新建 NCCL 组。"""
    global _ENABLE_EP
    _ENABLE_EP = bool(enabled)


# 谁查它：MoE 侧用它决定 forward 里"先 all-reduce 再加共享专家"的顺序
# （见 moe/block.py），以及 get_ep_group() 决定返回真组还是占位组。
def expert_parallel_enabled() -> bool:
    return _ENABLE_EP


# 三个 getter 都带 assert：忘了初始化就应该在这里炸，而不是把 None 传进通信函数里
def get_world_group() -> Group:
    assert WORLD is not None, "Distributed process group is not initialized."
    return WORLD


def get_tp_group() -> Group:
    assert TP is not None, "Tensor parallel process group is not initialized."
    return TP


def get_pp_group() -> Group:
    assert PP is not None, "Pipeline parallel process group is not initialized."
    return PP


# 【EP 的组从哪来】开了 EP 就直接把 TP 组当 EP 组用 —— 两者是同一批 rank，
# 区别只在 MoE 层内部怎么切（见上面的说明）。
# 没开 EP 时不能返回 None：调用方会无条件拿 size 做判断，所以给一个 size=1 的占位组。
def get_ep_group() -> Group:
    """EP 复用 TP；未开 ``--ep`` 时返回 size=1 的占位组。"""
    # 先拿 TP 组：它既是 EP 的宿主，也是占位组的模板。
    tp = get_tp_group()
    if _ENABLE_EP:
        return tp
    # 占位组：只含自己一个 rank ⇒ size == 1，
    # 于是 MoE 那边"要不要跨卡归约"的判断自然为假，调用方不需要写额外分支。
    return Group(ranks=[tp.rank], rank=tp.rank)


# 按算术规则把 rank 网格切成 TP / PP 两个组。输入只有 rank 与 size，没有别的布局描述
def initialize_model_parallel(
    tp_size: int,
    pp_size: int,
    tp_rank: int,
    pp_rank: int,
) -> None:
    """
        TP：同一 pp_rank 上连续 tp_size 个 global rank。
        PP：同一 tp_rank 上每隔 tp_size 取一个。
        pp==1 时 TP 复用 WORLD 的 NCCL 组，少建一次 clique。
    """
    # 只有一份全局实例，所以这里直接改模块级变量
    global TP, PP
    # 先把非法 rank 挡掉：越界的 rank 会静默算出「看起来合法」的组，事后极难查
    if not (0 <= tp_rank < tp_size and 0 <= pp_rank < pp_size):
        raise ValueError(
            f"invalid ranks: tp_rank={tp_rank}/{tp_size}, pp_rank={pp_rank}/{pp_size}"
        )
    # global_rank 的算法：PP 优先、TP 连续。整个 rank 布局都由这一行定义
    my_rank = pp_rank * tp_size + tp_rank
    # TP 组 = 同一个 pp_rank 下连续的 tp_size 个 rank
    tp_ranks = [pp_rank * tp_size + k for k in range(tp_size)]
    # PP 组 = 每个 tp_rank 位置上每隔 tp_size 取一个
    pp_ranks = [k * tp_size + tp_rank for k in range(pp_size)]
    # pp_size == 1 时 TP 组恰好等于整个 WORLD → 直接复用它的句柄，省一次 NCCL clique 的建链与握手
    if WORLD is not None and tp_ranks == WORLD.ranks:
        TP = Group(
            ranks=tp_ranks,
            rank=my_rank,
            device_group=WORLD.device_group,
            cpu_group=WORLD.cpu_group,
        )
    else:
        TP = Group(ranks=tp_ranks, rank=my_rank)
    # PP 组按上面算出的 pp_ranks 建，NCCL 与 Gloo 两份句柄由 Group 内部各建一个
    PP = Group(ranks=pp_ranks, rank=my_rank)


# 销毁顺序与建立相反；最后置回 None，让 getter 重新按「未初始化」报错
def destroy_model_parallel() -> None:
    global TP, PP
    set_expert_parallel(False)
    if TP is not None:
        TP.destroy()
    if PP is not None:
        PP.destroy()
    TP = None
    PP = None


# 真正初始化分布式环境：建 torch 的 world 与默认组。启动时调一次，可重复调用
def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    local_rank: int = -1,
    backend: str = "nccl",
    init_method: str = "env://",
    timeout: int = 600,
) -> None:
    global WORLD
    # 重复初始化直接返回 —— 多进程启动路径上很容易被调两次
    if dist.is_initialized():
        logger.info("Distributed process group is already initialized.")
        return
    # 用哪张卡：优先 local_rank（多机时只有它能区分本机内的卡号）
    device_id = local_rank if local_rank >= 0 else rank
    # NCCL 必须先把进程绑到指定 GPU，否则集合通信可能跑错卡
    if backend == "nccl":
        torch.cuda.set_device(device_id)
    # 建默认组。timeout 单位是秒；传 device_id 让 NCCL 走「不显式传 cuda 张量也能通信」的新路径
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=timeout),
        device_id=torch.device("cuda", device_id) if backend == "nccl" else None,
    )
    # 想把默认组包成 Group 复用，得先把它取出来
    default_device_group: ProcessGroup | None = None
    # 只有 NCCL 才有 GPU 侧默认组；Gloo 场景下 WORLD 只有 CPU 通道
    if backend == "nccl":
        from torch.distributed.distributed_c10d import _get_default_group

        default_device_group = _get_default_group()
    # WORLD 复用 torch 建好的默认组，因此不拥有它 → destroy 时不会重复销毁
    WORLD = Group(
        ranks=list(range(world_size)), rank=rank, device_group=default_device_group
    )


# 顺序不能反：先销毁 WORLD 的封装，再销毁 torch 的默认组
def destroy_distributed_environment() -> None:
    global WORLD
    if WORLD is not None:
        WORLD.destroy()
    WORLD = None
    if dist.is_initialized():
        dist.destroy_process_group()


# 直接透传，给上层判断「是否已建组」
def is_initialized() -> bool:
    return dist.is_initialized()


# 录 CUDA Graph 前后的全局对齐 —— 多卡 + 图捕获的关键小心点
@contextmanager
def graph_capture(device: torch.device):
    """录 CUDA Graph 前后对齐 TP/PP，避免图内 all-reduce 各 rank 步调不一致。PP send/recv 仍在图外。"""
    # 先把本卡上的异步工作排空，否则录图起点可能夹着上一段的 kernel
    torch.cuda.synchronize(device)
    # 进图前 TP / PP 两个维度都要对齐：图内包含 all-reduce，各 rank 必须同时开始录
    get_tp_group().barrier()
    get_pp_group().barrier()
    try:
        yield
    finally:
        torch.cuda.synchronize(device)
        get_tp_group().barrier()
        get_pp_group().barrier()


# 只设 NCCL 的环境变量，而且一律 setdefault —— 用户显式设过的值不被覆盖
def set_nccl_env(ifname: str = "", *, disable_ib: bool = False) -> None:
    # 内层小工具，纯粹为了少写几遍 os.environ.setdefault
    def _set(name: str, value: str) -> None:
        os.environ.setdefault(name, value)

    # 多网卡机器必须指定通信网卡名，否则 NCCL 可能挑到不对的那张
    if ifname:
        _set("NCCL_SOCKET_IFNAME", ifname)
    # 关掉 InfiniBand 走 TCP；本地没有 IB 时必须关，否则 NCCL 会卡在设备探测
    if disable_ib:
        _set("NCCL_IB_DISABLE", "1")
    else:
        _set("NCCL_IB_GID_INDEX", "3")
        _set("NCCL_NET_GDR_LEVEL", "5")
    # 默认只打 WARN：NCCL 的 INFO 日志刷屏非常厉害
    _set("NCCL_DEBUG", "WARN")
    # 心跳超时放宽到 30 分钟 —— 大模型加载/编译很容易超过默认 10 分钟而被误判成挂死
    _set("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "1800")
