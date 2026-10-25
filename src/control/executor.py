# ============================================================================
# 【本文件是做什么的】Executor = **执行面与门面之间的桥**。它把上层那句
# 「跑一步」翻译成两种完全不同的实现：
#     单卡（tp=pp=1）→ UniprocExecutor：同进程直接函数调用，零通信
#     多卡（tp×pp>1）→ MultiprocExecutor：rank0 广播调度指令，各 rank 各自执行前向，
#                      采样结果再回传给 rank0
# EngineCore 只依赖 Executor 这 5 个方法，完全感知不到底下有几个进程。
#
# 【为什么 rank0 要身兼两职（driver + 本地 worker）】
#   多进程如果再多开一个"纯 driver"进程，就要多一次转发、多一份内存。
#   让 rank0 一边广播一边干自己那份活，是省资源的选择 ——
#   代价是 rank0 的时序要小心：**必须先广播再本地执行**，
#   否则子进程在等命令、而 rank0 已经开始算自己的了。
#
# 【采样为什么只在一个 rank 上做】只有**最后一段 PP** 持有 lm_head、
#   也只有它能拿到完整 logits。所以采样固定落在末段 PP 的 tp0 上：
#   `sampler_rank = (pp_size - 1) * tp_size`（rank 编号是 pp 优先、tp 连续）。
#
# 【控制消息为什么走 Gloo 而不是 NCCL】命令内容是 CPU 侧的小数据（序列化后的
#   调度结果、token id 列表），NCCL 传不了 CPU 张量；Gloo 的小消息延迟也更低。
#
# 【⚠️ 四个容易踩的点】
#     ① 子进程的退出必须走「terminate → join → kill → join」四段式：
#        CUDA / NCCL 阻塞中的进程常常不响应 SIGTERM，只有 SIGKILL 能收掉。
#     ② `_owns_workers` 区分 run.py 与 torchrun 两条路径。误杀外部管理的进程，
#        会连带把别人的任务一起干掉。
#     ③ 采样结果回传是**阻塞**的（driver 侧 recv 会等）→ 这一步无法与下一次
#        调度重叠。这是 v4 多卡路径的吞吐上限之一，不是 bug。
#     ④ FileStore rendezvous 不要求 rank0 先监听，所以可以**先起子进程、再初始化
#        rank0**；反过来写也能跑，但会白等一段时间。
#
# 【读这个文件的顺序】
#     _sampler_rank / _terminate_procs（两个小工具）
#       → ControlCommand（控制消息的格式）
#       → Executor（接口契约）
#       → _spawn_worker / worker_loop（子进程里跑什么）
#       → UniprocExecutor（最短路径）
#       → MultiprocExecutor（rank0 的两副面孔）
#       → create_executor（怎么选）
# ============================================================================

from __future__ import annotations

import multiprocessing as mp
from abc import ABC, abstractmethod

import msgspec
import torch
import torch.distributed as dist

from ..compute.worker import Worker
from ..config import EngineConfig
from ..data import KVManager, ScheduledBatch, SchedulerOutput
from ..distributed import ControlChannel, get_world_group

# ==============================================================================
# Executor：EngineCore 与 Worker 之间的执行桥。
#
# EngineCore.step 调用 run / read_samples，本模块负责将这两个调用
# 路由到实际的 Worker（持有模型权重、执行前向和采样的进程）。
#
# 分叉逻辑只看 tp × pp，不涉及 DP：
#   tp=1, pp=1 → UniprocExecutor：同进程直接调用，无通信开销。
#   否则       → MultiprocExecutor：rank0 兼做 driver 与本地 Worker，
#                  通过 Gloo 将 SchedulerOutput 广播给所有 rank，
#                  各 rank 独立执行前向，采样结果回传 rank0。
# ==============================================================================


# 采样固定落在**最后一段 PP 的 tp0** 上 —— 只有那里有完整 logits。
def _sampler_rank(tp_size: int, pp_size: int) -> int:
    """
        采样发生在全局哪个 rank。

        只有最后一个 PP stage 才有完整的 logits，
        采样固定落在末 PP stage 的 tp0 上。
    """
    # rank 编号规则是 pp 优先、tp 连续，所以末段 tp0 = (pp-1) * tp。
    return (pp_size - 1) * tp_size


# 优雅关闭子进程：先礼后兵。
def _terminate_procs(procs: list[mp.Process]) -> None:
    """
        优雅关闭子进程。

        先发 SIGTERM 让子进程自行清理（释放 CUDA 上下文、关闭 NCCL 通信器等），
        等待 3 秒后若仍存活则 SIGKILL 强杀，
        避免子进程卡在 CUDA / NCCL 阻塞调用上导致主进程无法退出。
    """
    # 第一轮发 SIGTERM，让子进程自己清理（释放 CUDA 上下文、关 NCCL 通信器）。
    for p in procs:
        if p.is_alive():
            p.terminate()
    # 等 3 秒 —— 给清理留时间，但不无限等。
    for p in procs:
        p.join(timeout=3)
    # 还活着就 SIGKILL：卡在 CUDA / NCCL 阻塞调用里的进程往往不响应 SIGTERM。
    for p in procs:
        if p.is_alive():
            p.kill()
    # 收尸，最多再等 5 秒。
    for p in procs:
        p.join(timeout=5)
    # 清空列表，避免下次重复处理。
    procs.clear()


# ------------------------------------------------------------------ #
# 控制消息
# ------------------------------------------------------------------ #


# 控制消息的形状：字段都留可选，靠 op 区分语义。
class ControlCommand(msgspec.Struct):
    """
        driver（rank0）通过 Gloo 广播给所有 Worker 的控制消息。
        用 msgspec 序列化，走 ControlChannel（CPU 后端）
    """

    op: str                              # "execute" | "reset_kv" | "shutdown"
    sched: SchedulerOutput | None = None  # execute 时携带的调度数据
    prefix_backend: str | None = None     # reset_kv 时可选切换缓存后端


# ------------------------------------------------------------------ #
# Executor 抽象基类
# ------------------------------------------------------------------ #


# 统一接口：EngineCore 只依赖这组方法，不感知底下是单进程还是多进程。
class Executor(ABC):
    """
        EngineCore 调用 Worker 的统一接口。

        EngineCore 只依赖这组方法，不感知底层是单进程直接调用还是多进程广播。
    """

    # 记录设备与 dtype，并算出采样 rank；KV 管理器留到 init_kv 再建。
    def __init__(
        self, config: EngineConfig, device: torch.device, dtype: torch.dtype
    ) -> None:
        self.config = config
        self.device = device
        self.dtype = dtype
        # 采样 rank 由 tp × pp 决定，构造时就固定下来。
        self.sampler_rank = _sampler_rank(config.tp_size, config.pp_size)
        # 门面侧的 KV 管理器：只做记账，不持显存。
        self.kv: KVManager | None = None

    # 用 Worker 报上来的块数建一个记账用的 KVManager。
    def _kv_manager(self, num_blocks: int) -> KVManager:
        """根据 Worker 报告的可用块数，构造门面侧的 KVManager。"""
        # 注意这是**两份** KV 对象：Worker 里那份才是显存池，这里只是块分配账本。
        return KVManager(
            num_blocks=num_blocks,
            block_size=self.config.block_size,
            prefix_backend=self.config.prefix_backend,
        )

    # 六个抽象方法就是 Executor 的全部契约，其它都是可选的旁路。
    @abstractmethod
    def init_kv(self) -> KVManager: ...

    @abstractmethod
    def reset_kv(self, prefix_backend: str | None = None) -> KVManager: ...

    @abstractmethod
    def run(self, sched: ScheduledBatch) -> tuple[int, int, int]: ...

    @abstractmethod
    def read_samples(self, n_p: int, n_d: int, slot: int) -> tuple[list[int], list[int]]: ...

    @abstractmethod
    def destroy(self) -> None: ...

    # 广播控制命令；基类默认拒绝未知 op —— 单进程实现根本用不到它。
    def collective_rpc(self, op: str, **payload) -> object:
        """向所有 Worker 广播控制命令。基类默认拒绝未知操作。"""
        raise ValueError(f"unknown control op: {op}")

    # 通知所有 Worker 退出；单进程实现下没什么可做的。
    def shutdown(self) -> None:
        """通知所有 Worker 退出。单进程实现无需操作。"""
        return None


# ------------------------------------------------------------------ #
# 子进程入口与命令循环
# ------------------------------------------------------------------ #


# 子进程入口（rank ≥ 1）：建环境与模型 → 同步 → 进命令循环。
def _spawn_worker(
    config: EngineConfig, rank: int, tp_rank: int, pp_rank: int
) -> None:
    """
        子进程入口（rank ≥ 1）。

        初始化 CUDA 环境与模型权重，完成分布式握手后进入命令循环，
        等待 driver 广播指令。进程退出时清理 CUDA 上下文。
    """
    # 子进程只做自己的事，它不需要知道自己是第几个被拉起来的。
    worker = Worker(config, rank, tp_rank, pp_rank)
    try:
        worker.init_environment()
        worker.initialize()
        # 所有 rank 都就绪后再进循环：避免加载慢的 rank 错过第一条命令。
        dist.barrier()
        worker_loop(worker)
    finally:
        worker.destroy_environment()


# 非 driver rank 的主循环：收命令 → 执行 →（若是采样 rank）回传结果。
def worker_loop(worker: Worker) -> None:
    """
        非 driver rank（rank ≥ 1）的主循环。

        不断等待 driver 通过 Gloo 广播的 ControlCommand：
        "execute"：执行前向，若本 rank 是采样 rank，
                    则读取采样结果并送回 driver（rank0）。
        "reset_kv"：重建 KV cache。
        "shutdown"：退出循环，进程结束。

        注意：采样结果通过 Gloo送回。
        这意味着 driver 侧的 recv 是阻塞的，无法与下一次 schedule 重叠。
    """
    # 控制通道建在 WORLD 的 Gloo 组上：这是一条 CPU 字节通道。
    channel = ControlChannel(get_world_group().cpu_group)
    # 编解码器预先建好，别每步现建。
    enc = msgspec.msgpack.Encoder()
    dec = msgspec.msgpack.Decoder(ControlCommand)
    sampler_rank = _sampler_rank(worker.tp_size, worker.pp_size)

    while True:
        # 阻塞等待 driver 广播的下一条命令。
        # 阻塞等广播。broadcast(None) 在非 src 侧的意思就是「我来收」。
        cmd = dec.decode(channel.broadcast(None, src=0))

        # shutdown：跳出循环，清理交给 finally。
        if cmd.op == "shutdown":
            break

        # reset_kv：重建本 rank 的池，然后继续等下一条命令。
        if cmd.op == "reset_kv":
            worker.reset_kv(cmd.prefix_backend)
            continue

        if cmd.op != "execute":
            raise ValueError(f"unknown control op: {cmd.op}")

        # 执行前向。所有 rank 都参与（TP 需要集合通信，PP 需要逐段执行）。
        # 执行前向。**所有 rank 都要参与** —— TP 要集合通信，PP 要逐段接力。
        counts = worker.execute_model(cmd.sched)

        # 只有采样 rank 需要读结果并回传。
        # 只有采样 rank 需要读结果并回传，其它 rank 直接进入下一轮等待。
        if worker.rank != sampler_rank:
            continue

        # 分两条消息回传：先回计数（driver 靠它决定接下来怎么收），再回 token id。
        n_p, n_d, slot = counts
        channel.send(enc.encode([n_p, n_d, slot]), dst=0)
        p_ids, d_ids = worker.read_samples(n_p, n_d, slot)
        channel.send(enc.encode([p_ids, d_ids]), dst=0)


# ------------------------------------------------------------------ #
# 单进程执行器（tp=1, pp=1）
# ------------------------------------------------------------------ #


# 单卡路径最短：Worker 与 EngineCore 同进程，所有调用直接转发。
class UniprocExecutor(Executor):
    """
        单卡执行器：Worker 与 EngineCore 在同一进程内所有调用直接转发。
    """

    # 两种构造方式：worker=None（run.py 自己 spawn）／注入已建好的 rank0 Worker（torchrun）。
    def __init__(
        self, config: EngineConfig, device: torch.device, dtype: torch.dtype
    ) -> None:
        super().__init__(config, device, dtype)
        # 进程内建一个 rank0 的 Worker。
        self.worker = Worker(config, rank=0, tp_rank=0, pp_rank=0)
        # 通信环境要立刻建 —— 即使是单卡，也需要 Group 对象存在。
        self.worker.init_environment()
        # initialize() 延迟到首次 init_kv 时调用，
        # 避免在 Engine.__init__ 阶段过早分配模型显存。
        # 模型初始化推迟到首次 init_kv，避免 Engine 构造阶段就吃掉几十 GB 显存。
        self._ready = False

    # 首次调用时初始化模型；之后每次都只是重建记账对象。
    def init_kv(self) -> KVManager:
        """首次调用时初始化模型权重，之后每次调用构造 KVManager。"""
        if not self._ready:
            self.worker.initialize()
            self._ready = True
        # 块数由 Worker 报上来 —— 它才是真正开池的那一方。
        self.kv = self._kv_manager(self.worker.get_kv_cache_size())
        return self.kv

    # 换缓存后端 / 重开池：先转给 Worker，再重建账本。
    def reset_kv(self, prefix_backend: str | None = None) -> KVManager:
        if prefix_backend is not None:
            self.config.prefix_backend = prefix_backend
        self.worker.reset_kv(prefix_backend)
        self.kv = self._kv_manager(self.worker.get_kv_cache_size())
        return self.kv

    # 单进程直接调用：既没有序列化，也没有广播。
    def run(self, sched: ScheduledBatch) -> tuple[int, int, int]:
        """直接调用 Worker 执行前向，返回 (n_prefill, n_decode)。"""
        # 仍然走 to_wire：让单卡与多卡两条路径面对同一种数据形状，避免两套代码。
        counts = self.worker.execute_model(sched.to_wire())
        # 纯 TP 或 pp=1：采样就在 rank0，直接返回。
        assert counts is not None
        return counts

    def read_samples(
        self, n_p: int, n_d: int, slot: int
    ) -> tuple[list[int], list[int]]:
        return self.worker.read_samples(n_p, n_d, slot)

    # 单进程下「广播」就是本地执行。
    def collective_rpc(self, op: str, **payload) -> object:
        """单进程下无需广播，直接本地执行。"""
        # 只支持 reset_kv，其它 op 交回基类报错。
        if op == "reset_kv":
            self.worker.reset_kv(payload.get("prefix_backend"))
            return None
        return super().collective_rpc(op, **payload)

    # 释放：先断账本引用，再拆 Worker 环境。
    def destroy(self) -> None:
        self.kv = None
        self.worker.destroy_environment()


# ------------------------------------------------------------------ #
# 多进程执行器（tp × pp > 1）
# ------------------------------------------------------------------ #


# 多卡路径：rank0 = driver + 本地 Worker，其余 rank 在子进程里等命令。
class MultiprocExecutor(Executor):
    """
    多卡执行器：WORLD = tp × pp。

    rank0 同时承担两个角色：
      - driver：接收 EngineCore 的调用，通过 Gloo 广播调度指令。
      - 本地 Worker：执行自己那份模型前向。

    构造方式有两种：
      - worker=None（run.py 路径）：本进程负责 spawn 所有子进程。
      - worker=已初始化的Worker（torchrun 路径）：
        各进程已在外部完成初始化，这里只包装 rank0 的 Worker。
    """

    def __init__(
        self,
        config: EngineConfig,
        device: torch.device,
        dtype: torch.dtype,
        worker: Worker | None = None,
    ) -> None:
        super().__init__(config, device, dtype)

        # msgspec 编解码器，用于 Gloo 控制消息的序列化。
        self._enc = msgspec.msgpack.Encoder()
        self._dec_counts = msgspec.msgpack.Decoder(list[int])
        self._dec_sample = msgspec.msgpack.Decoder(list[list[int]])

        # 子进程句柄：destroy 时要用。
        self._procs: list[mp.Process] = []
        # 记录子进程是否由本实例创建。
        # torchrun 路径下子进程由外部管理，destroy 时不应 terminate。
        # 用一个 owns 标志区分 run.py 与 torchrun 两条路径：后者不能动外部管理的子进程。
        self._owns_workers = worker is None

        # 自己 spawn 一整队 Worker（含 rank0）。
        if worker is None:
            worker = self._spawn_world()

        # rank0 兼做本地 Worker。
        self.worker = worker
        # 块数由 Worker 报告，门面只负责记账。
        self.num_blocks = worker.get_kv_cache_size()
        # rank0 用这条通道广播命令、接收采样结果。
        self._channel = ControlChannel(get_world_group().cpu_group)

    # 启动顺序有讲究：先起 rank ≥ 1 的子进程，再初始化 rank0。
    def _spawn_world(self) -> Worker:
        """
            拉起所有子进程并初始化 rank0。

            顺序：先启动 rank ≥ 1 的子进程，再初始化本进程的 rank0 Worker。
            这样做是因为 torch.distributed 的 FileStore 会合机制
            不要求 rank0 先监听，所有 rank 可以并行完成握手。

            若初始化过程中任何一步失败，立即清理已启动的子进程。
        """
        driver: Worker | None = None
        try:
            # 用 spawn 而不是 fork：CUDA 上下文不能跨 fork 复制。
            ctx = mp.get_context("spawn")
            tp, pp = self.config.tp_size, self.config.pp_size

            # 先启动 rank ≥ 1 的子进程。
            # 按 pp 外层、tp 内层的顺序编号，与 initialize_model_parallel 的算术规则一致。
            for pp_rank in range(pp):
                for tp_rank in range(tp):
                    rank = pp_rank * tp + tp_rank
                    if rank == 0:
                        continue
                    # 每个子进程有自己的进程组与 CUDA 上下文。
                    p = ctx.Process(
                        target=_spawn_worker,
                        args=(self.config, rank, tp_rank, pp_rank),
                        daemon=False,
                    )
                    p.start()
                    self._procs.append(p)

            # 再初始化本进程的 rank0。
            # 再初始化 rank0：它也要建环境、加载自己那份权重。
            driver = Worker(self.config, rank=0, tp_rank=0, pp_rank=0)
            driver.init_environment()
            driver.initialize()

            # 所有 rank 就绪后同步，确保模型权重和通信器都已初始化。
            # 全部就绪后同步一次，然后才开始收命令。
            dist.barrier()
            return driver

        # 任何一步失败都要把已启动的子进程收掉，否则会留下占着显存的孤儿进程。
        except BaseException:
            # 初始化失败：清理已启动的子进程，避免残留。
            _terminate_procs(self._procs)
            if driver is not None:
                try:
                    driver.destroy_environment()
                except Exception:
                    pass
            raise

    # 多进程下模型在 __init__ 阶段已经加载完，这里只建记账对象。
    def init_kv(self) -> KVManager:
        """
            构造门面侧的 KVManager。
            多进程下模型已在 __init__ 阶段初始化完毕，
            这里只需根据 Worker 报告的块数构造管理对象。
        """
        self.kv = self._kv_manager(self.num_blocks)
        return self.kv

    # 重建：**先广播给子进程，再本地执行** —— 否则会各建各的、块数对不上。
    def reset_kv(self, prefix_backend: str | None = None) -> KVManager:
        """
            重建所有 rank 的 KV cache。
            先广播 reset_kv 命令让子进程执行，再本地执行，最后重建管理对象。
        """
        if prefix_backend is not None:
            self.config.prefix_backend = prefix_backend
        self.collective_rpc("reset_kv", prefix_backend=prefix_backend)
        self.worker.reset_kv(prefix_backend)
        self.num_blocks = self.worker.get_kv_cache_size()
        self.kv = self._kv_manager(self.num_blocks)
        return self.kv

    # 广播控制命令。注意 rank0 自己的执行在 run() 里单独做，不走这里。
    def collective_rpc(self, op: str, **payload) -> object:
        """
            通过 Gloo 向所有 Worker 广播控制命令。
            所有 rank（包括 rank0 自身）都会在 worker_loop 或
            execute_model 中收到并执行该命令。
        """
        cmd = ControlCommand(
            op=op,
            sched=payload.get("sched"),
            prefix_backend=payload.get("prefix_backend"),
        )
        # 阻塞式广播：所有 rank 都必须走到这一行。
        self._channel.broadcast(self._enc.encode(cmd), src=0)
        return None

    # 一步前向的四件事：序列化 → 广播 → 本地执行 →（PP 场景）等回传。
    def run(self, sched: ScheduledBatch) -> tuple[int, int, int]:
        """
            执行一步前向。

            1. 将 ScheduledBatch 序列化为 SchedulerOutput（to_wire）。
            2. 通过 Gloo 广播给所有子进程。
            3. rank0 本地也执行前向（它同时是 Worker）。
            4. 若采样不在 rank0（PP 场景），等待采样 rank 回传计数。
        """
        wire = sched.to_wire()

        # 广播调度数据，子进程在 worker_loop 中收到并执行。
        # 先广播、再本地执行 —— 顺序反了会让子进程等不到命令。
        self.collective_rpc("execute", sched=wire)

        # rank0 本地执行。
        # rank0 自己也算一份工作量。
        counts = self.worker.execute_model(wire)

        # 纯 TP 或 pp=1：采样就在 rank0，直接返回。
        if self.sampler_rank == 0:
            # 纯 TP 或 pp=1：采样就在 rank0，直接返回。
            assert counts is not None
            return counts

        # PP 场景：采样在末 stage，等待采样 rank 回传 (n_p, n_d)。
        # PP 场景：采样在末段，等它把 (n_p, n_d, slot) 送回来。
        n_p, n_d, slot = self._dec_counts.decode(
            self._channel.recv(self.sampler_rank)
        )
        return n_p, n_d, slot

    def read_samples(
        self, n_p: int, n_d: int, slot: int
    ) -> tuple[list[int], list[int]]:
        """
            读取采样结果。

            采样在 rank0 时直接本地读取；
            否则等待采样 rank 通过 Gloo 送回 token id 列表。
        """
        if self.sampler_rank == 0:
            return self.worker.read_samples(n_p, n_d, slot)
        p_ids, d_ids = self._dec_sample.decode(self._channel.recv(self.sampler_rank))
        return p_ids, d_ids

    # 广播 shutdown，让子进程跳出命令循环。
    def shutdown(self) -> None:
        """广播 shutdown 命令，通知所有子进程退出命令循环。"""
        self.collective_rpc("shutdown")

    # 释放顺序：广播退出 → 收子进程 → 拆本地 Worker。
    def destroy(self) -> None:
        """
            释放执行器资源。

            若子进程由本实例创建（run.py 路径），先广播 shutdown，
            再终止子进程，最后清理本地 Worker。
            torchrun 路径下子进程由外部管理，这里只断开引用。
        """
        # 只有自己 spawn 的那批才需要 shutdown + terminate。
        if self._owns_workers:
            try:
                self.shutdown()
            except Exception:
                pass
            _terminate_procs(self._procs)
            self.worker.destroy_environment()
        self.kv = None


# ------------------------------------------------------------------ #
# 工厂函数
# ------------------------------------------------------------------ #


# 工厂：按 tp × pp 是否都退化到 1 来选择实现。
def create_executor(
    config: EngineConfig,
    device: torch.device,
    dtype: torch.dtype,
    *,
    rank0_worker: Worker | None = None,
) -> Executor:
    """
        根据 tp / pp 配置选择执行器实现。

        - tp=1, pp=1：返回 UniprocExecutor，同进程直接调用。
        - 否则：返回 MultiprocExecutor。

        rank0_worker 仅在 torchrun 多机路径下使用：
        外部已在当前进程中初始化好 rank0 的 Worker，
        直接注入，避免 MultiprocExecutor 重复创建。
    """
    # 单卡单进程走最短路径。
    if config.tp_size == 1 and config.pp_size == 1:
        return UniprocExecutor(config, device, dtype)
    # 其余情况一律走多进程。
    return MultiprocExecutor(config, device, dtype, worker=rank0_worker)
