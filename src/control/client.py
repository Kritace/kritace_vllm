# ============================================================================
# 【本文件是做什么的】Engine 与 EngineCore 之间的**调用层**。
# 它存在的唯一目的：让「DP 的分叉」只发生在一个地方。
#
# 【为什么要多这一层】如果 Engine 直接持有 EngineCore，那么 dp>1 时
# Engine 就必须知道"我的请求要发给哪个副本" —— 那是副本管理的知识，
# 不该渗进引擎门面。抽出一个接口之后：
#     dp = 1 → InprocClient：同进程直接转发，零开销；
#     dp > 1 → MPClient  ：经 ZMQ 把请求路由到各副本子进程。
# 门面看到的永远是同一组方法。
#
# 【两维并行的分工，在这里能看得很清楚】
#     DP 的分叉  → 在本文件这一层（换一个 Client 实现）；
#     TP/PP 的分叉 → 在 EngineCore 内部的 Executor 那一层。
# 两个维度的并行在代码里是**两层独立的分叉**，不是混在一起判断的。
#
# 【读这个文件的顺序】
#     EngineCoreClient（四个方法就是全部契约）
#       → InprocClient（最短路径）
#       → make_client（按 dp_size 选实现）
# ============================================================================

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from ..config import EngineConfig
from ..data import EngineCoreOutput, Sequence
from .core import EngineCore

# ==============================================================================
# EngineCoreClient：Engine 与 EngineCore 之间的调用层。
#
# Engine 不直接持有 EngineCore，而是通过这一层间接调用。
# 这样做的目的是让 DP 的分叉只发生在这里：
#   - dp=1：InprocClient，同进程直接转发，零开销。
#   - dp>1：MPClient，经 ZMQ 将请求路由到各副本子进程。
#
# TP / PP 的多进程展开不在这层处理，由 EngineCore 内部的 Executor 负责。
# ==============================================================================


# 四个方法就是全部门面：加请求、走一步、还有没有活、销毁。
class EngineCoreClient(ABC):
    """
        Engine 调用 EngineCore 的统一接口。

        无论底层是同进程转发还是 ZMQ 跨进程通信，
        Engine 侧的 add_request / step / has_unfinished / destroy
        都只依赖这四个方法，不感知具体的执行路径。
    """

    # 抽象声明 —— 具体是同进程转发还是跨进程通信，由子类决定。
    @abstractmethod
    def add_request(self, seq: Sequence) -> None: ...

    @abstractmethod
    def step(self) -> tuple[list[EngineCoreOutput], int]: ...

    @abstractmethod
    def has_unfinished(self) -> bool: ...

    @abstractmethod
    def destroy(self) -> None: ...


# dp=1：直接持有 EngineCore，所有调用原地转发。
class InprocClient(EngineCoreClient):
    """
        单进程客户端（dp=1）。

        直接持有 EngineCore 实例，所有方法直接转发，与 Scheduler 共享同一对象。
    """

    # 同进程共享同一个 EngineCore 对象，没有序列化也没有通道。
    def __init__(self, engine_core: EngineCore) -> None:
        self.engine_core = engine_core

    # 下面四个方法都是纯转发，不做任何加工。
    def add_request(self, seq: Sequence) -> None:
        self.engine_core.add_request(seq)

    def step(self) -> tuple[list[EngineCoreOutput], int]:
        return self.engine_core.step()

    def has_unfinished(self) -> bool:
        return self.engine_core.has_unfinished()

    def destroy(self) -> None:
        self.engine_core.destroy()


# 工厂：按 dp_size 选择客户端实现。
def make_client(
    config: EngineConfig,
    *,
    eos_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
    executor=None,
) -> EngineCoreClient:
    """
        工厂函数：根据 dp_size 选择客户端实现。

        dp=1：构造 EngineCore，包装为 InprocClient 返回。
        此时可通过 executor 参数注入已初始化好的 MultiprocExecutor

        dp>1：构造 MPClient，启动多个副本子进程。
        每个副本拥有独立的 NCCL world 和设备，不允许共享外部注入的 executor。
    """
    # 只有 dp>1 时才 import MPClient —— 避免单卡场景也把 zmq 依赖拖进来。
    if config.dp_size > 1:
        from .mp_client import MPClient

        # dp>1 时禁止注入共享 executor：每个副本必须有自己的 world 与设备。
        if executor is not None:
            raise ValueError(
                "dp_size>1 uses MPClient replicas; cannot inject a shared executor"
            )
        return MPClient(
            config, eos_token_id=eos_token_id, device=device, dtype=dtype
        )
    # dp=1：就地建一个 EngineCore（executor 可由外部注入）。
    core = EngineCore(
        config, eos_token_id=eos_token_id, device=device, dtype=dtype, executor=executor
    )
    # 包一层直接调用的客户端返回。
    return InprocClient(core)