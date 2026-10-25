# ============================================================================
# 【本文件是做什么的】控制面的对外接口。这一层被拆成了五块，
# 从这里能一眼看全整条链：
#
#     Engine（门面）
#       → EngineCoreClient（DP 的分叉点）
#           → EngineCore（核心循环：调度 → 执行 → 采样 → 后处理）
#               → Executor（TP / PP 的分叉点）
#                   → Worker（一个进程 / 一张卡）
#
# 【最重要的是记住两个分叉点分别在哪儿】
#     DP 的分叉在 client 层：dp=1 就地转发（InprocClient），dp>1 走 ZMQ 副本（MPClient）；
#     TP/PP 的分叉在 executor 层：单卡进程内直调（Uniproc），多卡 rank0 广播（Multiproc）。
# 两个维度的并行是两层独立的分叉 —— 这是整个架构里最关键的一点。
#
# 【读的顺序】engine（门面做什么）→ core（循环怎么转）→ executor / mp_client（两种展开方式）。
# ============================================================================

"""Control plane: scheduling + engine orchestration (vLLM V1 shape)."""

from __future__ import annotations

from .engine import Engine
from .scheduler import Scheduler
from .core import EngineCore
from .client import EngineCoreClient, InprocClient, make_client
from .mp_client import MPClient, run_engine_core_proc
from .executor import (
    ControlCommand,
    Executor,
    MultiprocExecutor,
    UniprocExecutor,
    create_executor,
    worker_loop,
)

# 对外契约。
__all__ = [
    "Engine",
    "Scheduler",
    "EngineCore",
    "EngineCoreClient",
    "InprocClient",
    "MPClient",
    "make_client",
    "run_engine_core_proc",
    "Executor",
    "ControlCommand",
    "UniprocExecutor",
    "MultiprocExecutor",
    "create_executor",
    "worker_loop",
]
