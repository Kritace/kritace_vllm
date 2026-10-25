# ============================================================================
# 【本文件是做什么的】数据面的对外接口。这一层只有三类东西：
#     ① 跨层传递的结构：ScheduledBatch / SchedulerOutput / SeqSchedule / EngineCoreOutput
#     ② KV 账本：KVManager（只是块分配记录，真正的显存在 compute/kv_pool.py）
#     ③ 请求本体：Sequence 与它的状态机
#
# 【为什么这些都在 data/ 下】它们全部是CPU 侧的数据结构：调度器、门面、
# 控制通道、跨进程消息都在用它们。GPU 侧的常驻缓冲刻意放在 compute/batch.py，
# 两边分开 —— 这样"数据"和"显存"不会互相拖住。
# ============================================================================

from __future__ import annotations

from .batch import EngineCoreOutput, ScheduledBatch, SchedulerOutput, SeqSchedule
from .kv import KVManager
from .sequence import Sequence, SequenceStatus

# 对外契约。
__all__ = [
    "EngineCoreOutput",
    "ScheduledBatch",
    "SchedulerOutput",
    "SeqSchedule",
    "KVManager",
    "Sequence",
    "SequenceStatus",
]
