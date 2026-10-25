# ============================================================================
# 【本文件是做什么的】一个**全局分段计时器**：把一步推理按模块切开分别计时，
# 用来看"时间到底花在哪一段"（驱动循环 / RPC / 调度 / 前向 / 通信 / PP 传输）。
#
# 【为什么要按"模块名.细分名"打点】问题常常出在某一段内：
# 只知道 "runner 占 60%" 不够，还要能下钻到 "runner.inputs" 与 "runner.cg"。
# 于是约定名字里用 `.` 分层，rollup() 按第一段汇总、summary() 给出全部细分。
#
# 【为什么默认关闭】计时本身要花时间（perf_counter 调用、字典累加），
# 而且 `CudaTimer` 会做 `torch.cuda.synchronize()` —— 那会**改变被测程序的并行行为**：
# 本来 CPU 可以跑在前面、GPU 排队，一同步就退化成逐步串行，测出来的数字完全不同。
# 所以：默认关，靠环境变量 KRITACE_SYS_PROF 打开，且只用来定位瓶颈，不用来报性能数。
#
# ⚠️ 用 CudaTimer 测出来的段时**加起来不等于总时间**：每次同步都会砍掉重叠部分。
#   要报吞吐就用引擎自己的统计，不要拿 profiler 的数去加。
# ============================================================================

from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

# 模块名 → 一句话说明，只服务于 rollup 输出（让人看得懂哪一段是什么）。
# 这些 key 必须与打点处使用的一级名字严格一致，否则那一栏的说明就是空的。
MODULE_TREE: dict[str, str] = {
    "engine": "Driver loop (schedule → execute → postprocess)",
    "rpc": "ZMQ/pipe RPC + pickle of Sequence batch",
    "sched": "Scheduler + BlockManager + Radix prefix cache",
    "runner": "ModelRunner: inputs / attn meta / forward / sample / CG",
    "worker": "Worker brackets around forward + PP wait",
    "nccl": "TP all-reduce",
    "pp": "PP P2P tensor + metadata",
    "layer": "Optional fine-grained layer hooks",
}

# 开关在**模块导入时**读一次环境变量。
# 这意味着：运行中改环境变量不生效（要配合 set_enabled 或重启进程）。
_ENABLED = os.environ.get("KRITACE_SYS_PROF", "").strip() in ("1", "true", "True")


def enabled() -> bool:
    return _ENABLED


# 运行中动态开关，主要给测试用。改的是模块级全局变量，全进程生效。
def set_enabled(flag: bool) -> None:
    """Test helper / dynamic toggle (normally set via env before import)."""
    global _ENABLED
    _ENABLED = bool(flag)


# 每个统计桶：累计毫秒 + 调用次数。用 count 是为了能算「平均每次多少毫秒」。
@dataclass
class _Bucket:
    total_ms: float = 0.0
    count: int = 0

    def add(self, ms: float) -> None:
        self.total_ms += ms
        self.count += 1


@dataclass
# 全局唯一的实例（见文件末尾的 _PROF）。
# 做成"全局单例 + 模块级函数"而不是到处传对象：打点处往往在最底层，
# 把 profiler 一层层传下去会污染所有函数签名。
class SysProfiler:
    buckets: dict[str, _Bucket] = field(default_factory=lambda: defaultdict(_Bucket))

    def add(self, name: str, ms: float) -> None:
        self.buckets[name].add(ms)

    def reset(self) -> None:
        self.buckets.clear()

    # 返回**全部**细分条目的完整统计（名字 → count/total/mean）。
    # 按名字排序输出，方便两次运行之间做 diff。
    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, b in sorted(self.buckets.items()):
            out[k] = {
                "count": b.count,
                "total_ms": round(b.total_ms, 3),
                "mean_ms": round(b.total_ms / max(b.count, 1), 4),
            }
        return out

    # 按名字的第一段（`.` 之前）汇总 —— 同一模块下的细分加到一起，
    # 得到"时间都花在哪个模块"这张表。
    def rollup(self) -> list[dict[str, Any]]:
        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        # 只取一级名字：runner.inputs 与 runner.forward 都归到 runner。
        for name, b in self.buckets.items():
            top = name.split(".", 1)[0]
            totals[top] += b.total_ms
            counts[top] += b.count
        # 总时间用来算占比。or 1e-9 是防除零（什么都没打点时 grand=0）。
        grand = sum(totals.values()) or 1e-9
        rows = []
        # 按耗时降序：一眼看到最大的那块，不用在表里找。
        for top, ms in sorted(totals.items(), key=lambda x: -x[1]):
            rows.append(
                {
                    "module": top,
                    "desc": MODULE_TREE.get(top, ""),
                    "total_ms": round(ms, 3),
                    "count": counts[top],
                    "pct": round(100.0 * ms / grand, 2),
                }
            )
        return rows


# 模块级单例：所有打点都进这一份，所以进程内同名打点会跨请求累计。
_PROF = SysProfiler()


def get_profiler() -> SysProfiler:
    return _PROF


def reset() -> None:
    _PROF.reset()


def summary() -> dict[str, Any]:
    return _PROF.summary()


def rollup() -> list[dict[str, Any]]:
    return _PROF.rollup()


# 【CPU 侧计时】context manager 用法：`with Timer("sched"): ...`。
# 关闭时 __enter__ 什么都不做 —— 这是"零开销埋点"的关键：
# 埋点代码可以常驻，不打开就退化成两次空函数调用。
class Timer:
    """Wall-clock timer; no-op when profiling is disabled."""

    __slots__ = ("name", "_t0")

    def __init__(self, name: str):
        self.name = name
        self._t0 = 0.0

    # 只在启用时读时钟：关闭时连 perf_counter 都不调（见文件头：埋点不应污染被测对象）。
    # 用 perf_counter 而不是 time.time：后者会被系统时钟调整影响，单调时钟才适合测间隔。
    def __enter__(self):
        if _ENABLED:
            self._t0 = time.perf_counter()
        return self

    # 退出时把本段耗时累加进桶里；返回 False 表示不吞异常（异常照常向外抛）。
    def __exit__(self, *exc):
        if _ENABLED:
            _PROF.add(self.name, (time.perf_counter() - self._t0) * 1000.0)
        return False


# 【GPU 侧计时】用 CUDA event 而不是 CPU 时钟：
# kernel 是**异步**提交的，CPU 上的 start/end 之间只代表"提交完"，不代表算完。
# event 打在流里，测的是真实的 GPU 执行时间。
class CudaTimer:
    """CUDA-event timer (device sync on exit). Prefer ``Timer`` for CPU paths."""

    __slots__ = ("name", "_start", "_end")

    def __init__(self, name: str):
        self.name = name
        self._start = None
        self._end = None

    # 三层保护，缺一不可：没开 profiler、没装/没 GPU、torch 未导入。
    # 任何一条不满足就退化成空操作，保证埋点不会把主流程搞挂。
    def __enter__(self):
        if not _ENABLED:
            return self
        import torch

        if not torch.cuda.is_available():
            return self
        # 两个 event 必须成对创建：elapsed_time 要用它们算差值（毫秒）。
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._start.record()
        return self

    # 没记录过 start 就直接返回：说明 __enter__ 走了空操作分支。
    def __exit__(self, *exc):
        if not _ENABLED or self._start is None or self._end is None:
            return False
        import torch

        # 在流里打个结束标记，然后**同步等待** —— 不等待就读 elapsed_time 会读到 0。
        # 注意代价：这个 sync 会把 CPU/GPU 的重叠抹掉（见文件头 ⚠️）。
        self._end.record()
        torch.cuda.synchronize()
        # 单位已经是毫秒（torch 的 elapsed_time 返回 ms），与 Timer 的口径一致。
        _PROF.add(self.name, self._start.elapsed_time(self._end))
        return False
