"""测每步的 CPU 各段耗时（依赖 flash-attn：v6 原生路径硬依赖它；
没有 flash-attn 的机器会自动走 src/compute/layers/_flash_attn_fallback.py）（B 的量级），以及 ZOS 下是否还有可填的气泡。

用法::

    KRITACE_ZOS=0 PYTHONPATH=. python bench/cpu_bubble.py --n 8 --max-tokens 32
    KRITACE_ZOS=1 PYTHONPATH=. python bench/cpu_bubble.py --n 8 --max-tokens 32

输出（每行一个阶段，单位 ms，取中位数）::

    schedule   CPU 调度
    run        CPU 组包 + H2D + 发射 kernel
    read       读采样结果（阻塞等待的部分也计入）
    finish     回灌账本 + 推进序列 + 产出 EngineCoreOutput
    step       整步墙上时间

ZOS=0 时 GPU 每步的空泡 ≈ finish + schedule + run（因为 read 返回时 GPU 刚好空）。
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))


import argparse
import os
import statistics
import time

SHORT = [
    "The capital of France is",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    return",
    "1, 1, 2, 3, 5, 8, 13,",
    "Q: What is the boiling point of water at sea level?\nA:",
]

PHASES: dict[str, list[float]] = {}


def wrap(obj, name: str, label: str):
    # v6 原生只有 update()；做过 ZOS 的分支才有 finish()。两边都能跑。
    if not hasattr(obj, name):
        return
    fn = getattr(obj, name)

    def inner(*a, **kw):
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            PHASES.setdefault(label, []).append((time.perf_counter() - t0) * 1e3)

    setattr(obj, name, inner)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=48)
    ap.add_argument("--context-len", type=int, default=512)
    args = ap.parse_args()

    from src import Engine, EngineConfig, SamplingParams

    cfg = EngineConfig(
        model=args.model,
        enforce_eager=True,
        max_num_seqs=max(args.n, 4),
        context_len=args.context_len,
        num_kvcache_blocks=args.blocks,
        gpu_memory_utilization=0.75,
        prefix_backend="none",
        max_num_batched_tokens=512,
    )
    eng = Engine(cfg)
    try:
        core = eng._client.engine_core
        wrap(core.scheduler, "schedule", "schedule")
        wrap(core.executor, "run", "run")
        wrap(core.executor, "read_samples", "read")
        wrap(core.scheduler, "finish", "finish")
        wrap(core.scheduler, "update", "finish")   # v6 原生路径
        from src.data.batch import ScheduledBatch
        from src.compute.batch import Batch
        wrap(ScheduledBatch, "to_wire", "to_wire")
        wrap(Batch, "build_decode", "build_dec")
        wrap(Batch, "build_prefill", "build_pre")

        seqs = [
            eng.add_request(SHORT[i % len(SHORT)], SamplingParams(max_tokens=args.max_tokens))
            for i in range(args.n)
        ]
        while not all(s.is_finished for s in seqs):
            eng.step()
    finally:
        eng.destroy()

    print(f"mode={os.environ.get(chr(86)+chr(53)+chr(95)+chr(90)+chr(79)+chr(83), chr(45))} n={args.n} max_tokens={args.max_tokens}")
    for k in ("schedule", "to_wire", "build_dec", "build_pre", "run", "read", "finish"):
        v = PHASES.get(k) or [0.0]
        print(f"  {k:<9} med={statistics.median(v):7.3f}  mean={statistics.mean(v):7.3f}  max={max(v):7.3f}  n={len(v)}")
    m = min(len(PHASES.get(k, [])) for k in ("finish", "schedule", "run")) or 0
    b = [PHASES["finish"][i] + PHASES["schedule"][i] + PHASES["run"][i] for i in range(m)] or [0.0]
    print(f"  B=fin+sch+run  med={statistics.median(b):7.3f}  (ZOS=0 时这一步就是纯空泡)")


if __name__ == "__main__":
    main()
