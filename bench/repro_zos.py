#!/usr/bin/env python3
"""一键复现 ZOS 的三层证据。

跑 N 轮，每轮 KRITACE_ZOS=0/1 各一次（交错 A B A B），输出：

  ① 正确性    两个模式的逐 token sha 是否一致
  ② 机制      zos-stats 的 ready%（read_samples 有多少次"不用等"）与累计等待
  ③ 气泡      gpu-stats 的「步墙钟 − GPU 忙时」= 每步 GPU 空转多少 ms

用法::

    PYTHONPATH=. python bench/repro_zos.py                  # 默认 3 轮，丢掉第 1 轮
    PYTHONPATH=. python bench/repro_zos.py --rounds 4 --n 8 --max-tokens 32
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))


import argparse
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CASE_RE = re.compile(r"mode=(\d+) steps=(\d+) .*elapsed=([\d.]+)s sha=([0-9a-f]+)")
ZOS_RE = re.compile(r"calls=(\d+) ready=(\d+) \(([\d.]+)%\) total_wait=([\d.]+)ms")
GPU_RE = re.compile(r"steps=(\d+) gpu_total=([\d.]+)ms avg_gpu=([\d.]+)ms")


def run_once(mode: int, n: int, max_tokens: int, blocks: int) -> dict:
    env = dict(os.environ)
    env.update(PYTHONPATH=ROOT, KRITACE_ZOS=str(mode), KRITACE_ZOS_STATS="1", KRITACE_ZOS_GPU="1")
    cmd = [
        sys.executable,
        os.path.join(HERE, "zos_matrix.py"),
        "--case", "short",
        "--n", str(n),
        "--max-tokens", str(max_tokens),
        "--blocks", str(blocks),
    ]
    out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True).stdout
    rec = {"mode": mode, "sha": "?", "steps": 0, "elapsed": 0.0,
           "ready_pct": None, "wait_ms": None, "avg_gpu_ms": None}
    if m := CASE_RE.search(out):
        rec.update(steps=int(m.group(2)), elapsed=float(m.group(3)), sha=m.group(4))
    if m := ZOS_RE.search(out):
        rec.update(ready_pct=float(m.group(3)), wait_ms=float(m.group(4)))
    if m := GPU_RE.search(out):
        rec.update(avg_gpu_ms=float(m.group(3)))
    if rec["steps"]:
        rec["step_ms"] = rec["elapsed"] * 1000.0 / rec["steps"]
        if rec["avg_gpu_ms"] is not None:
            rec["bubble_ms"] = rec["step_ms"] - rec["avg_gpu_ms"]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=48)
    ap.add_argument("--drop-first", type=int, default=1,
                    help="丢掉前几轮：GPU 时钟爬坡时单轮能差 3 倍")
    args = ap.parse_args()

    rows = []
    for r in range(1, args.rounds + 1):
        for mode in (0, 1):                      # 交错：A B A B
            rec = run_once(mode, args.n, args.max_tokens, args.blocks)
            rec["round"] = r
            rows.append(rec)
            print(f"  round {r}  KRITACE_ZOS={mode}  step={rec.get('step_ms', 0):.1f} ms  "
                  f"gpu={rec.get('avg_gpu_ms') or 0:.1f} ms  "
                  f"ready={rec.get('ready_pct')}%  sha={rec['sha']}", flush=True)

    used = [r for r in rows if r["round"] > args.drop_first] or rows
    print(f"\n  （统计用 {len(used)} 条，丢掉前 {args.drop_first} 轮 —— GPU 时钟爬坡）\n")

    def med(mode: int, key: str):
        vals = [r[key] for r in used if r["mode"] == mode and r.get(key) is not None]
        return statistics.median(vals) if vals else None

    print("| 指标 | KRITACE_ZOS=0（同步） | KRITACE_ZOS=1（零开销） |")
    print("|---|---|---|")
    print(f"| 步墙钟（含 GPU 降频，别直接比） | {med(0,'step_ms'):.1f} ms | {med(1,'step_ms'):.1f} ms |")
    print(f"| GPU 忙时 | {med(0,'avg_gpu_ms'):.1f} ms | {med(1,'avg_gpu_ms'):.1f} ms |")
    print(f"| **每步气泡 = 墙钟 − GPU 忙** | **{med(0,'bubble_ms'):+.2f} ms** | **{med(1,'bubble_ms'):+.2f} ms** |")
    print(f"| read 不用等的比例 | {med(0,'ready_pct'):.1f}% | {med(1,'ready_pct'):.1f}% |")
    print(f"| 累计等待 | {med(0,'wait_ms'):.1f} ms | {med(1,'wait_ms'):.1f} ms |")

    shas = {r["sha"] for r in used}
    print(f"\n**正确性**：{len(shas)} 种 sha → {'逐 token 一致 (OK)' if len(shas) == 1 else '不一致 (FAIL)'}")

    b0, b1 = med(0, "bubble_ms"), med(1, "bubble_ms")
    if b0 is not None and b1 is not None:
        s0 = med(0, "step_ms")
        ratio = 100.0 * (b0 - b1) / s0 if s0 else 0.0
        print(f"**机制**：每步气泡 {b0:+.2f} ms -> {b1:+.2f} ms，收回 {b0 - b1:+.2f} ms"
              f"（占步时 {ratio:.2f}%）")
        print("**判读**：气泡/步时 = 吞吐上限 1 + B/G 里的 B/G。"
              "占比 <1% 时端到端必然测不出差别 —— 不是 ZOS 失效，是 B 太小。")

        s0, s1 = med(0, "step_ms"), med(1, "step_ms")
        g0, g1 = med(0, "avg_gpu_ms"), med(1, "avg_gpu_ms")
        if s0 and g0 and abs(s1 - s0) / s0 > 0.02:
            ds = 100.0 * (s1 - s0) / s0
            dg = 100.0 * (g1 - g0) / g0
            if abs(ds - dg) < 2.0:
                print(f"**警告**：步墙钟差 {ds:+.1f}%，但 GPU 忙时也差 {dg:+.1f}% —— "
                      "同向同量，是 GPU 时钟漂移，不能算 ZOS 的功劳。")


if __name__ == "__main__":
    main()
