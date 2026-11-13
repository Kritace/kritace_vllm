#!/usr/bin/env python3
"""把 nsys 导出的 sqlite 画成 ASCII 甘特图：看清"CPU 在干什么"与"GPU 有没有活"的时间关系。

用法::

    nsys export --type sqlite --force-overwrite true -o x.sqlite x.nsys-rep
    python bench/3050test/timeline.py x.sqlite 4        # 只看前 4 步

思路（对应 nsys 里的三张表）::

    NVTX_EVENTS                 → CPU 行：zos.schedule / zos.launch / zos.read / zos.finish
    CUPTI_ACTIVITY_KIND_KERNEL  → GPU 行：每个 kernel 的 [start, end)
    一个"step 窗口" = 相邻两个 zos.launch 的起点之间

每列的宽度 = 窗口宽度 / 96。字符含义：

    GPU 行   # = 有 kernel 在跑      （空白 = GPU 空闲）
    CPU 行   = = 该 NVTX 区间覆盖的时间

⚠️ 只在 KRITACE_NVTX=1 且 nsys 带了 `-t nvtx,cuda` 时才有数据。
⚠️ nsys 采集本身会把墙钟拉长（每个 kernel 都有采样开销），
   所以**用来看构成，不要拿这里的毫秒数比吞吐**。
"""
from __future__ import annotations

import sqlite3
import sys

W = 96  # 每行字符数


def load(db: str):
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute(
        "SELECT start, end, text FROM NVTX_EVENTS WHERE end IS NOT NULL ORDER BY start"
    )
    nv = cur.fetchall()
    # CUPTI_ACTIVITY_KIND_KERNEL 里没有 name 列（kernel 名在 StringIds 表里），
    # 本脚本只关心起止时间，所以只取 [start, end)。
    cur.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")
    ks = cur.fetchall()
    con.close()
    return nv, ks


def to_cols(t0: int, span: int, s: int, e: int) -> tuple[int, int]:
    """时间区间 → [起始列, 结束列]（都夹在 0..W-1）。"""
    a = int((s - t0) / span * W)
    b = int((e - t0) / span * W)
    return max(0, min(W - 1, a)), max(0, min(W - 1, b))


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return
    db = sys.argv[1]
    nshow = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    nv, ks = load(db)
    launches = [r for r in nv if r[2] == "zos.launch"]
    if len(launches) < 2:
        print("NVTX 里找不到 zos.launch：确认采集时设了 KRITACE_NVTX=1 且 -t 里含 nvtx")
        return

    for i in range(min(nshow, len(launches) - 1)):
        t0 = launches[i][0]
        t1 = launches[i + 1][0]
        span = t1 - t0
        in_win = [k for k in ks if t0 <= k[0] < t1]
        ktot = sum(k[1] - k[0] for k in in_win)

        print(
            f"\nstep {i + 1}  窗口 {span / 1e6:7.1f} ms | "
            f"kernel {ktot / 1e6:5.1f} ms ({100 * ktot / span:5.1f}%) | "
            f"GPU 空转 {(span - ktot) / 1e6:7.1f} ms"
        )

        # ---- GPU 行：所有 kernel 并成一条 ----
        gpu = [" "] * W
        for s, e in in_win:
            a, b = to_cols(t0, span, s, e)
            for c in range(a, b + 1):
                gpu[c] = "#"
        print("  GPU    |" + "".join(gpu) + "|")

        # ---- CPU 行：四类 NVTX 区间各一行 ----
        for name in ("zos.schedule", "zos.launch", "zos.read", "zos.finish"):
            segs = [r for r in nv if r[2] == name and t0 <= r[0] < t1]
            if not segs:
                continue
            tot = sum(e - s for s, e, _ in segs) / 1e6
            row = [" "] * W
            for s, e, _ in segs:
                a, b = to_cols(t0, span, s, e)
                for c in range(a, b + 1):
                    row[c] = "="
            print(f"  {name:<13}|" + "".join(row) + f"| {tot:6.2f} ms")

    # ---- 全局汇总：GPU 到底忙了多少 ----
    if ks:
        k0 = min(k[0] for k in ks)
        k1 = max(k[1] for k in ks)
        tot = sum(k[1] - k[0] for k in ks)
        print(
            f"\n全局：kernel {len(ks)} 次 / {tot / 1e6:.1f} ms，"
            f"首末 kernel 之间 {((k1 - k0) / 1e6):.1f} ms，"
            f"GPU 占空比 {100 * tot / (k1 - k0):.1f}%"
        )


if __name__ == "__main__":
    main()
