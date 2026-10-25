#!/usr/bin/env python3
"""从 nsys 导出的 sqlite 里，算「每两步之间 GPU 空转了多久」。

思路：NVTX 里的 zos.launch 区间 = CPU 的「一步」窗口。
      在相邻两个窗口之间，GPU 最后一个 kernel 结束 → 下一个 kernel 开始的
      这段时间，就是 GPU 在等 CPU（气泡）。ZOS 应该把它压到 0。

用法：
    python bench/gap_step.py /tmp/g0.sqlite
    python bench/gap_step.py /tmp/g1.sqlite
"""
import sqlite3
import sys
import statistics

SKIP = 3                      # 跳过 warmup 步（前几步含编译）
MAX_STEP_NS = 3_000_000_000   # 超过 3 s 的窗口算异常（GC/编译），跳过


def load(path):
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        "SELECT start, end FROM NVTX_EVENTS "
        "WHERE text = 'zos.launch' AND end IS NOT NULL ORDER BY start"
    )
    wins = [(s, e) for s, e in cur.fetchall()]
    cur.execute(
        "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
    )
    ks = cur.fetchall()
    con.close()
    return wins, ks


def main(path):
    wins, ks = load(path)
    wins = [(s, e) for s, e in wins if e - s < MAX_STEP_NS][SKIP:]
    if len(wins) < 3:
        print("窗口太少（%d）" % len(wins))
        return

    ks.sort()
    starts = [k[0] for k in ks]
    import bisect

    def kernels_in(t0, t1):
        i = bisect.bisect_left(starts, t0)
        out = []
        while i < len(ks) and ks[i][0] < t1:
            out.append(ks[i])
            i += 1
        return out

    gaps, busy, span = [], [], []
    for i in range(len(wins) - 1):
        t0, t1 = wins[i]
        a = kernels_in(t0, t1)
        b = kernels_in(t1, wins[i + 1][1])
        if not a or not b:
            continue
        last_end = max(e for _, e in a)
        next_start = min(s for s, _ in b)
        if next_start < last_end:          # 上下步重叠（GPU 落后于 CPU），不算空转
            gaps.append(0.0)
        else:
            gaps.append((next_start - last_end) / 1e6)
        busy.append(sum(e - s for s, e in a) / 1e6)
        span.append((t1 - t0) / 1e6)

    if not gaps:
        print("没有可用的步")
        return

    def stat(xs):
        return "med %.3f  mean %.3f  p90 %.3f  max %.3f" % (
            statistics.median(xs), statistics.mean(xs),
            sorted(xs)[int(len(xs) * 0.9)], max(xs))

    print("文件      : %s" % path)
    print("步数      : %d（跳过前 %d 步 warmup）" % (len(gaps), SKIP))
    print("步墙钟 ms : " + stat(span))
    print("GPU 忙 ms : " + stat(busy))
    print("步间空转  : " + stat(gaps))
    print("空转占比  : %.2f%%（中位数 / 中位步时）"
          % (100.0 * statistics.median(gaps) / statistics.median(span)))
    zero = sum(1 for g in gaps if g < 0.05)
    print("空转<0.05ms 的步: %d/%d (%.0f%%)" % (zero, len(gaps), 100.0 * zero / len(gaps)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/g0.sqlite")

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

