# -*- coding: utf-8 -*-
"""检查 mermaid 渲染出的 SVG：每个子图框是否真的把自己的成员框住了。

为什么需要这个工具
------------------
mermaid 的 SVG 里 `clusters` 与 `nodes` 是**兄弟**关系（不是父子），
所以"谁属于哪个子图"在 SVG 里查不到 —— 只能拿**图源的声明**来对照：
按 .mmd 里的 subgraph 块算出每个子图的成员表，再看这些节点在 SVG 里的
坐标是否落在该子图的框内。

典型缺陷（本项目真实踩过）
------------------------
mermaid 的归属按「**第一次提到**」判定。若边 `C0h --> X1` 写在
`subgraph C0` 里、而 `subgraph TPA` 的声明在更后面，则 X1 被判给**根图**：
渲染出来 X1 的位置是对的（被上游边拉到了该在的深度），
但 TPA 的框只围住了在 TPA 里首次出现的 X0 —— 看起来就是「X1 没放进框里」。

用法
----
    python probe_clusters.py <source.mmd> <rendered.svg>
"""
from __future__ import annotations

import io
import re
import sys
from xml.etree import ElementTree as ET

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
NS = "{http://www.w3.org/2000/svg}"


# --------------------------------------------------------------------------- #
# 图源：解析 subgraph 块 -> {子图id: [节点id, ...]}
#
# 必须复刻 mermaid 真正的归属规则：**谁第一次提到就归谁**。
# 所以不能只看 `ID[...]` 声明 —— `subgraph C0` 里的边 `C0h --> X1`
# 也是一次"提到"。若 X1 的首次提到发生在 C0 里，它就归 C0；
# 只有把 `subgraph TPA` 挪到 C0 之前，X1 的首次提到才落在 TPA。
# --------------------------------------------------------------------------- #
_KEYWORDS = {
    "subgraph", "end", "flowchart", "graph", "direction", "TB", "TD", "BT", "LR", "RL",
    "classDef", "class", "linkStyle", "style", "click",
}


def parse_source(path: str) -> tuple[dict[str, list[str]], dict[str, str | None]]:
    lines = io.open(path, encoding="utf-8").read().split("\n")
    owner: dict[str, str | None] = {}   # 节点 -> 首次提到时所在的子图（None = 根图）
    stack: list[str] = []
    declared: list[str] = []            # 按首次提及顺序

    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("%%"):
            continue
        m = re.match(r"subgraph\s+([A-Za-z0-9_]+)", s)
        if m:
            stack.append(m.group(1))
            continue
        if s == "end":
            if stack:
                stack.pop()
            continue
        if s.startswith(("flowchart", "graph ", "classDef", "class ", "linkStyle", "style ", "click")):
            continue
        # 去掉引号内容（节点正文 / 边标签）与 :::class 后缀，剩下的才是 id
        body = re.sub(r'"[^"]*"', '""', s)
        body = re.sub(r":::[A-Za-z0-9_]+", "", body)
        here = stack[-1] if stack else None
        for nid in re.findall(r"[A-Za-z][A-Za-z0-9_]*", body):
            if nid in _KEYWORDS or nid in owner:
                continue
            owner[nid] = here
            declared.append(nid)

    members: dict[str, list[str]] = {}
    for nid in declared:
        o = owner[nid]
        if o is not None:
            members.setdefault(o, []).append(nid)
    return members, owner


# --------------------------------------------------------------------------- #
# SVG：子图框 + 节点坐标
# --------------------------------------------------------------------------- #
def _norm(gid: str) -> str:
    """把 mermaid 生成的 id 还原成图源里的名字。

    cluster : mmd_render_0-C3            -> C3
    node    : mmd_render_0-flowchart-X0-1 -> X0-1   （调用方再去掉 -<序号>）
    注意 cluster 的 id 里**没有** `flowchart-` 这一段，两种前缀都要剥。
    """
    s = re.sub(r"^mmd_render_\d+-", "", gid)
    return re.sub(r"^flowchart-", "", s)


class Fig:
    def __init__(self, svg: str) -> None:
        self.root = ET.fromstring(io.open(svg, encoding="utf-8").read())
        self.clusters: dict[str, tuple[float, float, float, float, str]] = {}
        self.nodes: dict[str, tuple[float, float]] = {}
        self._scan()

    def _scan(self) -> None:
        for g in self.root.iter(NS + "g"):
            cls = (g.get("class") or "").split()
            short = _norm(g.get("id", ""))
            if "cluster" in cls:
                for r in g.iter(NS + "rect"):
                    self.clusters[short] = (
                        float(r.get("x", 0)),
                        float(r.get("y", 0)),
                        float(r.get("width", 0)),
                        float(r.get("height", 0)),
                        "".join(g.itertext()).strip(),
                    )
                    break
            elif "node" in cls:
                m = re.search(r"translate\(\s*([-\d.]+)\s*[, ]\s*([-\d.]+)", g.get("transform", ""))
                if m:
                    # mermaid 把节点 id 变成 "<id>-<序号>"
                    self.nodes[re.sub(r"-\d+$", "", short)] = (float(m.group(1)), float(m.group(2)))


# --------------------------------------------------------------------------- #
def main() -> int:
    mmd = sys.argv[1] if len(sys.argv) > 1 else "full_flow.mmd"
    svg = sys.argv[2] if len(sys.argv) > 2 else "_rendered.svg"
    members, owner = parse_source(mmd)
    fig = Fig(svg)

    bad = 0
    known = set(fig.clusters)
    for sg in sorted(members, key=lambda k: (k not in known, k)):
        if sg not in known:
            print(f"!! 子图 {sg} 在图源里但没有渲染成 cluster（成员被判给了别的层级）")
            bad += len(members[sg])
            continue
        x, y, w, h, lab = fig.clusters[sg]
        got = [n for n in members[sg] if n in fig.nodes]
        out = [n for n in got if not (x <= fig.nodes[n][0] <= x + w and y <= fig.nodes[n][1] <= y + h)]
        flag = "OK " if not out else "BAD"
        print(f"[{flag}] {sg:5s} 框=x[{x:.0f},{x+w:.0f}] y[{y:.0f},{y+h:.0f}]  成员 {len(got)}/{len(members[sg])}  越框 {len(out)}   {lab[:38]}")
        for n in out:
            nx, ny = fig.nodes[n]
            print(f"        ❌ {n:6s} ({nx:.0f},{ny:.0f}) 落在框外")
        bad += len(out)

    # 根图里不应出现的节点（被误判给根图的通信节点）
    rootish = sorted(n for n, o in owner.items() if o is None and n in fig.nodes)
    if rootish:
        print(f"\n位于根图（不属于任何子图）的节点：{rootish}")

    print(f"\n越框成员合计 = {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
