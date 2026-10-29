# -*- coding: utf-8 -*-
"""扫描仓库内的 import，产出模块依赖图（分层 + 边表 + 环报告）。

区分两类边：
  eager —— 模块顶层（import 时就会执行）的 import，决定真实的分层；
  lazy  —— 写在函数体里的 import（延迟到调用时），只用于打破环、规避重依赖。

用法（仓库根目录）：
    python3 tools/dep_graph.py            # 打印报告
    python3 tools/dep_graph.py --mmd      # 额外输出 mermaid 边表
"""
from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = ["src", "utils", "tests", "bench"]


def module_name(p: Path) -> str:
    parts = list(p.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def resolve(module: str, node: ast.ImportFrom, known: set[str]) -> list[str]:
    """把一个 ImportFrom 解析成本仓库模块名。

    取"最深且存在"的那个：`from ..data import batch` → `src.data.batch`（而不是 `src.data`）；
    `from .layers import attention, linear` → 两个子模块各一条。
    """
    if node.level:
        pkg = module.split(".")
        base = pkg[: len(pkg) - node.level]
        target = ".".join(base + ([node.module] if node.module else []))
    else:
        target = node.module or ""
    if not target:
        return out_none()

    out: list[str] = []
    for alias in node.names:
        sub = f"{target}.{alias.name}"
        if sub in known:
            out.append(sub)
    if target in known and not out:
        out.append(target)
    if out:
        return out
    parts = target.split(".")
    for i in range(len(parts), 0, -1):
        c = ".".join(parts[:i])
        if c in known:
            return [c]
    return out_none()


def out_none() -> list[str]:
    return []


def scan() -> dict[str, dict]:
    files: dict[str, Path] = {}
    for d in SCAN_DIRS:
        for p in sorted((ROOT / d).rglob("*.py")):
            files[module_name(p)] = p
    known = set(files)

    info: dict[str, dict] = {}
    for mod, path in files.items():
        src = path.read_text(encoding="utf-8", errors="replace")
        eager: set[str] = set()
        lazy: set[str] = set()

        def visit(node: ast.AST, depth: int) -> None:
            for child in ast.iter_child_nodes(node):
                inner = depth + 1 if isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
                ) else depth
                if isinstance(child, ast.Import):
                    for a in child.names:
                        parts = a.name.split(".")
                        for i in range(len(parts), 0, -1):
                            c = ".".join(parts[:i])
                            if c in known:
                                (lazy if inner else eager).add(c)
                                break
                elif isinstance(child, ast.ImportFrom):
                    tgt = resolve(mod, child, known)
                    (lazy if inner else eager).update(tgt)
                visit(child, inner)

        visit(ast.parse(src), 0)
        info[mod] = {
            "path": path,
            "lines": src.count("\n") + 1,
            "eager": eager,
            "lazy": lazy - eager,
        }
    return info


def _graph(info: dict[str, dict], key: str = "eager") -> dict[str, list[str]]:
    known = set(info)
    return {m: sorted(info[m][key] & known) for m in info}


def sccs(graph: dict[str, list[str]]) -> list[list[str]]:
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on: dict[str, bool] = {}
    stack: list[str] = []
    out: list[list[str]] = []
    counter = [0]

    def strong(v: str) -> None:
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on[v] = True
        for w in graph.get(v, []):
            if w not in index:
                strong(w)
                low[v] = min(low[v], low[w])
            elif on.get(w):
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on[w] = False
                comp.append(w)
                if w == v:
                    break
            out.append(comp)

    for v in graph:
        if v not in index:
            strong(v)
    return out


def layers(info: dict[str, dict]):
    graph = _graph(info, "eager")
    comps = sccs(graph)
    comp_of: dict[str, int] = {}
    for i, c in enumerate(comps):
        for m in c:
            comp_of[m] = i
    dag: dict[int, set[int]] = defaultdict(set)
    for m in info:
        for d in graph[m]:
            if comp_of[m] != comp_of[d]:
                dag[comp_of[m]].add(comp_of[d])
    lvl: dict[int, int] = {}

    def depth(i: int, seen: frozenset = frozenset()) -> int:
        if i in lvl:
            return lvl[i]
        if i in seen:
            return 0
        v = 0 if not dag[i] else 1 + max(depth(j, seen | {i}) for j in dag[i])
        lvl[i] = v
        return v

    for i in sorted(range(len(comps)), key=lambda i: len(dag[i])):
        depth(i)
    return {m: lvl[comp_of[m]] for m in info}, comps


def main() -> None:
    info = scan()
    mod_lvl, comps = layers(info)
    graph = _graph(info, "eager")
    rev: dict[str, set[str]] = defaultdict(set)
    for m in info:
        for d in graph[m]:
            rev[d].add(m)

    prod = [m for m in info if m == "src" or m.startswith(("src.", "utils"))]
    print(f"# 全仓 {len(info)} 个模块 / {sum(v['lines'] for v in info.values())} 行；"
          f"产品代码 {len(prod)} 个 / {sum(info[m]['lines'] for m in prod)} 行\n")

    by_lvl: dict[int, list[str]] = defaultdict(list)
    for m in prod:
        by_lvl[mod_lvl[m]].append(m)
    print("## 按层分组（只列产品代码；层号越小越底层）")
    for l in sorted(by_lvl):
        print(f"\nL{l}:")
        for m in sorted(by_lvl[l]):
            e = len(info[m]["eager"] & set(info))
            lz = len(info[m]["lazy"] & set(info))
            print(f"  {m:40s} {info[m]['lines']:5d} 行  eager={e:2d} lazy={lz:2d} used_by={len(rev[m]):2d}")

    print("\n## 模块级环（eager 边）")
    cyc = [c for c in comps if len(c) > 1]
    print("  无" if not cyc else "")
    for c in cyc:
        print("  " + " <-> ".join(sorted(c)))

    print("\n## 靠 lazy import 打破的环（把 lazy 边也算上才会成环）")
    both = {m: sorted(info[m]["eager"] | info[m]["lazy"]) for m in info}
    cyc2 = [c for c in sccs(both) if len(c) > 1]
    print("  无" if not cyc2 else "")
    for c in cyc2:
        print("  " + " <-> ".join(sorted(c)))

    print("\n## 入度 Top10（被最多文件依赖，不含 tests/bench）")
    for m, users in sorted(rev.items(), key=lambda kv: -len(kv[1]))[:10]:
        if m.startswith(("tests", "bench")):
            continue
        print(f"  {m:40s} <- {len(users):2d}")

    print("\n## 出度 Top8")
    for m in sorted(prod, key=lambda m: -len(info[m]["eager"] & set(info)))[:8]:
        print(f"  {m:40s} -> {sorted(info[m]['eager'] & set(info))}")

    if "--mmd" in sys.argv:
        print("\n## mermaid")
        print("flowchart BT")
        for m in sorted(prod, key=lambda m: (mod_lvl[m], m)):
            print(f'  {m.replace(".", "_")}["{m}"]')
        for m in sorted(prod):
            for d in graph[m]:
                if d.startswith(("src", "utils")):
                    print(f"  {m.replace('.', '_')} --> {d.replace('.', '_')}")


if __name__ == "__main__":
    main()
