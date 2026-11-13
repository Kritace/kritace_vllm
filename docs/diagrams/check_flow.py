# -*- coding: utf-8 -*-
r"""图源机械校验 + 渲染指标检查（改图后先跑这个，别靠肉眼）。

用法：
    python check_flow.py [图源.mmd]         # 默认 full_flow.mmd
    python check_flow.py notes.mmd          # 也可以校验说明区
    python check_flow.py --metrics [图源]   # 另外打印那一轮渲染的指标
    python check_flow.py --fix-bold [图源]  # 把 mermaid 不生效的 **粗体** 换成【】

校验项分三类：
  结构 —— 子图配对、引号 / $$ 成对、节点 id 唯一、边引用的 id 有定义
  规则 —— 不出现代码标识符；不出现 **粗体**（mermaid 不生效，会原样吐出）
  公式 —— 不出现 \textbf（它只能在文本模式用，公式行里会报未定义命令）；
          不出现空分母的 \frac{...}{}；\begin{gathered} 与 \end{gathered} 成对

关于「代码标识符」的白名单口径：`allreduce` 是集合通信的标准名字，图里本来就要写，
不算代码标识符 —— 会被拦的是 `all_reduce`（下划线那份，真代码里的名字）。
"""
from __future__ import annotations

import io
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))

# 允许命令行覆盖图源；`--metrics` / `--fix-bold` 之后的非选项参数才是文件名。
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
MMD = os.path.join(HERE, _args[0] if _args else "full_flow.mmd")
METRICS = os.path.join(HERE, "_metrics.json")


def load(path: str) -> str:
    return io.open(path, encoding="utf-8").read()


# 本图特有的硬规则：不出现代码标识符。
# 注意 allreduce 是集合通信的标准名（图里故意要写），不在拦截范围；all_reduce 才是代码名。
CODE_PAT = re.compile(
    r"\b(flash_attn\w*|store_kvcache|_pad_bs|can_append_n|may_append_n|deallocate|"
    r"greedy_accept_length|IntermediateTensors|block_table\w*|slot_mapping|cache_seqlens|"
    r"cu_seqlens\w*|all_reduce\w*|tensor_model_parallel\w*|num_cached_tokens|spec_draft\w*|"
    r"\w+\.\w+\(|def |class |import )"
)


def validate(path: str) -> int:
    text = load(path)
    lines = text.split("\n")
    bad: list[str] = []
    depth = 0
    defined: dict[str, list[int]] = {}
    gathered = 0

    for i, l in enumerate(lines, 1):
        s = l.strip()
        if s.startswith("subgraph"):
            depth += 1
        elif s == "end":
            depth -= 1
            if depth < 0:
                bad.append(f"L{i}: 多余的 end")
        for m in re.finditer(r'^\s*([A-Za-z][A-Za-z0-9_]*)\s*(?:\(\[|\[\(|\[\(\(|\[|\{)', l):
            defined.setdefault(m.group(1), []).append(i)
        if l.count('"') % 2:
            bad.append(f"L{i}: 引号数为奇数")
        if l.count("$$") % 2:
            bad.append(f"L{i}: $$ 数为奇数 -> {s[:100]}")
        if l.replace("$$", "").count("$") % 2:
            bad.append(f"L{i}: 单 $ 数为奇数 -> {s[:100]}")
        if "**" in l:
            bad.append(f"L{i}: 出现 **粗体**（mermaid 不生效，会原样吐出）-> {s[:100]}")
        # 公式类硬规则
        if "\\textbf" in l:
            bad.append(f"L{i}: 出现 \\textbf —— 它只能在文本模式用，公式行里会报未定义命令 -> {s[:100]}")
        if re.search(r"(?<!\\)\\(?![\\\\\w{}\[\]()|.,;:!+\-*/=<>&'\"^_~#$% \n])([^\x00-\x7F])", l):
            bad.append(f"L{i}: 单个反斜杠后面跟了非 ASCII 字符 —— KaTeX 会把它当命令名 -> {s[:100]}")
        if re.search(r"\\frac\{[^}]*\}\{\}", l):
            bad.append(f"L{i}: 空分母的 \\frac{{...}}{{}} -> {s[:100]}")
        gathered += l.count("\\begin{gathered}") - l.count("\\end{gathered}")

    print(f"图源         : {os.path.basename(path)}")
    print("subgraph 配对 :", "OK" if depth == 0 else f"**不平衡 depth={depth}**")
    print("gathered 配对 :", "OK" if gathered == 0 else f"**不平衡 {gathered:+d}**")
    print("行内问题数   :", len(bad))
    for b in bad:
        print("   ", b)

    dupes = {k: v for k, v in defined.items() if len(v) > 1}
    print("节点定义数   :", len(defined), "| 重复 id:", len(dupes))
    for k, v in list(dupes.items())[:10]:
        print("    重复:", k, v)

    # 边引用的 id 是否都定义过（子图 id 允许被引用，这里放宽）
    sub_ids = set(re.findall(r'^\s*subgraph\s+([A-Za-z0-9_]+)', text, re.M))
    refs: set[str] = set()
    for l in lines:
        for m in re.finditer(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)\s*(?:-->|-\.->|==>|~~~)', l):
            refs.add(m.group(1))
        for m in re.finditer(r'(?:-->|-\.->|==>|~~~)\s*(?:\|[^|]*\|\s*)?([A-Za-z][A-Za-z0-9_]*)', l):
            refs.add(m.group(1))
    missing = sorted(refs - set(defined) - sub_ids)
    print("未定义的边端点:", missing if missing else "无")

    hits = [(i, l.strip()[:90]) for i, l in enumerate(lines, 1) if CODE_PAT.search(l)]
    print("疑似代码标识符:", len(hits))
    for i, l in hits[:8]:
        print(f"    L{i}: {l}")

    print("总行数       :", len(lines))
    bad_n = bool(bad) or depth != 0 or gathered != 0 or missing
    return 1 if bad_n else 0


def show_metrics() -> None:
    if not os.path.exists(METRICS):
        print("\n[_metrics.json 不存在：先跑 run_render.ps1]")
        return
    m = json.loads(load(METRICS))
    print("\n================ 渲染指标 ================")
    print("画布        :", m["viewBox"][2], "x", m["viewBox"][3])
    print("节点/子图   : %s / %s" % (m["nodes"], m["clusters"]))
    print("边          :", m["edgePaths"])
    print("KaTeX 公式  :", m["katexNodes"])
    print("字体        : %s/%s" % (m["fontsLoaded"], m["fontsTotal"]))
    print("--- 硬判据（必须全 0）---")
    print("clipped     :", m["clipped"])
    print("shapeClips  :", m["shapeClips"])
    print("nodeOverlap :", m["nodeOverlapArea"])
    print("clusterOvlp :", m["clusterOverlapArea"])
    print("katexErrors :", m.get("katexErrors", "n/a"), "<- 公式写错的直接证据")
    for s in (m.get("katexErrorSample") or [])[:3]:
        print("     样例:", s)
    print("--- 参考指标 ---")
    print("crossings   :", m["crossings"])
    print("longestEdge :", m["longestEdge"], " <- 最重要的布局指标，越小越顺")


def fix_bold(path: str) -> int:
    """把 mermaid 不生效的 **粗体** 换成【】，返回替换处数。"""
    text = load(path)
    new, n = re.subn(r"\*\*(?![*\s])([^*\n]+?)\*\*", r"【\1】", text)
    if n:
        io.open(path, "w", encoding="utf-8", newline="\n").write(new)
    print(f"**粗体** → 【】：替换 {n} 处；剩余 {new.count('**')}")
    return n


if __name__ == "__main__":
    if "--fix-bold" in sys.argv:
        fix_bold(MMD)
    rc = validate(MMD)
    if "--metrics" in sys.argv:
        show_metrics()
    raise SystemExit(rc)
