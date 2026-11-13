#!/usr/bin/env python3
"""把 full_flow.mmd + _rendered.svg 组装成独立的离线查看器。

产出（默认都在本目录）：
  * 推理全流程图.mmd   —— 图源（可单独喂给别的 mermaid 渲染器）
  * 推理全流程图.html  —— 查看器：把 SVG 内联进页面，自带滚轮缩放 / 拖拽平移 /
                          适应宽度 / 适应高度 / 1:1，零外部请求
  * 推理全流程图.svg   —— 静态图（公式样式与字体引用已内联）

用法：
    python build_flow_viewer.py            # 重新生成
    python build_flow_viewer.py --check    # 只校验 svg 与 mmd 是否都比 html 新

注意：svg / html 都按相对路径引用 fonts/，所以这三个东西必须同目录。
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANUAL_NAME = "项目全景手册.html"      # 与 HERE 同级的 docs/ 下的手册文件名

# 每个产物：图源 / 渲染出的 svg / 产物前缀 / 查看器标题
# 前缀决定渲染中间产物的名字（render_page.html 里 ?out=<prefix>），
# 空串表示沿用旧命名的 _rendered.svg 等。
BUILDS = [
    {
        "src": "full_flow.mmd",
        "svg": "_rendered.svg",
        "out_mmd": "推理全流程图.mmd",
        "out_svg": "推理全流程图.svg",
        "out_html": "推理全流程图.html",
        "title": "dzyy-vllm v0–v6 · 一次完整推理全流程图",
    },
    {
        "src": "notes.mmd",
        "svg": "notes_rendered.svg",
        "out_mmd": "说明区.mmd",
        "out_svg": "说明区.svg",
        "out_html": "说明区.html",
        "title": "dzyy-vllm v0–v6 · 说明区（主图之外的查用块）",
    },
    {
        "src": "cg_flow.mmd",
        "svg": "cg_rendered.svg",
        "out_mmd": "CUDA-Graph-过程图.mmd",
        "out_svg": "CUDA-Graph-过程图.svg",
        "out_html": "CUDA-Graph-过程图.html",
        "title": "dzyy-vllm v0–v6 · CUDA Graph 过程图（捕获 → 重放，含多卡边界）",
        "embed_svg": True,
    },
    {
        "src": "compile_flow.mmd",
        "svg": "cc_rendered.svg",
        "out_mmd": "torch-compile-过程图.mmd",
        "out_svg": "torch-compile-过程图.svg",
        "out_html": "torch-compile-过程图.html",
        "title": "dzyy-vllm v0–v6 · torch.compile 过程图（追踪 → 断图 → 融合）",
        "embed_svg": True,
    },
    {
        "src": "boundary_flow.mmd",
        "svg": "bd_rendered.svg",
        "out_mmd": "边界与观测-过程图.mmd",
        "out_svg": "边界与观测-过程图.svg",
        "out_html": "边界与观测-过程图.html",
        "title": "dzyy-vllm v0–v6 · 图与编译的边界（叠加 / 四路并行 / 观测）",
        "embed_svg": True,
    },
    {
        "src": "merged_flow.mmd",
        "svg": "mg_rendered.svg",
        "out_mmd": "推理全流程图-合并版.mmd",
        "out_svg": "推理全流程图-合并版.svg",
        "out_html": "推理全流程图-合并版.html",
        "title": "dzyy-vllm v6 · 一次完整推理：单卡流程 × 四卡切分 × 投机一轮（合并版）",
        "embed_svg": True,
    },
    {
        "src": "merged_cards_flow.mmd",
        "svg": "mgc_rendered.svg",
        "out_mmd": "推理全流程图-逐卡版.mmd",
        "out_svg": "推理全流程图-逐卡版.svg",
        "out_html": "推理全流程图-逐卡版.html",
        "title": "dzyy-vllm v6 · 一次完整推理：四张卡逐卡展开（配合合并版看）",
        "embed_svg": True,
    },
    {
        "src": "merged_super_flow.mmd",
        "svg": "mgs_rendered.svg",
        "out_mmd": "推理全流程图-完整版.mmd",
        "out_svg": "推理全流程图-完整版.svg",
        "out_html": "推理全流程图-完整版.html",
        "title": "dzyy-vllm v6 · 一次完整推理：骨架 × 四卡逐卡 × 投机一轮 × 账本与并行（完整版）",
        "embed_svg": True,
        "manual_link": True,
    },
]

HTML_TEMPLATE = """<!doctype html>
<!-- 由 build_flow_viewer.py 生成，请不要手改；
     改图请改 full_flow.mmd 后重跑无头渲染 + 本脚本。 -->
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  html, body {{ margin: 0; height: 100%; background: #f6f6f4;
                font: 13px/1.6 system-ui, "Segoe UI", "Microsoft YaHei", sans-serif;
                color-scheme: light; }}
  #bar {{ position: fixed; inset: 0 0 auto 0; height: 40px; display: flex; align-items: center;
          gap: 8px; padding: 0 12px; background: #fffffff2; border-bottom: 1px solid #ddd8cf;
          backdrop-filter: blur(4px); z-index: 10; }}
  #bar b {{ font-weight: 600; margin-right: 4px; }}
  #bar button {{ font: inherit; padding: 3px 10px; border: 1px solid #cfc9be; border-radius: 6px;
                 background: #fff; cursor: pointer; }}
  #bar button:hover {{ background: #f2efe9; }}
  #pct {{ min-width: 56px; color: #6b6357; }}
  #hint {{ margin-left: auto; color: #8a8272; }}
  #stage {{ position: absolute; inset: 40px 0 0 0; overflow: hidden; cursor: grab; background: #f6f6f4; }}
  #stage.dragging {{ cursor: grabbing; }}
  #wrap {{ position: absolute; top: 0; left: 0; transform-origin: 0 0; }}
  #wrap > svg {{ display: block; background: #fff; box-shadow: 0 1px 10px #0000000f; }}
  #empty {{ position: absolute; inset: 80px 24px auto 24px; padding: 16px; border: 1px solid #e0b4b4;
            background: #fff6f6; border-radius: 8px; color: #84343a; white-space: pre-wrap; display: {empty_display}; }}
<!--EXTRA_CSS-->
</style>
</head>
<body>
<div id="bar">
  <b>{title}</b>
  <button id="fitw">适应宽度</button>
  <button id="fith">适应高度</button>
  <button id="one">1:1</button>
  <button id="zi">＋</button>
  <button id="zo">－</button>
  <span id="pct">—</span>
  <span id="hint">滚轮缩放 · 拖拽平移 · 双击复位 · 键盘 +/−/0</span>
</div>
<div id="stage"><div id="wrap">{svg}</div></div>
<div id="empty">{empty_msg}</div>
<!--EXTRA_BODY-->
<script>
const stage = document.getElementById('stage');
const wrap  = document.getElementById('wrap');
const pctEl = document.getElementById('pct');
const svg   = wrap.querySelector('svg');
let scale = 1, tx = 0, ty = 0;
const W = svg ? (+svg.getAttribute('width')  || svg.viewBox.baseVal.width  || 1000) : 1000;
const H = svg ? (+svg.getAttribute('height') || svg.viewBox.baseVal.height || 1000) : 1000;
function apply() {{
  wrap.style.transform = `translate(${{tx}}px, ${{ty}}px) scale(${{scale}})`;
  pctEl.textContent = Math.round(scale * 100) + '%';
}}
function fit(mode) {{
  const r = stage.getBoundingClientRect();
  const k = mode === 'w' ? (r.width - 24) / W : (r.height - 24) / H;
  scale = Math.max(0.01, k);
  tx = (r.width  - W * scale) / 2;
  ty = (r.height - H * scale) / 2;
  apply();
}}
function zoomAt(cx, cy, factor) {{
  const ns = Math.min(20, Math.max(0.02, scale * factor));
  tx = cx - (cx - tx) * (ns / scale);
  ty = cy - (cy - ty) * (ns / scale);
  scale = ns; apply();
}}
document.getElementById('fitw').onclick = () => fit('w');
document.getElementById('fith').onclick = () => fit('h');
document.getElementById('one').onclick  = () => {{ scale = 1; tx = 12; ty = 12; apply(); }};
document.getElementById('zi').onclick   = () => {{ const r = stage.getBoundingClientRect(); zoomAt(r.width/2, r.height/2, 1.25); }};
document.getElementById('zo').onclick   = () => {{ const r = stage.getBoundingClientRect(); zoomAt(r.width/2, r.height/2, 0.8); }};
stage.addEventListener('wheel', (e) => {{
  e.preventDefault();
  const r = stage.getBoundingClientRect();
  zoomAt(e.clientX - r.left, e.clientY - r.top, e.deltaY < 0 ? 1.12 : 0.89);
}}, {{ passive: false }});
let drag = null;
stage.addEventListener('mousedown', (e) => {{ drag = {{ x: e.clientX - tx, y: e.clientY - ty }}; stage.classList.add('dragging'); }});
window.addEventListener('mousemove', (e) => {{ if (!drag) return; tx = e.clientX - drag.x; ty = e.clientY - drag.y; apply(); }});
window.addEventListener('mouseup', () => {{ drag = null; stage.classList.remove('dragging'); }});
stage.addEventListener('dblclick', () => fit('w'));
window.addEventListener('keydown', (e) => {{
  if (e.key === '+' || e.key === '=') document.getElementById('zi').click();
  else if (e.key === '-') document.getElementById('zo').click();
  else if (e.key === '0') fit('w');
}});
window.addEventListener('resize', () => fit('w'));
fit('w');
</script>
<!--EXTRA_JS-->
</body>
</html>
"""


def strip_katex_fallbacks(css: str) -> str:
    css = re.sub(r",\s*url\([^)]*\.woff\)[^,;]*", "", css)
    css = re.sub(r",\s*url\([^)]*\.ttf\)[^,;]*", "", css)
    return css


def inline_fonts(svg_text: str, base: Path) -> str:
    """把 @font-face 里的 url(fonts/*.woff2) 换成 base64 data URI。

    为什么需要：SVG 被 markdown 以 `<img src="...svg">` 引用时，浏览器出于安全
    考虑**不会**为它加载任何外部资源（字体、图片、样式表都不行）。字体加载不了，
    KaTeX 就会退回系统衬线体；字形宽度与算布局时用的不一样，文本会溢出错位。
    内联之后 SVG 完全自包含，放在任何 markdown 查看器里都和查看器里一致。
    只对内联进文档的那份 OUT_SVG 做（查看器 HTML 与 fonts/ 同目录，不需要）。
    """
    cache: dict[str, str] = {}

    def repl(m: re.Match[str]) -> str:
        rel = m.group(1)
        if rel not in cache:
            p = base / rel
            if not p.exists():
                cache[rel] = m.group(0)
            else:
                b64 = base64.b64encode(p.read_bytes()).decode("ascii")
                cache[rel] = f"url(data:font/woff2;base64,{b64})"
        return cache[rel]

    return re.sub(r"url\((fonts/[^)]+\.woff2)\)", repl, svg_text)


def manual_links(svg_text: str, mmd_text: str):
    """把流程图节点接到《项目全景手册》：节点按章节着色，点节点给出手册章节 + 源码行。

    依赖 `_manual/flowmap.py`（节点 → 章节 → 候选源码文件）；拿不到就静默跳过，
    查看器仍按原样生成。
    """
    src_root = HERE.parents[1]                      # _v6src
    for cand in (HERE, src_root.parent / "_manual", src_root.parent):   # 本目录 / 仓库的 _manual
        sys.path.insert(0, str(cand))
    try:
        import flowmap                              # type: ignore
    except Exception as e:                          # pragma: no cover
        print(f"[WARN] 载入 flowmap 失败（{e}），跳过手册锚点", file=sys.stderr)
        return svg_text, "", "", ""

    known = {p.relative_to(src_root).as_posix() for p in src_root.rglob("*.py")}
    svg_text, nodes, chap2nodes, stats = flowmap.build(mmd_text, svg_text, known)
    if not nodes:
        return svg_text, "", "", ""

    docs = None
    _dm = HERE / "flow_docmap.json"                 # 由 _manual/gen_manual.py 生成（节点 → 手册段落）
    if _dm.exists():
        try:
            docs = json.loads(_dm.read_text(encoding="utf-8"))
            docs = {k: v for k, v in docs.items() if k in nodes}
        except Exception as e:
            print(f"[WARN] 读 flow_docmap.json 失败（{e}）", file=sys.stderr)
            docs = None
    data = flowmap.detail_json(nodes, docs) if hasattr(flowmap, "detail_json") else json.dumps(
        {nid: {"t": m["title"], "c": m["cid"], "n": flowmap.CID_NAME.get(m["cid"], ""),
               "lab": [], "cite": [], "vllm": [], "vcmp": [], "doc": None} for nid, m in nodes.items()},
        ensure_ascii=False)

    # 面板与手册共用同一套（flowmap.PANEL_CSS / PANEL_JS）：这一步做什么 + 精确代码片段 +
    # 「在 VS Code 打开」深链 + vLLM 对照；宿主只提供章节链接与"手册里那一行"的锚点。
    css = flowmap.PANEL_CSS
    body = '<div id="flowpanel" hidden></div>'
    js = """
<script>
window.FLOWD = __DATA__;
const M = '../__MANUAL__';
const sg = s => String(s).replace(/[^0-9a-zA-Z]+/g, '-').replace(/^-+|-+$/g, '').toLowerCase();
function FLOWCHAP(cid) { return M + '#sec-' + (cid || '').toLowerCase(); }
function FLOWDOC(a) { return M + '#' + a; }
function FLOWSRC(rel, line) { return line ? (M + '#src-' + sg(rel) + '-L' + line) : null; }
__PANELJS__
flowPanelInit();
</script>
"""
    js = js.replace('__DATA__', data).replace('__MANUAL__', MANUAL_NAME).replace('__PANELJS__', flowmap.PANEL_JS)
    print(f"[manual] 节点 {stats['nodes']} 个接入手册锚点（链接到 {MANUAL_NAME}）")
    return svg_text, css, body, js


def build_one(cfg: dict, check_only: bool) -> int:
    MMD = HERE / cfg["src"]
    SVG = HERE / cfg["svg"]
    OUT_MMD = HERE / cfg["out_mmd"]
    OUT_SVG = HERE / cfg["out_svg"]
    OUT_HTML = HERE / cfg["out_html"]

    if not MMD.exists():
        print(f"[ERR] 缺少图源：{MMD}", file=sys.stderr)
        return 1

    if check_only:
        if not OUT_HTML.exists() or not OUT_SVG.exists():
            print(f"[ERR] {OUT_HTML.name} 产物缺失，需要重跑", file=sys.stderr)
            return 1
        html_m = OUT_HTML.stat().st_mtime
        ok = MMD.stat().st_mtime <= html_m and SVG.stat().st_mtime <= html_m
        print(("已同步: " if ok else "[ERR] 图源比产物新: ") + OUT_HTML.name)
        return 0 if ok else 1

    if not SVG.exists():
        print(f"[ERR] 缺少渲染产物 {SVG}（先用无头 Chrome 跑 _render.html）", file=sys.stderr)
        return 1

    svg_text = SVG.read_text(encoding="utf-8")
    # 只保留根 <svg>，去掉 XML 声明
    svg_text = re.sub(r"^<\?xml[^>]*\?>\s*", "", svg_text)
    # 给内联 SVG 一个稳定的 class，便于样式
    svg_text = svg_text.replace("<svg ", '<svg class="flow-svg" ', 1)

    OUT_MMD.write_text(MMD.read_text(encoding="utf-8"), encoding="utf-8")

    raw_svg = SVG.read_text(encoding="utf-8")
    if cfg.get("embed_svg"):
        # 供 markdown 以 <img> 直接引用：字体必须内联，否则浏览器不给 <img> 里的
        # SVG 加载外部字体。查看器 HTML 不用（它与 fonts/ 同目录，且要控制体积）。
        OUT_SVG.write_text(inline_fonts(raw_svg, HERE), encoding="utf-8")
    else:
        OUT_SVG.write_text(raw_svg, encoding="utf-8")

    empty_display = "block" if "<svg" not in svg_text else "none"
    empty_msg = "未找到内联 SVG：请先跑无头渲染生成对应的 *_rendered.svg。"
    extra_css = extra_body = extra_js = ""
    if cfg.get("manual_link"):
        svg_text, extra_css, extra_body, extra_js = manual_links(svg_text, MMD.read_text(encoding="utf-8"))
    OUT_HTML.write_text(
        HTML_TEMPLATE.format(title=cfg["title"], svg=svg_text,
                             empty_display=empty_display, empty_msg=empty_msg)
        .replace("<!--EXTRA_CSS-->", extra_css)
        .replace("<!--EXTRA_BODY-->", extra_body)
        .replace("<!--EXTRA_JS-->", extra_js),
        encoding="utf-8",
    )
    print(f"已生成 {OUT_HTML.name}（{OUT_HTML.stat().st_size} bytes）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", default="", help="只重建这一张图（src 文件名，如 merged_flow.mmd）")
    args = ap.parse_args()

    rc = 0
    for cfg in BUILDS:
        if args.only and cfg["src"] != args.only:
            continue
        rc |= build_one(cfg, args.check)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
