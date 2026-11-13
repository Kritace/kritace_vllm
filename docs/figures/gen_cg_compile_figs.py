#!/usr/bin/env python3
"""生成 CUDA Graph / torch.compile 三篇教学文档的插图（docs/posts/*.png）。

用法（在 vllm-v6 下，或任意目录）：
    python docs/posts/gen_cg_compile_figs.py

数据来源（每个数字都能在仓库里查到，见各函数 docstring 的出处标注）：
    rental/实测结果汇总.md                       —— 端到端吞吐矩阵
    profiler/tutorial/01扫盲/README.md            —— 五种 compile 模式、三方案单步耗时
    profiler/tutorial/01扫盲/traces/**/*.txt      —— torch.profiler 原始表
    vllm-v3/docs/08-Prefill-Compile.md            —— TTFT 随 prompt 长度

口径（引用任何一张图都必须一起说）：
    2 × RTX 4090 D 24GB（sm_89），无 NVLink、P2P 不支持（卡间数据过主机内存）；
    torch 2.8.0+cu128 / flash-attn 2.8.3 / flashinfer 0.7.0；
    模型 Qwen3-0.6B（28 层）/ Qwen1.5-MoE-A2.7B / Qwen3-8B（36 层）；单请求 greedy。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- 中文字体
# 本机没有 DroidSansFallback（v4 的 gen_*.py 用的那个），所以按可用性挑一个。
# 缺字体会让中文变成方框 —— 渲染完请打开图看一眼，别只看脚本没报错。
_CJK_CANDIDATES = [
    # Windows 原生
    r"C:\Windows\Fonts\msyh.ttc",             # 微软雅黑
    r"C:\Windows\Fonts\simhei.ttf",           # 黑体
    # WSL 里读 Windows 字体
    "/mnt/c/Windows/Fonts/msyh.ttc",
    "/mnt/c/Windows/Fonts/simhei.ttf",
    # Linux 自带
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
for _p in _CJK_CANDIDATES:
    if Path(_p).exists():
        fm.fontManager.addfont(_p)
        break
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                               "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------- 配色
C_FIX = "#9ca3af"      # 固定开销（提交 + 空隙）
C_CMP = "#2563eb"      # 真实计算
C_CPU = "#f59e0b"      # CPU 侧
C_GPU = "#2563eb"      # GPU 侧
C_BAD = "#dc2626"
C_OK = "#16a34a"
C_BG = "#f8fafc"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#cbd5e1",
    "axes.labelcolor": "#334155",
    "text.color": "#1e293b",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "axes.grid": True,
    "grid.color": "#e2e8f0",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})


def _save(fig, name: str) -> None:
    p = OUT / name
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {name}  ({p.stat().st_size // 1024} KiB)")


# =====================================================================
# 图 1：一步 decode 的时间去哪了 —— 04 的核心公式的可视化
# =====================================================================
def fig_step_breakdown() -> None:
    """$T_step = T_fixed + T_compute$，CG 只消掉前一项。

    数据：rental/实测结果汇总.md §三（v4 全矩阵）
      0.6B 单卡 eager 49.5 tok/s → 20.20 ms/步；cg 449.7 → 2.22 ms/步
      8B   单卡 eager 36.0 tok/s → 27.78 ms/步；cg  58.1 → 17.21 ms/步
    T_fixed 由 (eager 每步 − cg 每步) 反解，含义是「没能被重叠掉的固定开销」。
    """
    rows = [
        ("Qwen3-0.6B\n单卡 eager", 17.98, 2.22, None),
        ("Qwen3-0.6B\n单卡 cg", 0.0, 2.22, "9.1×"),
        ("Qwen3-8B\n单卡 eager", 10.57, 17.21, None),
        ("Qwen3-8B\n单卡 cg", 0.0, 17.21, "1.61×"),
    ]
    labels = [r[0] for r in rows]
    fixed = np.array([r[1] for r in rows])
    comp = np.array([r[2] for r in rows])
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.bar(x, comp, 0.56, color=C_CMP, label=r"$T_{\mathrm{compute}}$（真正在算）")
    ax.bar(x, fixed, 0.56, bottom=comp, color=C_FIX,
           label=r"$T_{\mathrm{fixed}}$（提交 + kernel 间空隙，与计算量无关）")

    for i, r in enumerate(rows):
        total = r[1] + r[2]
        ax.text(i, total + 0.45, f"{total:.2f} ms", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#1e293b")
        if r[3]:
            ax.text(i, total / 2, r[3], ha="center", va="center", fontsize=15,
                    fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.28", fc=C_OK, ec="none"))
        if r[1] > 0.5:
            ax.text(i, r[2] + r[1] / 2, f"{r[1]:.1f}", ha="center", va="center",
                    fontsize=9.5, color="#334155")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("一步 decode 的墙钟耗时（ms）")
    ax.set_ylim(0, 31.5)
    ax.set_title("CUDA Graph 消掉的是「与计算量无关」的那一段\n"
                 "倍数 = " + r"$1 + T_{\mathrm{fixed}} / T_{\mathrm{compute}}$"
                 + "　·　2×4090D 无 NVLink · 单请求 greedy",
                 fontsize=12.5, pad=12)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95)
    ax.annotate("0.6B 的固定开销占 89%\n→ 图一开，几乎全消",
                xy=(0, 20.2), xytext=(0.58, 9.2), fontsize=9.5, color="#475569",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.2))
    ax.annotate("8B 的固定开销只占 38%\n→ 只剩 1.61×",
                xy=(1.98, 27.0), xytext=(1.66, 22.3), fontsize=9.5, color="#475569",
                ha="right", va="center",
                arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.2))
    _save(fig, "cg_step_breakdown.png")


# =====================================================================
# 图 2：提交一次 kernel vs 算完这个 kernel —— 「开销受限 → 计算受限」的反转
# =====================================================================
def fig_launch_vs_kernel() -> None:
    """数据：profiler/tutorial/01扫盲/traces/01_matmul_add_graph/{64,4096}_eager.txt

    64×64 : cuLaunchKernel avg 6.872 µs；GEMM 主 kernel 2.613 µs
    4096² : cuLaunchKernel avg 7.346 µs；GEMM 主 kernel 980.075 µs
    两个尺寸下 CPU 侧提交开销几乎不变（6.87 → 7.35），GPU 侧涨了 375 倍。
    """
    sizes = ["64 × 64", "4096 × 4096"]
    launch = np.array([6.872, 7.346])
    kernel = np.array([2.613, 980.075])
    x = np.arange(2)
    w = 0.34

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    b1 = ax.bar(x - w / 2, launch, w, color=C_CPU, label="CPU 提交一次 kernel（cuLaunchKernel avg）")
    b2 = ax.bar(x + w / 2, kernel, w, color=C_GPU, label="这个 kernel 真正算完（CUDA avg）")
    ax.set_yscale("log")
    ax.set_ylim(0.6, 30000)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=11)
    ax.set_ylabel("耗时（µs，对数轴）")
    ax.set_title("同样一次「提交」，小矩阵上比它要算的活还贵 2.6 倍\n"
                 "bf16 matmul+add · 单卡 · torch.profiler 20 次调用",
                 fontsize=12.5, pad=12)

    for b, v in zip(b1, launch):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.18, f"{v:.2f}", ha="center",
                va="bottom", fontsize=10, fontweight="bold", color="#92400e")
    for b, v in zip(b2, kernel):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.18, f"{v:.3f}" if v < 10 else f"{v:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1d4ed8")

    # 双向箭头：一眼看出哪根柱子更长
    ax.annotate("", xy=(0 - w / 2, 6.872), xytext=(0 + w / 2, 2.613),
                arrowprops=dict(arrowstyle="<->", color=C_BAD, lw=1.6))
    ax.text(0, 33, "开销受限\nCPU 是瓶颈", ha="center", va="bottom", fontsize=10,
            color=C_BAD, fontweight="bold", linespacing=1.4)
    ax.text(1, 2600, "计算受限：GPU 是瓶颈，CPU 提交开销被 980 µs 的计算淹没",
            ha="center", va="bottom", fontsize=10, color=C_OK, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95)
    _save(fig, "cg_launch_vs_kernel.png")


# =====================================================================
# 图 3：收益为什么随模型变大而掉下来
# =====================================================================
def fig_gain_vs_size() -> None:
    """左：CG 的 decode 倍数；右：每层 eager 耗时 vs 参数量。

    数据：rental/实测结果汇总.md §三、§六
      0.6B 49.5→449.7 = 9.1×（28 层，20.20/28 = 0.72 ms/层）
      MoE  29.29→201.04 = 6.9×（TP=2）
      8B   36.0→58.1 = 1.61×（36 层，27.78/36 = 0.77 ms/层）
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.3),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    names = ["Qwen3-0.6B\n单卡", "Qwen1.5-MoE-A2.7B\nTP=2", "Qwen3-8B\n单卡"]
    gains = [9.1, 6.9, 1.61]
    colors = [C_OK, C_OK, C_BAD]
    bars = ax1.bar(names, gains, 0.55, color=colors)
    for b, g in zip(bars, gains):
        ax1.text(b.get_x() + b.get_width() / 2, g + 0.22, f"{g}×", ha="center",
                 va="bottom", fontsize=12, fontweight="bold")
    ax1.set_ylabel("CG 相对 eager 的 decode 倍数")
    ax1.set_ylim(0, 11)
    ax1.axhline(1.0, color="#94a3b8", ls="--", lw=1)
    ax1.text(1.5, 1.30, "不开图 = 1×", fontsize=8.5, color="#64748b", ha="center")
    ax1.set_title("同一个机制，收益差 5.6 倍", fontsize=11.5)
    ax1.tick_params(axis="x", labelsize=9.5)

    ax2.bar(["Qwen3-0.6B", "Qwen3-8B"], [0.72, 0.77], 0.45, color=C_CMP)
    for i, v in enumerate([0.72, 0.77]):
        ax2.text(i, v + 0.018, f"{v:.2f} ms", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 0.98)
    ax2.set_ylabel("eager 下每层 decode 耗时（ms）")
    ax2.set_title("参数量差 13 倍，每层耗时几乎一样\n→ 时间不是被「算」主导的",
                  fontsize=11.5)
    ax2.text(0.5, 0.30, "0.6B：28 层\n8B：36 层", ha="center", fontsize=9.5,
             color="#64748b",
             bbox=dict(boxstyle="round,pad=0.4", fc=C_BG, ec="#cbd5e1"))

    fig.suptitle("CUDA Graph 的收益只取决于「固定开销占真实计算的比例」",
                 fontsize=13, y=1.10)
    fig.tight_layout()
    _save(fig, "cg_gain_vs_size.png")


# =====================================================================
# 图 4：compile 对 prefill 的加速 —— 短 prompt 收益大，长 prompt 收敛
# =====================================================================
def fig_ttft_vs_prompt() -> None:
    """数据：vllm-v3/docs/08-Prefill-Compile.md（bench_prefill.py，greedy）

    两边都开 decode CG，只隔离 compile 这一项：
      prompt  32 → eager 16.47 ms / compile  2.77 ms（5.94×）
      prompt 128 →       16.23 /        3.20（5.08×）
      prompt 512 →       15.97 /        6.40（2.49×）
      prompt1024 →       15.79 /       12.00（1.32×）
      prompt2048 →       22.94 /       22.25（1.03×）
    """
    L = np.array([32, 128, 512, 1024, 2048])
    eager = np.array([16.47, 16.23, 15.97, 15.79, 22.94])
    comp = np.array([2.77, 3.20, 6.40, 12.00, 22.25])
    speed = eager / comp

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(L, eager, "o-", color=C_BAD, lw=2.2, ms=7,
            label="eager prefill（不编译）")
    ax.plot(L, comp, "s-", color=C_OK, lw=2.2, ms=7,
            label="torch.compile prefill")
    ax.set_xscale("log", base=2)
    ax.set_xticks(L)
    ax.set_xticklabels([str(v) for v in L])
    ax.set_xlabel("prompt 长度（token）")
    ax.set_ylabel("TTFT（ms）")
    ax.set_ylim(0, 29)
    ax.set_title("compile 对 prefill 的加速：短 prompt 5.9×，长 prompt 收敛到 1.03×\n"
                 "Qwen3-0.6B · 单卡 4090D · 两边都开 decode CG，只隔离 compile 这一项",
                 fontsize=12, pad=12)

    for xi, ei, ci, si in zip(L, eager, comp, speed):
        ax.text(xi, ei + 0.85, f"{ei:.1f}", ha="center", fontsize=9, color="#991b1b")
        if xi == L[-1]:
            # 最后一个点：两条线在这里交汇，绿标签左移，避开红线
            ax.text(xi * 0.80, ci - 0.35, f"{ci:.2f}", ha="right", va="center",
                    fontsize=9, color="#166534")
        else:
            ax.text(xi, ci - 1.55, f"{ci:.2f}", ha="center", fontsize=9,
                    color="#166534")
        ax.text(xi, 26.4, f"{si:.2f}×", ha="center", fontsize=10.5,
                fontweight="bold", color="#0f172a",
                bbox=dict(boxstyle="round,pad=0.22", fc="#fef3c7", ec="none"))
    ax.axhspan(0, 3.5, color=C_OK, alpha=0.06)
    ax.text(40, 4.3, "短 prompt：TTFT 几乎全是「提交开销」→ 编译收益最大",
            fontsize=9.5, color="#166534")
    ax.annotate("长 prompt：GPU 算力成了物理下界\n两条线必然收敛",
                xy=(1900, 21.6), xytext=(80, 9.0), fontsize=9.5, color="#475569",
                ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.2))
    ax.legend(loc="center left", fontsize=10, framealpha=0.95)
    _save(fig, "compile_ttft_vs_prompt.png")


# =====================================================================
# 图 5：compile 五种模式 —— 小矩阵全变慢，大矩阵才有反超
# =====================================================================
def fig_compile_modes() -> None:
    """数据：profiler/tutorial/01扫盲/README.md §4.3（bf16，100 次调用，取中位）

    none                        64² 30.0 µs / 4096² 1036.4 µs
    default                     64² 136.7  / 4096² 1051.0
    reduce-overhead             64² 150.5  / 4096² 1142.0
    max-autotune                64² 148.2  / 4096² 1115.7
    max-autotune-no-cudagraphs  64²  83.1  / 4096²  936.9
    """
    modes = ["none\n(eager 基线)", "default", "reduce-\noverhead",
             "max-\nautotune", "max-autotune-\nno-cudagraphs"]
    v64 = np.array([30.0, 136.7, 150.5, 148.2, 83.1])
    v4096 = np.array([1036.4, 1051.0, 1142.0, 1115.7, 936.9])
    r64 = v64 / v64[0]
    r4096 = v4096 / v4096[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6))
    x = np.arange(len(modes))
    w = 0.38

    for ax, raw, rel, title, ylab in (
        (ax1, v64, r64, "64 × 64（开销受限）", "单次调用耗时（µs）"),
        (ax2, v4096, r4096, "4096 × 4096（计算受限）", "单次调用耗时（µs）"),
    ):
        colors = [C_FIX] + [C_BAD] * 3 + [C_OK]
        bars = ax.bar(x, raw, 0.6, color=colors)
        for b, v, rr in zip(bars, raw, rel):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.03, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold")
            ax.text(b.get_x() + b.get_width() / 2, v * 0.5, f"{rr:.2f}×",
                    ha="center", va="center", fontsize=10.5, color="white",
                    fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(modes, fontsize=8.5)
        ax.set_title(title, fontsize=11.5)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, raw.max() * 1.62)

    ax1.text(2.0, 150.5 * 1.30, "小矩阵：编译只优化 GPU 启动，\n却在 CPU 引入新开销 → 全部更慢",
             ha="center", va="bottom", fontsize=9.5, color=C_BAD, linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.35", fc="#fef2f2", ec="#fecaca"))
    ax2.text(2.0, 1142.0 * 1.24, "大矩阵：计算受限，\nInductor 调优的 Triton kernel 反超 cuBLAS",
             ha="center", va="bottom", fontsize=9.5, color=C_OK, linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.35", fc="#f0fdf4", ec="#bbf7d0"))

    fig.suptitle("torch.compile 的五种 mode：同一组开关，在两个尺寸上结论相反"
                 "　（bf16 · 单卡 · 100 次调用取中位）",
                 fontsize=12.5, y=1.10)
    fig.tight_layout()
    _save(fig, "compile_five_modes.png")


# =====================================================================
# 图 6：三方案的单步 CPU / GPU 耗时 —— CG 把 CPU 侧打下来，GPU 侧不动
# =====================================================================
def fig_three_way() -> None:
    """数据：profiler/tutorial/01扫盲/README.md §4.4（三方案 profiler 单步耗时）

    eager           64² CPU  75.3 µs / GPU 5.63 µs ；4096² CPU  66.1 µs / GPU 1.034 ms
    compile（无图） 64² CPU 184.6 µs / GPU 1.44 µs ；4096² CPU 209.3 µs / GPU 0.890 ms
    compile + 图    64² CPU  23.2 µs / GPU 1.43 µs ；4096² CPU  15.8 µs / GPU 0.958 ms
    """
    schemes = ["eager", "compile\n（无图）", "compile\n+ 手动 CUDA Graph"]
    cpu64 = np.array([75.3, 184.6, 23.2])
    gpu64 = np.array([5.63, 1.44, 1.43])
    cpu4096 = np.array([66.1, 209.3, 15.8])
    gpu4096 = np.array([1034.0, 890.0, 958.0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    x = np.arange(3)
    w = 0.36

    for ax, cpu, gpu, title, unit, legloc in (
        (ax1, cpu64, gpu64, "64 × 64（开销受限）", "µs", "upper right"),
        (ax2, cpu4096, gpu4096, "4096 × 4096（计算受限）", "µs", "upper left"),
    ):
        b1 = ax.bar(x - w / 2, cpu, w, color=C_CPU, label="CPU 侧")
        b2 = ax.bar(x + w / 2, gpu, w, color=C_GPU, label="GPU 侧")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(schemes, fontsize=9.5)
        ax.set_title(title, fontsize=11.5)
        ax.set_ylabel(f"单步耗时（{unit}，对数轴）")
        top = max(cpu.max(), gpu.max()) * 9
        ax.set_ylim(0.8, top)
        for bars, col in ((b1, "#92400e"), (b2, "#1d4ed8")):
            for b in bars:
                v = b.get_height()
                ax.text(b.get_x() + b.get_width() / 2, v * 1.12,
                        f"{v:.2f}" if v < 10 else f"{v:.0f}", ha="center",
                        va="bottom", fontsize=9, color=col)
        ax.legend(loc=legloc, fontsize=9.5, framealpha=0.95)

    ax1.text(0.5, 430, "编译让 GPU 更快（5.63→1.44）\n但 CPU 更慢（75.3→184.6）",
             ha="center", va="bottom", fontsize=9.5, color=C_BAD, linespacing=1.5)
    ax1.text(1.62, 62, "再加图：CPU 184.6→23.2\nGPU 侧 1.44→1.43 纹丝不动",
             ha="center", va="bottom", fontsize=9.5, color=C_OK, linespacing=1.5)

    fig.suptitle("三方案的单步耗时：CUDA Graph 只动 CPU 侧，算子融合只动 GPU 侧"
                 "　（bf16 · 单卡 · torch.profiler）",
                 fontsize=12.5, y=1.10)
    fig.tight_layout()
    _save(fig, "cg_three_way_profiler.png")


def fig_tp_comm_breakdown() -> None:
    """TP=2 的账为什么算不平 —— CG 吃掉的是通信的「提交」，不是「传输」。

    数据：rental/实测结果汇总.md §三（v4 全矩阵，Qwen3-0.6B 单请求 greedy）
      单卡 eager 49.5 tok/s → 20.20 ms/步
      单卡 cg   449.7 tok/s →  2.224 ms/步
      TP=2 eager 37.9 tok/s → 26.39 ms/步
      TP=2 cg   401.9 tok/s →  2.488 ms/步

    推导（详见 06-图与编译的边界与观测.md §2.2）：
      C1 = 2.224（单卡真实计算）          F1 = 20.20 - 2.224 = 17.98（固定开销）
      C2 = C1 / 2 = 1.112（TP 把计算切半）
      T_AR(cg)    = 2.488 - 1.112 = 1.376 ms/步  ← 干净：cg 下没有启动开销
      T_AR(eager) ≤ 6.19 + 1.112 = 7.30 ms/步    ← 上界：混着 57 个新 kernel 的启动开销
      一次前向 2×28 + 1 = 57 次集合通信 ⇒ cg 24 µs/次、eager ≤128 µs/次
      纯传输：消息是 [T,H] = 1×1024×2B = 2 KiB，按 all-reduce busbw 7.09 GB/s
              ⇒ 2 KiB / 7.09 GB/s ≈ 0.29 µs/次（这是下界，小消息实际由延迟主导）
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.8))

    # ---------- 左：每步毫秒的三段拆分 ----------
    labels = ["单卡 cg", "TP=2 cg", "单卡 eager", "TP=2 eager"]
    comp = np.array([2.224, 1.112, 2.224, 1.112])
    comm = np.array([0.0, 1.376, 0.0, 7.30])
    fixed = np.array([0.0, 0.0, 17.98, 17.98])
    x = np.arange(4)

    ax1.bar(x, comp, 0.56, color=C_CMP, label=r"真实计算 $T_{\mathrm{compute}}$")
    ax1.bar(x, comm, 0.56, bottom=comp, color=C_BAD,
            label=r"集合通信 $T_{\mathrm{AR}}$（eager 含新增启动开销，是上界）")
    ax1.bar(x, fixed, 0.56, bottom=comp + comm, color=C_FIX,
            label=r"固定开销 $T_{\mathrm{fixed}}$（CG 消掉的那段）")

    for i in range(4):
        total = comp[i] + comm[i] + fixed[i]
        ax1.text(i, total + 0.5, f"{total:.2f} ms", ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color="#1e293b")
        if comm[i] > 0.5:
            ax1.text(i, comp[i] + comm[i] / 2, f"{comm[i]:.2f}", ha="center", va="center",
                     fontsize=9.5, color="white", fontweight="bold")
        if comp[i] > 0.3:
            ax1.text(i, comp[i] / 2, f"{comp[i]:.2f}", ha="center", va="center",
                     fontsize=9.5, color="white", fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("一步 decode 的墙钟耗时（ms）")
    ax1.set_ylim(0, 33)
    ax1.set_title("TP 的账：cg 下每步多出 1.376 ms，全部是通信", fontsize=11.5)
    ax1.legend(loc="upper left", fontsize=8.8, framealpha=0.95)

    ax1.annotate("同一份通信，\n两种模式差 5 倍以上\n⇒ 差额不是「传输」",
                 xy=(1.24, 1.9), xytext=(1.30, 12.6), fontsize=9.5, color=C_BAD,
                 ha="left", va="center", linespacing=1.5,
                 arrowprops=dict(arrowstyle="->", color=C_BAD, lw=1.4,
                                 connectionstyle="arc3,rad=-0.18"))

    # ---------- 右：摊到每一次集合通信 ----------
    ax2.set_xscale("log")
    names = ["纯传输下界\n2 KiB @ 7.09 GB/s", "cg 下实测\n（纯通信）", "eager 下实测\n（通信 + 启动，上界）"]
    vals = np.array([0.29, 24.0, 128.0])
    colors = [C_OK, C_CMP, C_BAD]
    ypos = np.arange(3)[::-1]
    bars = ax2.barh(ypos, vals, 0.5, color=colors)
    for b, v in zip(bars, vals):
        ax2.text(v * 1.25, b.get_y() + b.get_height() / 2, f"{v:g} µs",
                 va="center", ha="left", fontsize=11, fontweight="bold")

    ax2.set_yticks(ypos)
    ax2.set_yticklabels(names, fontsize=9.5)
    ax2.set_xlabel("一次集合通信摊到的暴露代价（µs，对数轴）")
    ax2.set_xlim(0.15, 900)
    ax2.set_title("一次 all-reduce 的代价拆开看", fontsize=11.5)

    ax2.text(0.36, 1.5, "83×", fontsize=11, fontweight="bold", color="#475569")
    ax2.text(0.36, 0.5, "5.3×", fontsize=11, fontweight="bold", color="#475569")

    fig.suptitle("CUDA Graph 消掉的是集合通信的「提交」，不是「传输」"
                 "　（Qwen3-0.6B · 2×4090D 无 NVLink · 单请求 greedy · 一次前向 57 次集合通信）",
                 fontsize=12.5, y=1.11)
    fig.tight_layout()
    _save(fig, "cg_tp_comm_breakdown.png")


# =====================================================================
# 图 8：piecewise 是唯一「两项都变差」的方案 —— 05 §4.2
# =====================================================================
def fig_kernel_copy() -> None:
    """三方案的 kernel 数与 copy 数对照（L=512，profiler 实测）。

    数据：vllm-v3/docs/08-Prefill-Compile.md
      eager    : kernel 646 / copy 168
      piecewise: kernel 784 / copy 267   ← 两项都变差，方案被放弃
      compile  : kernel 537 / copy 116
    """
    schemes = ["eager", "piecewise CG\n（已放弃）", "torch.compile"]
    kernels = np.array([646, 784, 537])
    copies = np.array([168, 267, 116])
    x = np.arange(3)
    w = 0.34

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    b1 = ax.bar(x - w / 2, kernels, w, color=C_CPU, label="kernel 个数")
    b2 = ax.bar(x + w / 2, copies, w, color=C_GPU, label="copy 次数")

    for bars, vals in ((b1, kernels), (b2, copies)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.06, f"{v}",
                    ha="center", va="bottom", fontsize=10.5, fontweight="bold")

    # 把 piecewise 那组圈出来 —— 它是唯一两项都变差的
    ax.axvspan(0.5, 1.5, color=C_BAD, alpha=0.07, zorder=0)
    ax.text(1.0, 900, "唯一两项都变差的方案", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=C_BAD)

    ax.annotate("", xy=(0.83, 830), xytext=(0.17, 700),
                arrowprops=dict(arrowstyle="->", color=C_BAD, lw=1.5))
    ax.text(0.5, 845, "+138", ha="center", fontsize=9.5, color=C_BAD, fontweight="bold")
    ax.annotate("", xy=(1.17, 300), xytext=(1.83, 140),
                arrowprops=dict(arrowstyle="->", color=C_OK, lw=1.5))
    ax.text(1.5, 320, "−52", ha="center", fontsize=9.5, color=C_OK, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(schemes, fontsize=10.5)
    ax.set_ylabel("计数")
    ax.set_ylim(0, 1010)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title("piecewise CUDA Graph 为什么被放弃：kernel 与 copy 双双上升",
                 fontsize=12)
    fig.suptitle("L=512 · 单卡 RTX 4090 D · Qwen3-0.6B · torch.profiler 实测",
                 fontsize=10.5, y=1.03)
    fig.tight_layout()
    _save(fig, "compile_kernel_copy.png")


# =====================================================================
# 图 9：投机收益完全由文本重复度决定 —— 06 §3.2
# =====================================================================
def fig_spec_gain_by_text() -> None:
    """三条文本类型 × 四档 K 的相对加速比。

    数据：rental/实测结果汇总.md §九 9.2（本仓库 v6）
      高重复: K=0/3/5/8 → 48.7 / 151.3 / 182.6 / 285.9
      自然散文: 49.9 / 74.5 / 82.1 / 87.6
      代码:     50.4 / 58.8 / 58.4 / 58.6
    括号里是相对各自 K=0 的倍数：5.87× / 1.75× / 1.16×
    """
    ks = np.array([0, 3, 5, 8])
    series = [
        ("高重复（问答模板）", [48.7, 151.3, 182.6, 285.9], C_OK),
        ("自然散文", [49.9, 74.5, 82.1, 87.6], C_CPU),
        ("代码", [50.4, 58.8, 58.4, 58.6], C_BAD),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for name, vals, color in series:
        base = vals[0]
        rel = np.array(vals) / base
        ax.plot(ks, rel, "o-", color=color, lw=2.2, ms=7, label=name)
        ax.text(ks[-1] + 0.24, rel[-1], f"{rel[-1]:.2f}×", color=color,
                fontsize=11, fontweight="bold", va="center")
        for k, r in zip(ks, rel):
            if k == 0:
                continue
            # 绿线最后一点让开右侧的粗体倍数标签
            if name.startswith("高重复") and k == 8:
                continue
            ax.text(k, r + 0.14, f"{r:.2f}", ha="center", fontsize=8.8, color=color)

    ax.axhline(1.0, color="#94a3b8", lw=1.2, ls="--")
    ax.text(6.6, 1.34, "不开投机 = 1.00×（基线）", fontsize=9, color="#64748b",
            ha="right", va="bottom")

    ax.annotate("代码文本 K=3 就到顶，\nK 再大反而略降",
                xy=(5, 1.16), xytext=(3.35, 2.35), fontsize=10, color=C_BAD,
                ha="left", va="center", linespacing=1.5,
                arrowprops=dict(arrowstyle="->", color=C_BAD, lw=1.4,
                                connectionstyle="arc3,rad=0.22"))
    ax.annotate("同一套引擎、同一份代码，\n收益差 5 倍 —— 全看文本自身重复度",
                xy=(7.75, 5.60), xytext=(2.6, 5.60), fontsize=10, color="#334155",
                ha="left", va="center", linespacing=1.5,
                arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.3,
                                connectionstyle="arc3,rad=-0.14"))

    ax.set_xticks(ks)
    ax.set_xlabel("每步提交验证的草稿长度 $K$")
    ax.set_ylabel("相对各自 $K=0$ 的加速比")
    ax.set_ylim(0.5, 6.6)
    ax.set_xlim(-0.45, 9.2)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.set_title("投机解码的收益完全由文本自身的重复度决定", fontsize=12.5)
    fig.suptitle("Qwen3-0.6B · 单请求 64 token · eager · n-gram 草稿 · 2×4090D",
                 fontsize=10.5, y=1.03)
    fig.tight_layout()
    _save(fig, "spec_gain_by_text.png")


# =====================================================================
# 图 10：EP × CUDA Graph 的负交互 —— 06 §2.5
# =====================================================================
def fig_ep_interaction() -> None:
    """2×2 分组柱状：EP 的效果在 eager 下无害、在 cg 下有害。

    数据：rental/实测结果汇总.md §四（v5，Qwen1.5-MoE-A2.7B，2 卡 TP=2）
      无 EP eager 29.29 / cg 201.04
      有 EP eager 30.01 / cg 161.33
    负交互定义：一个因素的效果依赖于另一个因素的取值 —— 这里 EP 的效果
    在 eager 下是 +2.5%，在 cg 下是 −19.8%，符号都反了。
    """
    groups = ["无 EP", "有 EP"]
    eager = np.array([29.29, 30.01])
    cg = np.array([201.04, 161.33])
    x = np.arange(2)
    w = 0.32

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    b1 = ax.bar(x - w / 2, eager, w, color=C_FIX, label="eager")
    b2 = ax.bar(x + w / 2, cg, w, color=C_CMP, label="cg（CUDA Graph 开）")

    for bars, vals in ((b1, eager), (b2, cg)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=10.5, fontweight="bold")

    # 两组各自的 EP 效果
    ax.annotate("", xy=(1 - w / 2, 30.01), xytext=(0 - w / 2, 29.29),
                arrowprops=dict(arrowstyle="<->", color="#64748b", lw=1.5))
    ax.text(0.66, 34.0, "EP 的效果：+2.5%\n（几乎无差别）",
            ha="center", va="bottom", fontsize=10, color="#475569",
            linespacing=1.4, bbox=dict(facecolor="white", edgecolor="none",
                                      alpha=0.9, pad=2.0))

    ax.annotate("", xy=(1 + w / 2, 161.33), xytext=(0 + w / 2, 201.04),
                arrowprops=dict(arrowstyle="<->", color=C_BAD, lw=1.8))
    ax.text(0.5 + w / 2 - 0.02, 178.0, "EP 的效果：−19.8%\n（明显变差）",
            ha="center", va="center", fontsize=10.5, color=C_BAD,
            fontweight="bold", linespacing=1.4)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=12)
    ax.set_ylabel("decode 吞吐（tok/s，越高越好）")
    ax.set_ylim(0, 245)
    ax.legend(loc="upper left", fontsize=10.5, framealpha=0.95)
    ax.set_title("「负交互」长什么样：EP 的效果在 eager 下是正、在 cg 下是负",
                 fontsize=12)
    fig.suptitle("Qwen1.5-MoE-A2.7B · 2×RTX 4090D · TP=2 · 单请求 greedy",
                 fontsize=10.5, y=1.03)
    fig.tight_layout()
    _save(fig, "cg_ep_interaction.png")


# =====================================================================
# 图 11：PP piecewise 的三段时序 —— 04 §3.6 / 06 §2.3
# =====================================================================
def fig_pp_piecewise() -> None:
    """泳道时序：图外 recv → 图内 forward → 图外 send，两个依赖点切出三段。

    这是「为什么 PP 下 CG 倍数从 9.1× 掉到 4.2×」的机制图。
    源码依据：vllm-v6/src/compute/model_runner.py:248-269（_run_decode）
              vllm-v6/src/distributed/communication_op.py:145-168（阻塞 P2P）
    数值口径：rental/实测结果汇总.md §三（单卡 9.1× vs PP=2 4.2×）
    """
    fig, ax = plt.subplots(figsize=(10.6, 4.4))

    # 三条泳道：卡 0（段 0）/ 段边界 / 卡 1（段 1）
    lanes = [2, 1, 0]
    lane_labels = ["卡 0（段 0，层 0–13）", "段边界（NCCL P2P）", "卡 1（段 1，层 14–27）"]
    ax.set_yticks(lanes)
    ax.set_yticklabels(lane_labels, fontsize=10.5)
    ax.set_ylim(-0.65, 2.65)
    ax.set_xlim(0, 10.4)

    def seg(x0, x1, y, text, color, txtcolor="white"):
        ax.broken_barh([(x0, x1 - x0)], (y - 0.22, 0.44),
                       facecolors=color, edgecolor="white", lw=1.5)
        ax.text((x0 + x1) / 2, y, text, ha="center", va="center",
                fontsize=10, color=txtcolor, fontweight="bold")

    # 卡 0：recv（图外）→ forward（图内）→ send（图外）
    seg(0.2, 1.5, 2, "recv 上一段", C_BAD)
    seg(1.6, 5.2, 2, "图内 forward（本段 14 层）", C_CMP)
    seg(5.3, 6.6, 2, "send", C_BAD)
    seg(6.8, 10.2, 2, "等下一段（lockstep，无重叠）", "#e2e8f0", "#64748b")

    # 段边界：只有两笔 P2P
    seg(1.5, 1.6, 1, "", "#94a3b8")
    seg(5.2, 5.3, 1, "", "#94a3b8")
    ax.text(3.4, 1.0, "IntermediateTensors(hidden_states, residual)　每步两笔",
            ha="center", va="center", fontsize=9.5, color="#64748b", style="italic")

    # 卡 1
    seg(0.2, 1.5, 0, "等 recv", "#e2e8f0", "#64748b")
    seg(1.6, 5.2, 0, "图内 forward（本段 14 层）", C_CMP)
    seg(5.3, 6.6, 0, "recv + 采样", C_BAD)
    seg(6.8, 10.2, 0, "下一轮", "#e2e8f0", "#64748b")

    # 两个依赖点
    for xv in (1.55, 5.25):
        ax.axvline(xv, color=C_BAD, ls="--", lw=1.6, alpha=0.75)
    ax.text(1.62, 2.42, "依赖点①", fontsize=9.5, color=C_BAD, fontweight="bold")
    ax.text(5.32, 2.42, "依赖点②", fontsize=9.5, color=C_BAD, fontweight="bold")

    ax.set_xticks([])
    ax.set_xlabel("时间 →")
    ax.set_title("PP 的 piecewise 结构：两个依赖点把流水切成三段，CG 只能吃中间那段",
                 fontsize=12)

    # 图例放到画布下方，避免压住泳道
    ax.legend(handles=[
        Patch(facecolor=C_CMP, label="图内（CUDA Graph 重放，提交开销被消掉）"),
        Patch(facecolor=C_BAD, label="图外（主机侧同步阻塞，CG 消不掉）"),
        Patch(facecolor="#e2e8f0", label="等待（lockstep 无流水线重叠）"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3,
        fontsize=9.5, framealpha=0.95)

    fig.suptitle("Qwen3-0.6B · 2×RTX 4090D 无 NVLink · PP=2 lockstep（非 1F1B）· "
                 "CG 倍数 9.1× → 4.2×",
                 fontsize=10.5, y=1.04)
    fig.tight_layout()
    _save(fig, "cg_pp_piecewise.png")


def main() -> int:
    print(f"输出目录：{OUT}")
    fig_step_breakdown()
    fig_launch_vs_kernel()
    fig_gain_vs_size()
    fig_ttft_vs_prompt()
    fig_compile_modes()
    fig_three_way()
    fig_tp_comm_breakdown()
    fig_kernel_copy()
    fig_spec_gain_by_text()
    fig_ep_interaction()
    fig_pp_piecewise()
    print("完成。请打开图片确认中文没有变成方框。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
