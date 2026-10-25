# ============================================================================
# 【本文件是做什么的】在**导入 FlashInfer 之前**，把它要编译的 GPU 架构列表钉死。
#
# 【为什么必须提前做】FlashInfer 的 JIT kernel 在**进程启动时**根据环境变量
# `FLASHINFER_CUDA_ARCH_LIST` 决定编译哪几档架构。默认行为是"按当前可见的 GPU"猜：
#     · 异构机器（例如 4090 + 4070 混插）上，它可能只编译其中一档，
#       另一张卡跑到对应 kernel 时就报 "no kernel image"；
#     · 用 CUDA_VISIBLE_DEVICES 限制可见卡时，它看到的是**被限制后**的设备，
#       与真正要用的卡可能对不上。
# 所以这里显式把"本次要用的那几张卡"的架构列出来，交给 FlashInfer 一次性编全。
#
# 【架构名不是简单的 sm_XX】FlashInfer 用的是带后缀的 CUDA 架构名：
#     a = accelerated 特性（需要对应 CUDA 版本支持）；
#     f = family（同族通吃，例如 12.0f 要求 nvcc ≥ 12.9）。
# 后缀选错会编译失败，选保守了会丢失加速特性 —— 下面的推导就是为这件事服务的。
#
# ⚠️ 必须在 import flashinfer 之前调用。已经 import 过再设环境变量是无效的，
#   所以 refresh_flashinfer_compilation_context() 才会去重建那个进程级全局对象。
# ============================================================================

from __future__ import annotations

import os
from collections.abc import Sequence


# 【compute capability → FlashInfer 架构名】
# 不同 major 的命名规则不一样，这里就是全部规则。
def format_flashinfer_arch(major: int, minor: int) -> str:
    # major 9（4090 / 4090D / H100 这一代）统一用 `9.xa`：加速特性可用且 nvcc 版本要求低。
    if major == 9:
        return f"{major}.{minor}a"
    # major 12 是最麻烦的一档：
    # `12.0f`（family 名）能覆盖同族所有小版本，但**要求 nvcc ≥ 12.9**；
    # 若 nvcc 只有 12.8，只能用 `12.0a`。
    # 关键提醒（作者原注释）：不要只看 torch.version.cuda ——
    # 那说的是 PyTorch 编译时的 CUDA 版本，而这里要的是**编译 kernel 用的 nvcc**，
    # 完全可能是一个更新/更旧的本地 toolkit（wheel 是 cu129、系统 nvcc 还是 12.8）。
    if major == 12:
        # SM120: ``12.0f`` needs *nvcc* ≥ 12.9. Do NOT trust torch.version.cuda
        # alone — wheels may be cu129 while system toolkit is still 12.8.
        import re
        import subprocess

        # 探测本地 nvcc 版本（与 moe/parallel.py 里那段同源：先 CUDA_HOME，再 PATH）。
        nvcc_ver: tuple[int, int] | None = None
        home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or ""
        candidates = []
        if home:
            candidates.append(os.path.join(home, "bin", "nvcc"))
        candidates.append("nvcc")
        for nvcc in candidates:
            try:
                out = subprocess.check_output(
                    [nvcc, "--version"],
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=5,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            m = re.search(r"release\s+(\d+)\.(\d+)", out)
            if m:
                nvcc_ver = (int(m.group(1)), int(m.group(2)))
                break
        # 分界线就在这里：nvcc ≥ 12.9 才能用 `f` 后缀（family），否则退回 `a` 后缀。
        # 返回值的差异不只是"名字不同"——它决定编译出来的 kernel 能不能利用加速特性。
        if nvcc_ver is not None and nvcc_ver >= (12, 9):
            return "12.0f" if minor == 0 else f"12.{minor}a"
        # nvcc 12.8 (or unknown): a-family works with CUDA toolkit 12.8.
        return "12.0a" if minor == 0 else f"12.{minor}a"
    # major ≥ 10（B200 / 新架构）同样带 `a`：这些架构的加速特性是默认开启的。
    if major >= 10:
        return f"{major}.{minor}a"
    # 更老的架构（major < 10，如 sm_86 的 3090）没有后缀 ——
    # 写成带后缀的名字反而会因为不识别而编译失败。
    return f"{major}.{minor}"


# 【把一组设备换算成架构列表】输入是本次要用的 GPU 编号（例如 [0, 1] 或 [1, 2, 4, 5]），
# 输出是空格分隔的架构串（例如 "8.9a" 或 "8.9a 12.0a"），直接写进环境变量。
# 注意输入是**设备编号**而不是"当前设备"—— 多卡时每张卡都可能是不同架构。
def flashinfer_arch_list_for_devices(device_ids: Sequence[int]) -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to resolve FlashInfer arch list")
    # 先确认序号没越界：device_ids 往往来自命令行，写错时给一句人话而不是让 torch 抛晦涩的错。
    n = torch.cuda.device_count()
    archs: list[str] = []
    seen: set[str] = set()
    # 逐卡取架构；同一架构只留一份（异构机器上很可能重复）。
    for d in device_ids:
        di = int(d)
        # 越界立刻报错，并带上「可见卡数量」方便对照 CUDA_VISIBLE_DEVICES 的写法。
        if di < 0 or di >= n:
            raise ValueError(
                f"device_ids contains invalid GPU index {di} (visible count={n})"
            )
        # 取该卡的 compute capability —— 这才是「要编哪档架构」的真实依据。
        major, minor = torch.cuda.get_device_capability(di)
        arch = format_flashinfer_arch(int(major), int(minor))
        # 去重靠 seen 集合而不是去重列表：保持**首次出现顺序**，
        # 这样输出的架构串在不同卡序下仍然可读、可比对。
        if arch not in seen:
            seen.add(arch)
            archs.append(arch)
    if not archs:
        raise ValueError("device_ids is empty; cannot set FlashInfer arch list")
    return " ".join(archs)


# 【重建 FlashInfer 的进程级编译上下文】环境变量改了，但 FlashInfer 在
# 首次 import 时已经把架构集合固化进一个全局对象了 —— 必须把它换掉才能生效。
# 整个函数对"没装 flashinfer"是安全的：import 失败就静默返回（本项目允许不用 FlashInfer）。
def refresh_flashinfer_compilation_context() -> None:
    """Rebuild FlashInfer's process-global arch set from the current env."""
    try:
        import flashinfer.jit.core as fi_core
        from flashinfer.compilation_context import CompilationContext
    except ImportError:
        return
    # FlashInfer 内部的全局编译上下文对象，替换即「重新读环境变量」。
    fi_core.current_compilation_context = CompilationContext()


# 【对外唯一入口】解析架构串 → 写环境变量 → 刷新上下文 → 返回最终使用的串。
# 返回字符串而不是 None：调用方（引擎启动日志）可以直接把它打进日志，
# 事后排查"到底按哪档架构编的"时不用猜。
def apply_flashinfer_arch_list(
    device_ids: Sequence[int],
    *,
    overwrite: bool = False,
) -> str:
    # overwrite=False 时**尊重外部已设的值**：运维可以在启动脚本里
    # 自己指定 FLASHINFER_CUDA_ARCH_LIST（例如为了复用编译缓存），这里就不去覆盖它。
    # overwrite=True 用于"以代码里的推导为准"。
    if overwrite or "FLASHINFER_CUDA_ARCH_LIST" not in os.environ:
        arch_list = flashinfer_arch_list_for_devices(device_ids)
        os.environ["FLASHINFER_CUDA_ARCH_LIST"] = arch_list
    else:
        arch_list = os.environ["FLASHINFER_CUDA_ARCH_LIST"]
    # 无论走哪条分支都要刷新：外部设的值同样要让它生效。
    refresh_flashinfer_compilation_context()
    return arch_list
