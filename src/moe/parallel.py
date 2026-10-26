# ============================================================================
# 【本文件是做什么的】MoE 的并行布局（EP）与 fused MoE 后端的选择点。
#
# 【为什么 MoE 的切法和稠密层不一样】
#     稠密层用 TP：把一个矩阵切开，每卡算一片，最后 all-reduce。
#     MoE 层不该这么切 —— 专家之间互相独立，谁跟谁都不用相加。
#     以 tests/gpu/test_moe_kernel.py 那个真实用例为例（8 个专家、ep=2）：
#         · TP 切法：每卡都要算全部 8 个专家，只是每个专家的中间维切成一半
#           ⇒ 一半算力是白算的（结果还得 all-reduce 才凑得回一个专家）；
#         · EP 切法：卡 0 只放专家 0–3 的完整权重，卡 1 只放专家 4–7
#           ⇒ 显存按专家个数除。Qwen3-MoE 有 128 个专家，就是靠这个才装得下。
#
# 【EP 的通信是什么】all-reduce —— 和稠密 TP 同一个原语、同一条 hidden 维，
#     因为每张卡拿到的都是完整 [T, H]，单卡只能算出局部和（只含自己那几个专家的贡献），
#     出口一次 all-reduce 才是完整结果。
#     DeepEP / PPLX 那种"用 all-to-all 把 token 搬到专家所在卡"的做法本项目没有做
#     （token 留在本地、不搬，见 docs/04-MoE与并行.md）。
#
# 【共享专家带来的顺序约束】共享专家在每卡都是完整一份，
#     所以必须先 all-reduce、再相加；顺序写反同一项会被加 ep 次（顺序写在 block.py）。
#
# 【为什么后端要"探测"而不是写死】两个 fused 后端各有硬前提：
#     cutlass（FlashInfer）—— 只编了特定 SM 的 kernel，新架构还要求 nvcc ≥ 12.8；
#     triton              —— 自带 JIT，装得上就能跑。
#     SM 89 + CUDA 13 上两条路都能走，所以写成"自动探测 + 可显式指定"。
#
# ⚠️ 没有 CPU 回退路径：fused MoE 就是 GPU kernel，CPU 上直接不可用。
# ============================================================================
"""TP / EP layout and fused MoE backend selection."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from typing import Literal

# 两个 Literal 只服务于类型检查器；运行时它们仍是普通字符串，写错名字不会被拦住 ——
# 真正的校验在下面 resolve_moe_backend / from_engine 里手写。
MoeBackend = Literal["cutlass", "triton"]
MoeBackendPrefer = Literal["auto", "cutlass", "triton"]

# 【哪些 SM 才有 FlashInfer 的 cutlass 版 fused MoE】
#     89  = RTX 4090 / 4090D 这一代（项目实测用的两张 4090D）
#     90  = A100 / H100
#     100 = B200；120 / 121 = 新架构一代
# 不在名单里的架构（例如 SM 86 的 RTX 3090）只能用 Triton 后端，没有例外。
_CUTLASS_SMS = frozenset({89, 90, 100, 103, 110, 120, 121})


# 【compute capability → SM 编号】把 (major, minor) 折成 8*10+9 = 89 这种整数：
# 判断后端可用性时只需要"查集合 + 比大小"，比元组好写。
# torch 的 import 放在函数里，是为了这个模块在没有 GPU 的环境也能被 import（测试会用到）。
def _sm_tag(device: int | None = None) -> int | None:
    import torch

    if not torch.cuda.is_available():
        return None
    if device is None:
        major, minor = torch.cuda.get_device_capability()
    else:
        major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor


# 【为什么要探测 nvcc 版本】FlashInfer 的 cutlass kernel 要现场编译（JIT），
# 新架构（SM ≥ 120）需要 nvcc ≥ 12.8 才认识那些 SM 编号。
# 版本不够时是"编译失败"而不是"跑得慢" —— 与其等到跑起来才炸，不如在选后端时就把它排除。
def _nvcc_version() -> tuple[int, int] | None:
    # 候选顺序：先看 CUDA_HOME / CUDA_PATH（虚拟环境里最准），再退回 PATH 里的 nvcc。
    candidates: list[str] = []
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if home:
        candidates.append(os.path.join(home, "bin", "nvcc"))
    candidates.append("nvcc")
    # 拿不到版本号（没装 nvcc / 命令超时）就当不可用，返回 None，让调用方去退 Triton。
    for nvcc in candidates:
        try:
            out = subprocess.check_output(
                [nvcc, "--version"], stderr=subprocess.STDOUT, text=True, timeout=5
            )
        except (OSError, subprocess.SubprocessError):
            continue
        m = re.search(r"release\s+(\d+)\.(\d+)", out)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


# 【cutlass 可用的三个前提】架构在名单里 + flashinfer 装上了 + 新架构时 nvcc 够新。
# 三者缺一就返回 False —— 注意这里不抛异常，"能不能用"是查询，"用不用"由 resolve 决定。
# SM 89 在第 2 行就通过；换成 3090（SM 86）会直接返回 False。
def cutlass_moe_available(*, device: int | None = None) -> bool:
    sm = _sm_tag(device)
    if sm is None or sm not in _CUTLASS_SMS:
        return False
    try:
        from flashinfer import cutlass_fused_moe  # noqa: F401
    except ImportError:
        return False
    # 只有新架构才卡 nvcc：老架构的 kernel 早就编进 wheel 了，不需要现场编译。
    if sm >= 120:
        ver = _nvcc_version()
        if ver is None or ver < (12, 8):
            return False
    return True


# Triton 没有架构白名单 —— 它自带 JIT 编译器，任何 SM 都能现场生成 kernel。
# 所以这条只检查包在不在。
def triton_moe_available() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


# 【后端选择顺序】三层：
#     显式指定 → 不可用就报错（不静默降级）；
#     auto     → cutlass 优先（通例更快），退 Triton；
#     都没有   → 报错，并把 sm 与 nvcc 版本一起打出来。
# 为什么显式指定时不允许"悄悄换后端"：那样你会以为自己在跑 cutlass、其实跑的是 Triton，
# 拿这个数去做性能对比就是错的 —— 宁可启动就失败。
def resolve_moe_backend(
    prefer: MoeBackendPrefer = "auto",
    *,
    device: int | None = None,
) -> MoeBackend:
    if prefer == "cutlass":
        if not cutlass_moe_available(device=device):
            # 两个都没有：错误信息里带上 sm 与 nvcc —— 换机器时最常见的两个原因就是它们。
            raise RuntimeError(
                "moe backend=cutlass requested but unavailable "
                f"(sm={_sm_tag(device)} nvcc={_nvcc_version()})"
            )
        return "cutlass"
    if prefer == "triton":
        if not triton_moe_available():
            raise RuntimeError("moe backend=triton requested but triton is not installed")
        return "triton"
    # auto：先试 cutlass。
    if cutlass_moe_available(device=device):
        return "cutlass"
    # cutlass 不可用再试 Triton。
    if triton_moe_available():
        return "triton"
    raise RuntimeError(
        "No MoE backend available: need FlashInfer cutlass (SM89+) or Triton. "
        f"sm={_sm_tag(device)} nvcc={_nvcc_version()}"
    )


# 【MoE 的并行布局】frozen dataclass：构造后不可改，避免运行中被谁改掉一半。
# 关键点：tp 与 ep 互斥 —— use_ep=True 时 MoE 的 tp 被固定成 1（见 from_engine）。
@dataclass(frozen=True)
class MoEParallelConfig:
    """``enable_ep``: collapse MoE TP; shard experts across the old TP ranks."""

    # tp_size / tp_rank：MoE 这一层的张量并行度与自己的位置。
    # 开 EP 后恒为 1 / 0 —— 因为权重已经按专家切开，不再切单个专家的中间维。
    tp_size: int
    tp_rank: int
    # ep_size / ep_rank：专家并行的规模与自己负责的专家段。不开 EP 时恒为 1 / 0。
    ep_size: int
    ep_rank: int
    # use_ep：唯一决定 MoE 走哪条切法的开关；Attention 的 TP 不受它影响。
    use_ep: bool
    # fused_backend：fused MoE kernel 走 cutlass 还是 triton。
    fused_backend: MoeBackend

    @classmethod
    # 【唯一入口】从引擎的 TP 配置推出 MoE 的并行布局。
    # 调用方只给 tp_size / tp_rank / enable_ep，不用自己算 ep ——
    # ep 是"从 tp 推导出来"的，不该由调用方各自推一遍。
    def from_engine(
        cls,
        *,
        tp_size: int,
        tp_rank: int,
        enable_ep: bool,
        fused_backend: MoeBackend,
    ) -> "MoEParallelConfig":
        if tp_size < 1:
            raise ValueError("tp_size must be >= 1")
        if not 0 <= tp_rank < tp_size:
            raise ValueError(f"tp_rank={tp_rank} out of range for tp_size={tp_size}")
        if fused_backend not in ("cutlass", "triton"):
            raise ValueError(f"fused_backend must be cutlass|triton, got {fused_backend!r}")
        # 【核心一步：EP 把 MoE 的 TP 塌缩掉】
        # 条件 = 开了 EP 且确实不止一张卡。做的事是：
        #     tp 归 1、ep 取原来的 tp、ep_rank 沿用 tp_rank
        # 含义是「TP 组本身不变，但在这组 rank 上把 MoE 层从『切中间维』改成『切专家』」：
        # 第 i 个 TP rank 负责第 i 段专家（8 个专家、tp=2 → 卡 0 拿 0–3、卡 1 拿 4–7）。
        # 这样 Attention 仍在原来的 TP 组里做 all-reduce，MoE 复用同一个组，
        # 不需要新建 NCCL 组、world_size 也不变。
        if enable_ep and tp_size > 1:
            # 不满足条件（没开 EP，或只有一张卡）→ 退回普通 TP 布局：
            # ep 恒为 1，MoE 与稠密层一样按中间维切。
            return cls(
                tp_size=1,
                tp_rank=0,
                ep_size=tp_size,
                ep_rank=tp_rank,
                use_ep=True,
                fused_backend=fused_backend,
            )
        return cls(
            tp_size=tp_size,
            tp_rank=tp_rank,
            ep_size=1,
            ep_rank=0,
            use_ep=False,
            fused_backend=fused_backend,
        )

    @classmethod
    # 单卡 / 单元测试用的极简构造：tp=1、不开 EP。
    def single(cls, fused_backend: MoeBackend = "triton") -> "MoEParallelConfig":
        return cls.from_engine(
            tp_size=1, tp_rank=0, enable_ep=False, fused_backend=fused_backend
        )
