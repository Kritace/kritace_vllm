# ============================================================================
# 【本文件是做什么的】fused MoE 的**两个后端**（FlashInfer cutlass / Triton）与切换点。
# 上面 block.py 只负责"算哪几个专家、权重怎么排"，真正的 kernel 在这里选。
#
# 【为什么要缓存那三块临时张量】每一层、每一步都要调一次这个函数。
# 如果每次都新建 ids / scales / out：
#     ① 每步多出若干次显存分配（小张量也不例外，allocator 要走一遍）；
#     ② **CUDA Graph 捕获时张量地址必须稳定** —— 重放时要写回同一块内存，
#        每次都换新张量的话图就抓不住。缓存让地址在多次调用之间保持不变。
#
# 【为什么缓存的 key 里要带设备号】一个进程里可能有多张卡（TP worker），
# 不同卡的临时张量绝不能共用 —— shape 一样也会互相踩。
# key = (设备号, token 数, hidden, dtype)，形状/设备一变就重新分配。
#
# ⚠️ 注意 token 数在 key 里：eager 下每步的 t 都可能不同，于是每步都会新建一块
#   （旧块被替换、交给 allocator 回收）—— 缓存真正生效的场景是"形状稳定"，尤其是 CUDA Graph。
# ============================================================================
"""Fused MoE forward: FlashInfer cutlass (SM89+) or Triton GEMM."""

from __future__ import annotations

import torch

from .kernels import clear_cuda_state
from .kernels import triton as triton_moe

# 三个模块级缓存，分别对应融合核要的三样东西：
#     _IDS    —— top-k 的专家号（int32）
#     _SCALES —— top-k 的路由权重（float32，融合核要求 fp32）
#     _OUT    —— 输出缓冲
# 做成模块级而不是挂在某个类上：内核调用是"按层"发生的，缓存要跨层复用。
_IDS: dict[tuple, torch.Tensor] = {}
_SCALES: dict[tuple, torch.Tensor] = {}
_OUT: dict[tuple, torch.Tensor] = {}


# 清空全部临时状态。两处会调用它：
#     ① 单测之间（避免上一条用例的张量残留影响下一条）；
#     ② CUDA Graph 重新捕获前（图里的地址要重新绑定）。
# 同时把 Triton / CUDA 侧的缓存一起清掉 —— 它们各有自己的 workspace 池。
def clear_fused_state() -> None:
    _IDS.clear()
    _SCALES.clear()
    _OUT.clear()
    triton_moe.clear_state()
    clear_cuda_state()


# 【取/建临时张量】命中条件比"key 存在"严格：**形状、dtype、设备三者都要一致**。
# 只比 key 是不够的 —— key 只记录了"哪一类"，shape 变了就得重建。
def _scratch(cache, key, *, shape, dtype, device):
    buf = cache.get(key)
    # 四元组任一不符就重新分配并顶掉旧的。
    # 顶掉旧张量意味着这份缓存**不是池**：同一时刻每个 key 只保留最近一次用的那块。
    # 想同时服务多个不同形状的调用（例如 CG 捕获多个 batch 档），靠的是 key 不同、
    # 因此是不同的缓存项，而不是同一项里放多块。
    if buf is None or buf.shape != shape or buf.dtype != dtype or buf.device != device:
        buf = torch.empty(shape, dtype=dtype, device=device)
        cache[key] = buf
    return buf


# 【cutlass 后端：FlashInfer 的离线融合核】
# 它一次做完三件事：按专家号把 token 分组、做两段 GEMM（w13 → SwiGLU → w2）、
# 再按路由权重加权求和 —— 所以外面不需要再调激活函数。
# import 放在函数里：没装 flashinfer 的环境不该因为 import 这个模块就失败，
# 后端是运行时探测选定的（见 parallel.py）。
def _cutlass(
    *,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
) -> torch.Tensor:
    from flashinfer import cutlass_fused_moe
    from flashinfer.tllm_enums import ActivationType

    # t = 本轮 token 数，k = 每个 token 激活的专家数（top_k），h = hidden。
    # k 是融合核要知道的"每个 token 要算几路"。
    t, k = hidden_states.shape[0], topk_ids.shape[1]
    h = hidden_states.shape[1]
    dev = hidden_states.device
    # 缓存的设备号。dev.index 为 None 表示"当前设备"，此时用 0 兜底。
    # 多卡时不同卡的缓存必须分开（见文件头），所以才需要这个整数进 key。
    di = dev.index or 0
    # out / ids / scales 三块都从缓存里取（形状一致的复用，不一致的重建）。
    # ids 与 scales 是**输入**的搬运目标：融合核要求 int32 / float32，
    # 而 topk_ids 是 int64（查表用）、topk_weights 是 bf16，所以要各转一次。
    out = _scratch(
        _OUT, (di, t, h, hidden_states.dtype), shape=(t, h), dtype=hidden_states.dtype, device=dev
    )
    ids = _scratch(_IDS, (di, t, k), shape=(t, k), dtype=torch.int32, device=dev)
    scales = _scratch(
        _SCALES, (di, t, k), shape=(t, k), dtype=torch.float32, device=dev
    )
    # 用 copy_ 而不是直接传 topk_ids：融合核要求 int32，
    # 直接 .to(torch.int32) 会新建张量（地址不稳定，CG 下不能用），
    # 所以往缓存块里拷。
    ids.copy_(topk_ids.to(torch.int32))
    # 路由权重也转 fp32：融合核的 scales 参数就是这个精度。
    scales.copy_(topk_weights.float())
    # 融合核调用。参数顺序：hidden → 专家号 → 权重 → w13 → w2 → 输出 dtype，
    # 后面几个关键字参数里最值得注意的是 ep_size=1（见下）。
    result = cutlass_fused_moe(
        hidden_states,
        ids,
        scales,
        w13_weight,
        w2_weight,
        # 输出 dtype 跟输入走（bf16 模型就是 bf16），不在这里做类型转换。
        hidden_states.dtype,
        quant_scales=None,
        output=out,
        # 【ep_size 恒为 1、ep_rank 恒为 0】因为 v5 不搬 token：
        # 每张卡拿到的都是完整的 token 批，融合核只管"按 expert_map 算本卡的专家"，
        # 跨卡求和由 block.py 出口的那次 all-reduce 负责。
        # 所以这个参数虽然存在（官方走 DeepEP 时要用），本项目始终传 1。
        ep_size=1,
        ep_rank=0,
        # 激活类型在核内完成（SwiGLU = silu(gate) * up），不需要外面的 SiluAndMul。
        activation_type=ActivationType.Swiglu,
    )
    # FlashInfer 各版本返回值不统一：有的直接返回张量，有的包成 list / tuple。
    # 这里两种都接受 —— 否则换一版 flashinfer 就会在"取结果"这一行炸掉。
    return result[0] if isinstance(result, (list, tuple)) else result


# ============================================================================
# 统一入口：按 backend 分发。两个后端的签名完全一致，调用方不需要知道用的是哪个。
# ============================================================================
def fused_moe_forward(
    *,
    backend: str,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
) -> torch.Tensor:
    # cutlass：走 FlashInfer 的离线融合核（性能通常更好，但有架构 / nvcc 前提）。
    # backend 的取值在 parallel.py 里已经校验过，到这里只可能是这两个之一。
    if backend == "cutlass":
        return _cutlass(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w13_weight=w13_weight,
            w2_weight=w2_weight,
        )
    if backend == "triton":
        return triton_moe.forward(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w13_weight=w13_weight,
            w2_weight=w2_weight,
        )
    # 兜底：不该出现。真出现了说明有人绕过 resolve_moe_backend 直接传了字符串，
    # 这里宁可抛错也不默认回退 —— 静默回退会让性能对比的公允性无从谈起。
    raise ValueError(f"unknown MoE backend: {backend!r}")
