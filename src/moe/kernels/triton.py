# ============================================================================
# 【本文件是做什么的】Triton 版 grouped GEMM：把一层 MoE 的两次矩阵乘
# （w13 → Silu → w2）各用一次 kernel 启动算完所有专家。
# 重排 / SwiGLU / 求和三个辅助步骤复用同目录的 cuda.py。
#
# 【grouped GEMM 到底"grouped"在哪】
#     普通 GEMM：C = A @ B，全场只有一个 B。
#     这里 B 是三维的 [E, K, N]（每个专家一份权重）。kernel 里每个 block 先查
#     `expert_ids[pid_m]` 拿到自己负责哪个专家，再用 `expert * stride_be` 把
#     指针偏到那个专家的权重上 —— 于是同一次启动里，不同 block 用的是不同的 B。
#     这就是"按专家分组"的含义：分组信息不是数据，而是指针偏移。
#
# 【A 那边为什么要查表】token 被 align 重排过，行序变了；
#     `sorted_ids[pid_m * BLOCK_M + offs]` 把"第 m 行该取原始第几个 token"翻译回来（gather）。
#     所以两个操作数一个靠查表取行、一个靠偏移取专家，这是整个 kernel 的核心两行。
#
# 【两次 GEMM 的差别】
#     第一次（w13）：输入是 hidden，按 M*top_k 行算（每个 token 的每路路由各一行），
#                   row_div = top_k（行号除以 top_k 才是 token 号）；
#     第二次（w2） ：输入已经过 SwiGLU，行数变成 M，row_div = 1，
#                   并且要多乘一次路由权重（mul_w）再把 k 路相加 —— 加权就是在这里发生的。
#
# ⚠️ 参数里的 stride 顺序有坑：权重在显存里是 [E, N, K]（K 连续），
#   但传给 kernel 的顺序是 stride(0), stride(2), stride(1) —— 让 kernel 内部按 [E, K, N]
#   的语义寻址。这不是转置（没搬数据），只是换一种方式描述同一块内存。
# ============================================================================
"""Triton grouped GEMM for fused MoE (align / silu / sum via ``kernels.cuda``)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .cuda import moe_align_block_size, moe_sum, silu_and_mul

# workspace 池化：grouped GEMM 的两次输出各按形状缓存一份。
# 目的与 cuda.py 里那三组缓存一致 —— 让 CUDA Graph 捕获时每步的分配次数固定。
_WS: dict[tuple, torch.Tensor] = {}


def clear_state() -> None:
    _WS.clear()
    from .cuda import clear_state as clear_cuda_state

    clear_cuda_state()


# 命中条件：形状 / dtype / 设备三者都一致（键只记「哪一类」，不足以保证形状相同）。
def _scratch(key: tuple, *, shape, dtype, device) -> torch.Tensor:
    buf = _WS.get(key)
    if buf is None or buf.shape != shape or buf.dtype != dtype or buf.device != device:
        buf = torch.empty(shape, dtype=dtype, device=device)
        _WS[key] = buf
    return buf


# 【分块启发式】按"每个专家平均分到多少 token"和"总行数"挑 tile 尺寸。
# 这里的取舍依据是：decode（行数少）与 prefill（行数多）的瓶颈不一样 ——
# 行少时砖要小、K 维一次多读一些以便掩盖延迟；行多时反过来，砖要大以减少块间重复读权重。
def _blocks(M: int, E: int, top_k: int) -> tuple[int, int, int, int, int]:
    # tpe = tokens per expert 的粗略估计（均分假设）。
    # 它决定 BLOCK_M：行少意味着每个 block 里有效行少，砖太大会浪费线程。
    tpe = max((M * top_k) // max(E, 1), 1)
    # 16 / 32 / 64 / 128 四档 —— 行越少砖越小，减少「块内大半是填充行」的浪费。
    block_m = 16 if tpe <= 16 else 32 if tpe <= 40 else 64 if tpe <= 80 else 128
    # decode（M ≤ 64）时 BLOCK_N 取 64、BLOCK_K 取 128：
    # 行少 ⇒ N 方向不必一次吃满，把 K 维拉长能更好地掩盖访存延迟。
    # prefill 时反过来（N=128、K=64）：行多、算力受限，砖宽一点减少重复读权重。
    block_n = 64 if M <= 64 else 128
    block_k = 128 if M <= 64 else 64
    # 并行度也随规模调：M 大时用 8 个 warp（一个 block 里有更多并行工作）。
    warps = 4 if M <= 128 else 8
    # 流水线级数：M 很小时（≤32）用 4 级更好掩盖延迟；行多了 3 级就够，省 smem。
    stages = 4 if M <= 32 else 3
    return block_m, block_n, block_k, warps, stages


# 【kernel 本体】一个 program（block）负责一个 (BLOCK_M 行 × BLOCK_N 列) 的砖。
@triton.jit
# 传进来的东西分四类：
#     a_ptr / b_ptr / c_ptr      三块矩阵
#     sorted_ptr / expert_ptr / n_post_ptr   重排查表用的三个索引
#     topk_w_ptr                 路由权重（只有第二次 GEMM 会读）
#     其余是 stride 与编译期常量（constexpr 的那些在编译期就定下来，能参与优化）
def _gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sorted_ptr,
    expert_ptr,
    n_post_ptr,
    topk_w_ptr,
    num_valid,
    N,
    K,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    row_div: tl.constexpr,
    mul_w: tl.constexpr,
    naive: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 一维 grid 拆成二维逻辑坐标：行方向（token 块）× 列方向（N 块）。
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # pid_m 同时是三层含义：token 块号、expert_ids 的下标、sorted_ids 的偏移基准。
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    # 真实长度（不是缓冲容量）。容量是为了 CUDA Graph 形状固定，
    # 真实长度决定哪些块有活干。
    n_post = tl.load(n_post_ptr)
    # 上界之外的块直接退出：这些块本来就全是填充行，
    # 不做任何访存就返回，比让它们跑完再丢弃便宜得多。
    if pid_m * BLOCK_M >= n_post:
        return

    # 块内行号（0..BLOCK_M-1），用来算 sorted_ids 的偏移与掩码。
    offs = tl.arange(0, BLOCK_M)
    # 两条路径：naive = "不重排，一个 block 只管一行"。
    # 见 forward 里的触发条件（路由极稀疏时重排的收益还不如它自己的开销）。
    if naive:
        # naive 路径：只有第 0 个 lane 拿到真实行号 pid_m，其余 lane 全部设成 num_valid ⇒
        # 后面所有访存掩码都会把它们挡掉。所以这个 block 实际只算 1 行 ——
        # 用"块利用率"换"省掉一次 align kernel"。
        token = tl.where(offs == 0, pid_m, num_valid)
    else:
        # 常规路径：从 sorted_ids 取出这 BLOCK_M 行各自对应的原始 token 行号。
        # 注意数组长度按容量分配，超出真实长度的位置填的是越界下标，靠掩码兜住。
        token = tl.load(sorted_ptr + pid_m * BLOCK_M + offs)

    # 这个 block 属于哪个专家 —— grouped GEMM 的分组信息就从这一行来。
    expert = tl.load(expert_ptr + pid_m)
    # -1 是"这个块没有专家"（纯填充块，由 align 标出）：直接整块退出。
    # 上层（cuda.py 的 _align_torch）把超出真实长度的块标成 -1，正是为了这一行。
    if expert < 0:
        return

    # token → A 的输入行号。
    #     第一次 GEMM：A 是按 M*top_k 行存的（每路路由一行）⇒ 除以 top_k 才是 token 号；
    #     第二次 GEMM：A 已经变成 M 行 ⇒ 除以 1，即原值。
    # 把 token // row_div 统一成一种写法，两次调用共用同一个 kernel。
    # 掩码位置（token == num_valid 的填充行）统一指向第 0 行 —— 只为避开非法地址，
    # 算出来的值会被写入掩码丢掉。
    a_row = tl.where(token < num_valid, token // row_div, 0)
    # K 与 N 方向的偏移（K 方向分块循环使用，N 方向本块固定）。
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # A 的指针：行方向用"翻译回来的 token 行号"，列方向走 K。
    # 这就是 gather：A 的行是不连续的，靠 a_row 一行行指过去。
    a_ptrs = a_ptr + a_row[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = (
        b_ptr
        # B 的指针：先把基址偏到"本块负责的专家"，再按 K / N 索引。
        # 这一行就是 grouped GEMM 与普通 GEMM 的唯一区别 —— 不同块读不同的 B。
        + expert * stride_be
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )

    # 累加器用 fp32：Triton 的 tl.dot 在 fp32 里累加，避免长 K 维上的精度损失。
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # 沿 K 维分块循环（每块 BLOCK_K），是标准的分块 GEMM 结构。
    for _k in range(0, tl.cdiv(K, BLOCK_K)):
        k = _k * BLOCK_K
        a = tl.load(
            a_ptrs,
            # A 的掩码有两重：越界行（token ≥ num_valid，即填充行）与 K 维尾块。
            # other=0.0 让填充行贡献 0 —— 与"写入时会被丢弃"配套，中间结果不会被污染。
            mask=(token[:, None] < num_valid) & (offs_k[None, :] + k < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            # B 的掩码同理：K 维尾块与 N 维尾块（N 不整除 BLOCK_N 时最后一块会越界）。
            mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        # 核心矩阵乘：把 BLOCK_K 这一段累加进 acc。
        acc += tl.dot(a, b)
        # 指针推进而不是重新计算：下一轮 K 块直接用同一个指针加偏移。
        # 这也是 Triton 里能拿到不错性能的常见写法（配合 num_stages 做软件流水）。
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # 【只有第二次 GEMM 会进来】把每路的路由权重乘到结果上。
    # 第一次（算 gate/up）不乘：加权求和必须在 SwiGLU 之后做，
    # 否则相当于先加权再激活，与数学定义不符。
    # 权重按 token 取（同一个 token 的 k 路各取自己的权重）。
    if mul_w:
        w = tl.load(topk_w_ptr + token, mask=token < num_valid, other=0.0)
        # 逐行缩放：w 是 [BLOCK_M]，广播到 [BLOCK_M, BLOCK_N]。
        acc = acc * w[:, None]

    # 写回：行方向仍用 token（这就是"散写"—— 与读入的 gather 对称），
    # 掩码把填充行与 N 维尾块挡掉。
    tl.store(
        c_ptr + token[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(c_ptr.dtype.element_ty),
        mask=(token[:, None] < num_valid) & (offs_n[None, :] < N),
    )


# 【一次 GEMM 的薄封装】算 grid、传 stride、启动 kernel。
# 两次 GEMM 共用它，靠 row_div / mul_w / naive 三个开关区分。
def _gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    sorted_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_post: torch.Tensor,
    num_valid: int,
    row_div: int,
    block_m: int,
    block_n: int,
    block_k: int,
    warps: int,
    stages: int,
    grid_tokens: int,
    out_key: tuple,
    naive: bool,
    topk_w: torch.Tensor | None = None,
) -> torch.Tensor:
    # B 的形状约定是 [E, N, K]（最后一维是 K，与权重在显存里的布局一致）——
    # 所以这里只断言 K 维相等：Kb == K。
    _, K = A.shape
    _, N, Kb = B.shape
    assert Kb == K
    # 输出按「最大可能行数」num_valid 分配，实际用到的行由掩码决定。
    # 池化键里带 num_valid 与 N，是为了形状一致时复用同一块。
    C = _scratch(out_key, shape=(max(num_valid, 1), N), dtype=A.dtype, device=A.device)
    # 每步清零。掩码保证不会读到脏值，但清零让「池化复用」的结果可复现，
    # 也让 CG 重放时的输出与 eager 完全一致 —— 代价是每步一次全量写。
    C.zero_()
    if num_valid == 0 or grid_tokens == 0:
        # 返回池化缓冲：调用方拿到的是「本步的输出」，下一步同一形状会覆盖它。
        return C
    # naive 路径没有 sorted_ids（本来就不重排），但 kernel 需要一个指针对齐 ——
    # 直接拿 expert_ids 顶上：naive 下 kernel 不会去读 sorted_ptr。
    sorted_ids = expert_ids if sorted_ids is None else sorted_ids
    tw = topk_w if topk_w is not None else A
    # grid = 行方向的块数 × 列方向的块数。
    # 行方向按 grid_tokens 算：常规路径是重排后的容量，naive 路径是"每个真实行一块"。
    grid = (triton.cdiv(grid_tokens, block_m) * triton.cdiv(N, block_n),)
    _gemm_kernel[grid](
        A,
        B,
        C,
        sorted_ids,
        # stride 的传递顺序刻意是 (0, 2, 1)：权重在显存里是 [E, N, K]，
        # 而 kernel 内部按 [E, K, N] 的语义寻址 —— 这样 K 维就是连续的，
        # 分块循环里的 B 读取能连成一段（对访存友好），且不需要真的转置。
        expert_ids,
        num_post,
        tw,
        num_valid,
        N,
        K,
        # 三个编译期开关：row_div（行号换算）、mul_w（是否乘路由权重）、naive（是否重排）。
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        row_div=row_div,
        mul_w=topk_w is not None,
        naive=naive,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=warps,
        num_stages=stages,
    )
    return C


# ============================================================================
# 前向：align（或 naive）→ GEMM(w13) → SwiGLU → GEMM(w2, 乘权重) → 按 top_k 求和。
# ============================================================================
def forward(
    *,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
) -> torch.Tensor:
    # contiguous() 是为了让 kernel 能按 (stride_be, stride_bk, stride_bn) 三个 stride
    # 描述权重 —— 非连续张量也能传 stride，但连续化能让 K 维真的相邻，访存更整齐。
    # 若权重本来就是连续的，这一步是空操作（不复制）。
    w13 = w13_weight.contiguous()
    w2 = w2_weight.contiguous()
    M, top_k = hidden_states.shape[0], int(topk_ids.shape[1])
    # E = 本卡持有的专家数（EP 下是 num_experts / ep_size，不是全局专家数）。
    # num_valid = M * top_k：第一次 GEMM 的有效行数（每个 token 的每路路由各一行）。
    E, num_valid = w13.shape[0], M * top_k
    dev = hidden_states.device.index or 0
    # 分块尺寸随本轮形状变（decode / prefill 用不同的 tile）。
    # 注意它同时决定 BLOCK_M —— 而 BLOCK_M 又是 align 的分块粒度，两者必须一致，
    # 否则 expert_ids 的下标与 kernel 里 pid_m 的假设就对不上了。
    block_m, block_n, block_k, warps, stages = _blocks(M, E, top_k)

    # Decode: skip align when extremely sparse (same heuristic as vLLM).
    # 条件含义：M * top_k * 4 ≤ E，也就是"平均每个专家的 token 数不到 1/4"（4 来自 top_k 的放大）。
    # 这种极稀疏场景（典型是 decode 早期、token 比专家还少）里，
    # align 自己的开销 + 填充浪费会超过它带来的整齐性收益，于是干脆不重排。
    # 这是与官方 vLLM 同源的启发式。
    if M * top_k * 4 <= E:
        # naive 路径的"伪 sorted_ids"：直接拿原始 topk_ids 当分组信息。
        # 因为没有重排，行号就是原样，一个块负责一行（见 kernel 里 naive 那一行）。
        expert_ids = topk_ids.reshape(-1).to(torch.int32)
        # 行方向的块数 = 真实行数（每行一块），乘 block_m 只是让它与容量口径一致。
        grid_tokens = expert_ids.numel() * block_m
        num_post = torch.empty(1, dtype=torch.int32, device=topk_ids.device)
        # num_post 直接填成「全部都要算」 —— naive 下没有填充块需要跳过。
        num_post.fill_(grid_tokens)
        sorted_ids, naive = None, True
    else:
        # 常规路径：先按专家重排（这一步的容量上界、填充块 -1 标记都在 cuda.py 里）。
        # 重排后行数变成"容量"，真实长度在 num_post 里。
        sorted_ids, expert_ids, num_post = moe_align_block_size(topk_ids, block_m, E)
        grid_tokens, naive = sorted_ids.numel(), False

    # 第一次 GEMM：hidden @ w13ᵀ ⇒ gate/up 拼在一张 [M*top_k, 2I] 里。
    # row_div=top_k：因为 A 是以"每路路由一行"存的。
    gate_up = _gemm(
        hidden_states,
        w13,
        sorted_ids=sorted_ids,
        expert_ids=expert_ids,
        num_post=num_post,
        num_valid=num_valid,
        row_div=top_k,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        # 行号换算：行 i 属于第 i // top_k 个 token（同一 token 的 k 路是连续几行）。
        warps=warps,
        stages=stages,
        grid_tokens=grid_tokens,
        out_key=("gu", dev, num_valid, w13.shape[1], hidden_states.dtype),
        naive=naive,
    )
    # 第二次 GEMM：输入先过 SwiGLU（行数从 M*top_k 变成 M），再乘路由权重。
    # row_div=1：此时 A 的行号就是 token 号；
    # mul_w 打开：这是唯一做加权的地方，k 路的合并结果在最后一步 moe_sum 里完成。
    down = _gemm(
        # 融合 SwiGLU：up * silu(gate)，一步出结果（见 cuda.py 的同名函数）。
        silu_and_mul(gate_up),
        w2,
        sorted_ids=sorted_ids,
        expert_ids=expert_ids,
        num_post=num_post,
        num_valid=num_valid,
        row_div=1,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        # 第二次 GEMM 的输入已是 M 行，行号即 token 号。
        warps=warps,
        stages=stages,
        grid_tokens=grid_tokens,
        out_key=("dn", dev, num_valid, w2.shape[1], hidden_states.dtype),
        naive=naive,
        # 路由权重展平成一维，与第二次 GEMM 的行号一一对应。
        topk_w=topk_weights.reshape(-1).contiguous(),
    )
    # 最后按 top_k 把 k 路相加：[M*top_k, H] → [M, H]。
    return moe_sum(down, M, top_k)
