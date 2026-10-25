# ==============================================================================
# 投机解码的接受 / 拒绝判定。
#
# 目标：让"草稿 + 验证"的最终输出分布与"直接采样 target"严格一致。
# 依据：接受概率 min(1, p(x)/q(x))；被拒后从 (p-q) 的正部归一化得到的分布重采样。
#      两路合起来恰好还原 p(x) —— 推导见 docs/00-投机解码.md 第 4 节。
#
# 本模块只做概率层面的事，不接触 KV cache 与 GPU 张量布局，
# 因此可以在 CPU 上单测（tests/unit/test_spec_verify.py）。
# ==============================================================================
from __future__ import annotations

from typing import Sequence

import torch

__all__ = [
    "accept_probability",
    "residual_distribution",
    "greedy_accept_length",
    "sample_accept",
    "apply_top_k_top_p",
]

# 极小值：多处要拿它给分母兜底。
# 为什么需要一个正的下限而不是直接用 0：softmax 在 fp16/bf16 下可能给出精确的 0，
# 一旦做除法就会得到 inf / NaN —— 而这里恰好是"概率之间的比值"，最容易踩。
_TINY = 1e-12


# ============================================================================
# 【接受概率 min(1, p(x)/q(x))】这是整个投机解码"无损"的数学核心。
#
# 直觉：草稿模型 q 扔出一个 token 时，如果 target p 也认为它不错（p ≥ q），
#       那就照单全收；如果 target 觉得它没那么好（p < q），就以 p/q 的概率接受。
# 这样做之后，"先由 q 采样、再按此概率接受"这一路的产出分布**恰好等于 p** ——
# 被拒的那部分再由下面的 residual_distribution 补偿（两路合起来还原 p）。
# ============================================================================
def accept_probability(p_x: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
    """接受概率 min(1, p(x)/q(x))，逐元素计算。

    约定：q(x) == 0 时视为 1.0 —— 草稿根本采不到这个 token，
    不该因为它而拒绝（否则会改变分布）。

    参数形状任意，只要 p_x 与 q_x 可广播。
    """
    # q(x) == 0 时（草稿根本采不到这个 token）直接视为"接受概率 1"：
    # 不该因为一个草稿压根不会产出的 token 去拒绝 —— 那样会改变最终分布。
    # clamp_min 只是为了让除法有定义，真正的语义由 where 的第一个分支决定。
    ratio = torch.where(q_x > 0, p_x / q_x.clamp_min(_TINY), torch.ones_like(p_x))
    # 上限截到 1：p > q 时接受概率就是 100%（"接受"是个伯努利事件，
    # 不存在"接受率 120%"这种说法）。
    return ratio.clamp(max=1.0)


# ============================================================================
# 【被拒之后采什么】用 (p - q) 的**正部**归一化后的分布重采样。
#
# 为什么是 p - q 的正部：先看代数 ——
#     "未拒绝且命中 x"的概率 = q(x) · min(1, p(x)/q(x)) = min(q(x), p(x))
#     "拒绝"的总概率 = 1 - Σ min(p, q)
#     若被拒后用 (p - q)₊ 归一化重采样，命中 x 的概率再贡献 (p(x) - q(x))₊
#     两者相加正好等于 p(x) —— 这就是无损性的证明（docs/00-投机解码.md 第 4 节）。
# 负部要丢掉：在 q > p 的 token 上差值是负的，概率不能为负，而且那部分
# 已经由"以 min(1, p/q) 的概率接受"补偿过了。
# ============================================================================
def residual_distribution(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """拒绝后的修正分布：(p - q) 的正部，按行重新归一化。

    p、q 形状为 ``[..., vocab]``，返回同形状、最后一维求和为 1。

    ⚠️ 若某一行 (p-q).clamp_min(0) 全为 0（说明 p == q 处处成立，
    此时接受概率恒为 1，正常情况下不会走到这里），则回退到 p 本身，
    避免除零产生 NaN。
    """
    # 只保留「目标比草稿更偏好」的那部分差异；负的差值对概率没有意义。
    r = (p - q).clamp_min(0.0)
    # 逐行求和：每一行对应一个位置上的完整词表分布。
    total = r.sum(dim=-1, keepdim=True)
    # 正常情况除以总和即可；total == 0 说明 p 与 q 处处相等
    # （此时接受概率恒为 1，走不到这里），回退到 p 本身是为了**绝不出 NaN** ——
    # 投机解码是"跑得快"的优化，任何数值意外都会污染整条序列。
    return torch.where(total > 0, r / total.clamp_min(_TINY), p)


# ============================================================================
# 【贪心路径的接受判定】temperature == 0 时 min(1, p/q) 只能取 0 或 1：
# p、q 都退化成 one-hot，于是"按概率接受"等价于"token 是否相等"。
# 所以这里直接比 token id —— 不用构造概率分布，更快也更稳（没有浮点比较）。
# ============================================================================
def greedy_accept_length(
    draft_ids: Sequence[int], target_argmax: Sequence[int]
) -> int:
    """贪心解码（temperature == 0）下的接受长度。

    逐个比较草稿 token 与 target 在该位置的 argmax：第 i 个不同就停止，
    返回已接受的个数。全部相同则返回 ``len(draft_ids)``。

    这是第 4 节"贪心退化为 one-hot 比较"的直接实现 ——
    不必构造概率分布，比 token id 更快也更稳。
    """
    # n 既是「已接受的个数」，也是「第一个不同的位置」（两者数值相同）。
    n = 0
    for d, t in zip(draft_ids, target_argmax):
        # 第一个不同就停 —— 这是**自回归**决定的，不是优化：
        # 第 i 个草稿只有在前面 i 个都被接受的前提下才位于正确的位置上；
        # 拒绝点之后的所有草稿都作废（它们是在错误的前缀下产生的）。
        if d != t:
            break
        n += 1
    # 循环走完 = 全部接受，返回 K。
    return n


# ============================================================================
# 【采样路径的接受判定】逐个位置判定，**遇到第一个拒绝就返回**。
#
# 为什么不能"先全部判完再挑连续的接受段"：每个位置用的是**该位置**的
# (p, q) 对，而位置 i 的对只有在前面都被接受时才有意义（自回归）。
# 所以只能从头逐个判、遇到拒绝立刻停下。
# ============================================================================
def sample_accept(
    draft_ids: Sequence[int],
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[int, int | None]:
    """采样解码下的接受判定。

    参数
    ----
    draft_ids    : 草稿产出的 token，长度 K
    draft_probs  : 草稿模型在这 K 个位置的分布，形状 ``[K, vocab]``
    target_probs : target 模型在 K+1 个位置的分布，形状 ``[K+1, vocab]``

    返回
    ----
    ``(接受个数 j, 额外 token)``：

    - ``j < K``：第 j 个草稿被拒，额外 token 是**从修正分布重采样**的结果
    - ``j == K``：全部接受，额外 token 是**从 target 第 K 个位置**采样的结果
      （标准做法：全接受时白送一个 token，这是"1 + α + α² + …"里那个额外的 1）

    注意 draft_probs 与 target_probs 必须是**已应用温度**的概率，
    而不是原始 logits —— 比较在不同温度下的 logits 是没有意义的。
    """
    k = len(draft_ids)
    if k == 0:
        raise ValueError("draft_ids 不能为空")

    # 每个位置独立做一次伯努利判定：第 j 个位置用 (p_j, q_j) 这一对分布。
    for j in range(k):
        # 草稿在该位置给出的 token —— 注意 p、q 都取**这同一个 token** 的概率。
        tok = int(draft_ids[j])
        p_x = target_probs[j, tok]
        q_x = draft_probs[j, tok]
        # 均匀随机数 r ∈ [0, 1)，与接受概率比较即等价于"以该概率接受"。
        # 传同一个 generator 是为了让整条链路可复现（否则单测没法稳定断言）。
        r = torch.rand((), generator=generator, device=p_x.device)
        # 严格大于：r 取到 [0,1) 的闭开区间，所以 p_accept == 1 时永远不会拒绝；
        # p_accept == 0 时（p=0）几乎必然拒绝。
        if r > accept_probability(p_x, q_x):
            # 第 j 个被拒 -> 从修正分布重采样
            # 被拒：从修正分布重采样一个 token 顶上 —— 这一步是"无损"的另一半，
            # 少了它就退化成"草稿 + target 各出一半"的有偏解码。
            resid = residual_distribution(target_probs[j], draft_probs[j])
            # multinomial 只采 1 个；generator 同样要为可复现性显式传入。
            extra = int(torch.multinomial(resid, 1, generator=generator).item())
            # 返回 (接受的个数 j, 修正 token)：调用方拿 j 决定 KV 要保留几个位置。
            return j, extra

    # 全部接受，从 target 的最后一个位置采样
    # 全部接受：还要额外从 target 的**第 K 个位置**采一个 token。
    # 这就是投机解码"多产出一个"的来源 —— 验证时多算的那个位置不是浪费。
    extra = int(torch.multinomial(target_probs[k], 1, generator=generator).item())
    return k, extra


# ============================================================================
# 【截断：为什么 draft 与 target 必须用同一套 top-k / top-p / min-p】
# 因为真正被采样的分布是**截断并重新归一化之后**的分布，而不是原始 softmax。
# 若两边用不同的截断，p/q 就不在同一尺度上，min(1, p/q) 的推导前提不再成立 ——
# 表面上还在跑，实际已经不再是"无损"了。
# ============================================================================
def apply_top_k_top_p(
    probs: torch.Tensor,
    *,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
) -> torch.Tensor:
    """对概率做 top-k / top-p / min-p 截断并重新归一化。

    投机解码必须对 draft 与 target **使用同一套截断**，否则两者的
    分布形状不一致，接受概率会失去意义（截断后的分布才是真正被采样的分布）。
    """
    # clone 之后再原地改：不修改调用方传进来的张量。
    out = probs.clone()
    if top_k is not None and top_k > 0:
        k = min(top_k, out.shape[-1])
        # top-k：取第 k 大的概率当阈值，严格小于它的全部置 0。
        # 用 topk(...).values[..., -1] 直接拿第 k 大，不必排序整个词表。
        kth = out.topk(k, dim=-1).values[..., -1, None]
        out = out.masked_fill(out < kth, 0.0)
    if top_p is not None:
        sorted_p, idx = out.sort(dim=-1, descending=True)
        csum = sorted_p.cumsum(dim=-1)
        # 保留累积概率首次超过 top_p 的那个 token 本身
        # 保留"累计概率**首次超过** top_p 的那个 token 本身"：
        # csum 是含当前项的累计和，csum - sorted_p 是"不含当前项"的前缀和，
        # 所以 csum - sorted_p < top_p 正好表达"加我这一项之前还没超过阈值"。
        keep = csum - sorted_p < top_p
        mask = torch.zeros_like(out, dtype=torch.bool)
        # scatter 把 mask 还原到**原始词表顺序**：上面 sort 打乱了位置，
        # 直接在排序后的张量上做掩码会错位。
        mask.scatter_(-1, idx, keep)
        out = out.masked_fill(~mask, 0.0)
    if min_p is not None and min_p > 0:
        # min-p：阈值是"最大概率 × min_p"——相对阈值，
        # 比固定 top-k 更能自适应分布的尖锐程度（分布越平，砍掉得越少）。
        thresh = out.max(dim=-1, keepdim=True).values * min_p
        out = out.masked_fill(out < thresh, 0.0)
    # 截断之后各行之和不再为 1，必须重新归一化（这也是「被采样的分布」的定义）。
    total = out.sum(dim=-1, keepdim=True)
    return out / total.clamp_min(_TINY)
