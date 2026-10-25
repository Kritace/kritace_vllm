# ============================================================================
# 【本文件是做什么的】从 logits 里取下一个 token：贪心 / 温度 / top-k / top-p，
# 以及为 torch.compile 准备的那条"纯温度"快路径。
#
# 【为什么用 Gumbel-max 而不是 multinomial】从概率分布里采样，torch 的
#     `torch.multinomial` 要跑一次排序/累积和（C++ 侧逐行循环，batch 大时是瓶颈，
#     而且形状/取值依赖随机数，不利于图捕获）。
#     Gumbel-max 把采样变成"给 logits 加噪 → 取 argmax"：
#         `argmax(logits / T + gumbel_noise)` 与按 softmax(logits/T) 采样同分布，
#     全部是逐元素运算，能融进同一次 kernel、也能被 CUDA Graph 捕获。
#
# 【两条路径的分工】
#     "只有温度"这一种采样 → 可以 compile 成一张固定图（Gumbel-max 全逐元素）；
#     一旦混入 top-k / top-p → 走通用路径（排序、阈值、掩码，形状与数据相关）。
#
# ⚠️ 贪心（temperature=0）与采样的分支必须在图外决定：
#   把数据相关的分支放进 CUDA Graph 会导致重放时走错分支。
# ============================================================================

from __future__ import annotations
import torch
from torch import nn


# Gumbel-max 技巧：给每个 logits 加一个 Gumbel 噪声再取 argmax，
# 数学上等价于按 softmax 概率采样，但省掉了完整的概率归一化。
#
# 这里刻意不加 @torch.compile，理由写在本函数上方：
#   连续批处理下 decode 的 batch size 随请求陆续完成而动态递减，
#    torch.compile 默认按形状特化 → 每个新 batch size 都要重编译一次，
#    秒级的编译开销被摊进稳态吞吐，反而比 eager 慢一个量级。
#    采样器每步只跑几个轻量 kernel，融合收益根本抵不上重编译的代价。
def _gumbel_sample(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    """温度采样：Gumbel-max，全程 GPU，无 host 分支。

    注意：本函数不能加 @torch.compile。连续批处理中 decode 的 batch size
    随请求陆续完成而动态递减，torch.compile 默认按形状特化，每个新 batch
    都会触发一次重编译，编译开销(秒级)被计入稳态吞吐，反而比 eager 慢一个
    量级。采样器每步只跑几个轻量 kernel，融合收益远小于重编译代价。
    """
    # 先 float() 再除：bf16/fp16 直接做除法会掉精度，采样结果的分布会偏。
    scaled = logits.float() / temperatures.unsqueeze(1)
    # 概率化（这一步之后才是有意义的概率分布）。
    probs = torch.softmax(scaled, dim=-1)
    gumbel = torch.empty_like(probs).exponential_().clamp_min_(1e-10)
    return probs.div_(gumbel).argmax(dim=-1)


# 批量采样器。构造它不占任何资源 —— 完全是函数式的。
class Sampler(nn.Module):
    """批量 token 采样器。

    采样策略开关（是否启用 top-k/top-p/min-p、greedy 行、top-k 上限）由调用方在
    CPU 侧打包时判定并通过标量参数传入，保证 forward 全程无 GPU→CPU 同步。
    """

    # 【参数分成两类，看那个 * 号】
    # 前的：每条请求各一份的数据张量
    # 后的（keyword-only）：整批共享的开关与上限，全部由 CPU 侧预先判定
    # 之所以把开关做成整批的布尔量而不是每条一个，是因为 kernel 里没法按行分支 ——
    # 一旦有请求需要 top-k，整批就走 top-k 路径（多余的计算由掩码挡掉）。
    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ps: torch.Tensor | None = None,
        top_ks: torch.Tensor | None = None,
        min_ps: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        *,
        all_greedy: bool = False,
        any_greedy: bool = False,
        use_top_k: bool = False,
        use_top_p: bool = False,
        use_min_p: bool = False,
        max_top_k: int = 0,
    ) -> torch.Tensor:
        """执行批量采样，返回 [batch_size] token ids。

        top_ks/top_ps/min_ps 已在 CPU 侧归一化（禁用项填默认值），此处不再判空。
        """
        # 快路径①：整批都贪心 → 直接 argmax，连温度都不用除。
        if all_greedy:
            return logits.argmax(dim=-1)

        # 混合情况：一部分贪心、一部分要采样。
        # 贪心行不能直接把温度设成 0（会除零），所以下面把它的温度换成 1，
        # 先算出一个"无意义但安全"的采样结果，最后再用 where 覆盖成 argmax。
        greedy_mask = temperatures <= 1e-5 if any_greedy else None
        safe_temp = (
            torch.where(greedy_mask, torch.ones_like(temperatures), temperatures)
            if any_greedy
            else temperatures
        )

        # 仅温度：编译 Gumbel-max；mixed greedy 用 clamp 后覆盖 argmax。
        # 快路径②：只要温度，不需要任何截断 → Gumbel-max 一步到位。
        if not use_top_k and not use_top_p and not use_min_p:
            # 最后统一处理混合批次里的贪心行：用 argmax 覆盖掉采样结果。
            if any_greedy:
                sample = _gumbel_sample(logits, safe_temp)
                return torch.where(greedy_mask, logits.argmax(dim=-1), sample)
            return _gumbel_sample(logits, safe_temp)

        # 需要截断的路径：先做温度缩放（这里开始才真正需要完整概率）。
        scaled = logits.float() / safe_temp.unsqueeze(1)

        # 路径③-a：top-k 优先。用它就必须先把候选压到 max_top_k 个，
        # 这样后面的排序/累积只在很小的范围里做。
        if use_top_k:
            # 取前 max_top_k 个（值是 logits，还没 softmax）。
            vals, idx = torch.topk(scaled, max_top_k, dim=-1)
            # 列下标，用来和每条请求自己的 top_k 比较。
            col = torch.arange(max_top_k, device=logits.device)
            # 把"超过本请求 top_k 的那些列"置成 -inf，而不是真的切片 ——
            # 因为每条请求的 top_k 不同，没法用一个统一的切片表达。
            vals = vals.masked_fill(
                col.unsqueeze(0) >= top_ks.unsqueeze(1), float("-inf")
            )
            sample = self._sample_ordered(
                vals,
                idx,
                top_ps if use_top_p else None,
                min_ps if use_min_p else None,
                generator,
            )
        # 路径③-b：没有 top-k 但有 top-p → 必须全量排序（nucleus 需要累积概率）。
        elif use_top_p:
            # 降序排序，得到有序值 + 原始下标（最后要靠下标换回真实 token id）。
            vals, idx = torch.sort(scaled, descending=True, dim=-1)
            sample = self._sample_ordered(
                vals, idx, top_ps, min_ps if use_min_p else None, generator
            )
        else:
            # 注意这里用 softmax 而不是 Gumbel：因为 min-p 需要真实概率才能和阈值比。
            probs = torch.softmax(scaled, dim=-1)
            # min-p 的阈值是"相对最大值"的比例，所以先取每行最大值。
            max_p = probs.max(dim=-1, keepdim=True).values
            # 低于 min_p × max 的全部置零。
            probs = probs.masked_fill(probs < min_ps.unsqueeze(1) * max_p, 0.0)
            # 在剩余概率上做多项式采样（概率为 0 的项自然不会被选中）。
            sample = torch.multinomial(probs, 1, generator=generator).squeeze(1)

        if any_greedy:
            sample = torch.where(greedy_mask, logits.argmax(dim=-1), sample)
        return sample

    # 在"已排序的候选空间"里应用 top-p / min-p，最后映射回原始 token id。
    @staticmethod
    def _sample_ordered(
        vals: torch.Tensor,
        idx: torch.Tensor,
        top_ps: torch.Tensor | None,
        min_ps: torch.Tensor | None,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """在降序排列后的候选空间内应用 top-p / min-p，并采样返回原始 token id。"""
        # 这里是对已排序的（前 max_top_k 个）值做 softmax ——
        # 与全量分布不同，但这是有意的：截断本身就是要改变分布。
        probs = torch.softmax(vals, dim=-1)

        # 用一个 bool 掩码记录"要删掉哪些位置"，而不是反复重建张量。
        if top_ps is not None or min_ps is not None:
            remove = torch.zeros_like(probs, dtype=torch.bool)

            # top-p 的判据基于累积概率：把累计到当前（不含自己）已经超过 p 的位置去掉。
            # 写的是 (cum - probs) 而不是 cum：这一步保证"刚好跨过阈值的那一个"被保留，
            # 否则累积和恰好等于 p 时会把边界项也删掉，与定义不符。
            if top_ps is not None:
                cum = probs.cumsum(dim=-1)
                remove |= (cum - probs) > top_ps.unsqueeze(1)

            # min-p：任何低于"最大值 × min_p"的候选都删掉。
            if min_ps is not None:
                remove |= probs < min_ps.unsqueeze(1) * probs[:, :1]

            # 强制保留第一个（概率最大的）候选：否则某个请求的参数过严时，
            # 可能把所有候选都删光 → multinomial 拿到全零分布 → 行为未定义。
            remove[:, 0] = False
            probs = probs.masked_fill(remove, 0.0)

        # 在掩码后的分布上采样，再用之前存的下标换回真实 token id。
        pos = torch.multinomial(probs, 1, generator=generator)
        return idx.gather(1, pos).squeeze(1)
