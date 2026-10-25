# ============================================================================
# 【本文件是做什么的】一层稀疏 MoE 的完整前向：
#     门控打分 → top-k 路由 → fused 专家计算 → （可选）共享专家 → 跨卡归约。
#
# 【执行顺序是这段代码最重要的知识点】all-reduce 与"相加共享专家"的先后，
# 在两种并行方式下**必须相反**：
#     EP：先 all-reduce，再加共享专家
#     TP：先加共享专家，再 all-reduce
# 判据只有一句：**这个量在每张卡上是"完整的"还是"部分和"**。
#     · EP 下共享专家每卡都有一份**完整**权重（重复持有）⇒ 先加的话，
#       这一项会被 all-reduce 求和 ep 次 ⇒ 必须把 AR 提前；
#     · TP 下共享专家自己也被切开，每卡算出来的同样是部分和 ⇒ 只能先加、再一次性 AR。
# 顺序写反不会抛错，只会得到"数值偏大 ep 倍"的静默错误 —— 所以这里注释写得最重。
#
# 【门控为什么每卡都算一遍】gate 只是一个 [H, E] 的小线性层，权重很小，
# 与其通信同步，不如每卡自己算（ReplicatedLinear）—— 反正激活本来就每卡一份。
# 前提是**各卡的 top-k 结果必须完全一致**：否则同一个 token 在不同卡上会去找不同的专家，
# 出口 all-reduce 把两边的贡献拼起来就是错的。这一点靠"权重复制 + 输入相同"保证，
# 不靠通信保证 —— 这也是 gate 不能改成切片线性的原因。
#
# ⚠️ 没有 CPU 路径：hidden_states 必须是 CUDA 张量（fused kernel 的前提）。
# ============================================================================
"""Sparse MoE block: gate → route → fused experts → optional shared + AR."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..compute.layers import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    SiluAndMul,
)
from ..distributed import get_tp_group, tensor_model_parallel_all_reduce
from .experts import Experts
from .parallel import MoEParallelConfig


# 【路由：打分 → 取 top-k → 可选归一化】三步固定在 CPU 上做不到，但逻辑本身很简单。
# 返回顺序是 (weights, ids) —— 注意权重在前，别接反。
def route_experts(
    router_logits: torch.Tensor,
    top_k: int,
    *,
    norm_topk_prob: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 门控用 **float32** 算 softmax：分数之间要做"谁更大"的排序，
    # bf16 下两个相近分数可能被舍入成同一个值，top-k 的顺序就会抖。
    # 只在打分这一步升精度，后面的乘加仍回到原 dtype，代价可以忽略。
    scores = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    # top-k：返回每一行最大的 k 个**分数**（weights）与它们的**专家号**（ids）。
    # k 就是 num_experts_per_tok（Qwen1.5-MoE-A2.7B 是 4）。
    topk_weights, topk_ids = torch.topk(scores, top_k, dim=-1)
    # 归一化开关：不同模型对"top-k 权重是否要重新归一化到和为 1"的约定不同，
    # 由 config 给（Qwen3-MoE 是 True）。归一化与否会改变输出尺度，必须跟模型配置一致。
    if norm_topk_prob:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    # 权重转回 logits 的 dtype（bf16/fp16）参与后续乘加；ids 保持整型不动。
    return topk_weights.to(router_logits.dtype), topk_ids


# ============================================================================
# 一层 MoE：门控 + 专家 + （可选）共享专家。
# ============================================================================
class SparseMoeBlock(nn.Module):
    def __init__(self, config, parallel: MoEParallelConfig) -> None:
        super().__init__()
        self.parallel = parallel
        hidden = config.hidden_size
        # 全局专家数（配置里的那个数）：传给 Experts 建 expert_map。
        self.num_experts = int(config.num_experts)
        # 每个 token 激活几个专家（num_experts_per_tok）。
        self.top_k = int(config.num_experts_per_tok)
        self.norm_topk_prob = bool(getattr(config, "norm_topk_prob", False))
        # 是否需要跨卡归约要等到 forward 末尾才知道（取决于并行方式是 EP 还是 TP，而不是有几张卡）。
        self.use_ep = parallel.use_ep

        # 门控是**复制**的（每卡一份完整权重）：[hidden, num_experts] 很小，
        # 复制比切片更划算，还能保证各卡 top-k 结果一致（见文件头）。
        self.gate = ReplicatedLinear(hidden, self.num_experts, bias=False)
        # 专家权重：EP 下只装本卡那一段，TP 下装全部专家的中间维分片。
        self.experts = Experts(
            self.num_experts,
            hidden,
            int(config.moe_intermediate_size),
            parallel,
        )
        self._init_shared_expert(config, hidden)

    # 【共享专家】Qwen1.5-MoE / Qwen2-MoE 的机制：每个 token 都必走这条支路，
    # **不经过 top-k**，目的是兜住"路由没选中的那些通用特征"。
    # Qwen3-MoE 去掉了它（shared_expert_intermediate_size = 0），代码保留是为了兼容前两代。
    def _init_shared_expert(self, config, hidden: int) -> None:
        # 取 0 表示这个模型没有共享专家；用 getattr + or 0 是为了兼容不写这个字段的 config。
        shared_inter = int(getattr(config, "shared_expert_intermediate_size", 0) or 0)
        if shared_inter <= 0:
            self.shared_expert_gate = None
            self.shared_expert_gate_up = None
            self.shared_expert_down = None
            self.shared_activation = None
            return
        # EP: replicate shared; add after routed AR so it is not summed ep times.
        # shard_tp 决定共享专家的三个线性层怎么建：
        #     EP → 传 1（**不切**，每卡一份完整）—— 这正是"必须先 AR 再加"的根源；
        #     TP → 传 None（走环境里的 TP 组，照常按中间维切开）。
        # 一个变量同时喂给三个层的 tp_size，保证它们口径一致。
        shard_tp = 1 if self.use_ep else None
        self.shared_expert_gate = ReplicatedLinear(hidden, 1, bias=False)
        # gate 与 up 合并成一个矩阵乘（MergedColumnParallelLinear）：[shared_inter] * 2
        # 表示输出维是两段拼接。合并的好处是少一次权重读取 —— decode 是访存瓶颈（见 v5 性能文档）。
        self.shared_expert_gate_up = MergedColumnParallelLinear(
            hidden, [shared_inter] * 2, bias=False, tp_size=shard_tp
        )
        # down_proj 是行并行：输入是切开的中间维、输出需要跨卡汇总
        # —— 但这里 reduce_results=False，**故意不在这一层里做 all-reduce**。
        # 原因是它下游还要与 routed 结果相加，而那个加法在两种并行下的顺序不同（见 forward）：
        # 与其在这里通信一次、出去再通信一次，不如把归约留到最后一并做。
        # input_is_parallel=not use_ep：EP 下这个层的输入是完整的（每卡一份），TP 下是分片。
        self.shared_expert_down = RowParallelLinear(
            shared_inter,
            hidden,
            bias=False,
            reduce_results=False,
            tp_size=shard_tp,
            input_is_parallel=not self.use_ep,
        )
        self.shared_activation = SiluAndMul()

    # 共享专家前向：线性 → SwiGLU → 线性 → sigmoid 门控缩放。
    # 全程用完整/分片的本地权重算，跨卡的事交给调用方（forward）决定顺序。
    def _shared_expert(self, hidden: torch.Tensor) -> torch.Tensor:
        assert self.shared_expert_gate_up is not None
        assert self.shared_expert_down is not None
        assert self.shared_activation is not None
        assert self.shared_expert_gate is not None
        shared = self.shared_expert_down(
            self.shared_activation(self.shared_expert_gate_up(hidden))
        )
        # 【sigmoid 门控】共享专家的输出还要乘一个标量门 sigmoid(gate(hidden))，
        # 让它能自己调"通用特征占多少"。注意它与 top-k 权重是两套东西：
        # top-k 权重管路由到的专家之间怎么加权，这个门只管共享支路的整体强度。
        return F.sigmoid(self.shared_expert_gate(hidden)) * shared

    # ============================================================================
    # 前向：展平 → 路由 → 专家（局部和）→ 按并行方式决定 AR 与共享专家的顺序。
    # ============================================================================
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        # 把可能带 batch 维的形状压成 [T, H]：MoE 只在 token 维上做事。
        # 最后再 view 回原形状 —— 这样上一层不需要知道 MoE 内部是几维。
        hidden = hidden_states.view(-1, orig_shape[-1])
        topk_weights, topk_ids = route_experts(
            self.gate(hidden), self.top_k, norm_topk_prob=self.norm_topk_prob
        )
        # 到这里 out 是**局部和**：EP 下只含本卡专家的贡献，TP 下只含中间维分片的贡献。
        # 两种情况都需要一次跨卡归约，但"何时归约"取决于共享专家是否完整（见下）。
        out = self.experts(hidden, topk_ids, topk_weights)
        # 判断标准是 **TP 组的 size**（不是 ep）：EP 只是在同一组 rank 上改切法，
        # 组本身没变 —— 所以 EP 下这里同样为 True，仍然要 AR。
        need_ar = get_tp_group().size > 1
        # ★ EP：先把 routed 的局部和归约成全局，再加共享专家。
        # 若顺序反过来，共享专家（每卡完整一份）会被求和 ep 次 ⇒ 数值偏大 ep 倍。
        if self.use_ep:
            if need_ar:
                out = tensor_model_parallel_all_reduce(out)
            if self.shared_expert_gate_up is not None:
                out = out + self._shared_expert(hidden)
        else:
            if self.shared_expert_gate_up is not None:
                out = out + self._shared_expert(hidden)
            if need_ar:
                out = tensor_model_parallel_all_reduce(out)
        # 恢复成进来时的形状（可能带 batch 维），交给下一层。
        return out.view(orig_shape)
