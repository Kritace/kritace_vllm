# ============================================================================
# 【本文件是做什么的】张量并行（TP）的两块基础砖：**列并行**与**行并行**，
# 外加两个融合变体（QKV 融合、多段列融合）。
#
# 【TP 的核心思想，两句话】把权重切开分到各卡上，每卡只算一部分；
# 缺的那部分用一次集合通信补回来。**切在哪一维，决定了通信插在哪**：
#     列并行（Column）：按**输出维**切 → 每卡产出一段输出
#         → 需要完整输出时 all-gather；不需要就保持分片往下传
#     行并行（Row）   ：按**输入维**切 → 每卡吃一段输入、算出的是一份**部分和**
#         → 必须 all-reduce 才能得到完整结果
#
# 【它们为什么总是一对出现】Transformer 里每一块都是"先切再合"：
#     Attention：QKV 投影（列）→ 注意力 → out 投影（行）
#     FFN      ：gate/up 投影（列）→ 激活       → down 投影（行）
#   中间那段激活**始终以分片状态存在**，一次都不需要拼回完整形状 ——
#   这就是 Megatron 式 TP 的结构：全程只有两处通信（列后的 all-gather 可省，
#   行后的 all-reduce 必须有）。
#
# 【为什么 bias 在两种切法里待遇不同】
#     列并行：bias 与**输出**同维 → 跟权重一起切
#     行并行：bias 只加在输出上，与输入维无关 → **不能切**，必须全量持有；
#             而且要加在 all-reduce **之后**，否则每个 rank 各加一次、重复 tp 倍
#
# 【权重加载为什么做得这么绕】HF 权重是**全量**的，而本 rank 的参数只有分片。
# 所以每个 Parameter 上都挂了一个 `weight_loader` 回调，由它负责
# "从全量张量里切出本 rank 需要的那一块，再就地写进已有内存"。
# `_shard` 返回的是 **view**（`narrow`），所以整个过程不会多出一份全量权重的峰值显存。
#
# 【⚠️ 三个容易踩的点】
#     ① `narrow` 返回的是**视图不是拷贝**：对视图 `copy_` 是就地写入（这正是我们要的），
#        但如果把视图存起来当参数，就会与源张量共享内存 —— 千万别那么用。
#     ② 行并行 forward 里 `input_is_parallel=False` 时，narrow 之后**必须 contiguous**：
#        GEMM 要连续内存，否则会静默走慢路径。
#     ③ QKV 融合层用字符串 `shard_id`（'q'/'k'/'v'/'bias'）区分三段。
#        字符串写错**不会报错**，只会把数据静默写到错误的偏移上。
#
# 【读这个文件的顺序】
#     _shard（怎么按 rank 取分片）
#       → ColumnParallelLinear / RowParallelLinear（两种切法 + 通信位置）
#       → QKVParallelLinear（融合 + GQA 的 KV Head 复制）
#       → MergedColumnParallelLinear（gate/up 这类多段融合）
#       → get_weight_loader（参数与加载逻辑之间的唯一约定）
# ============================================================================

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from ...distributed import (
    divide,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)

# ==============================================================================
# 模块说明：
# 本模块实现了LLM中常用的张量并行（Tensor Parallelism, TP）线性层组件。
# 主要包含：
# 1. ColumnParallelLinear: 列并行（按输出维度切分权重），常用于 FFN 的 gate/up 投影或 Attention 的 QKV 投影。
# 2. RowParallelLinear: 行并行（按输入维度切分权重），常用于 FFN 的 down 投影或 Attention 的 out 投影。
# 3. QKVParallelLinear: 融合的 QKV 投影层，支持 GQA/MQA 注意力机制的 Head 切分。
# 4. MergedColumnParallelLinear: 多列融合并行（如将 gate 和 up 融合），用于减少 Kernel Launch 开销。
# ==============================================================================



# ============================================================================
# 【可以在构造时显式指定 tp 组】否则所有并行层只能"读环境里的 TP 组"。
# MoE 引入了新情况：**同一个进程里存在两套切法** ——
#     Attention 按环境 TP 组切；
#     EP 下的共享专家要求"不切"（tp_size=1）；
#     普通 TP 下共享专家又要求"按环境 TP 组切"。
# 于是并联层多了一个可选参数：传 1 = 不切、传 None = 用环境组。
# ============================================================================
def _tp_meta(tp_size: int | None) -> tuple[int, int]:
    # 获取当前进程的 TP rank 和权重维度
    # 拿到 TP 组：既知道有几张卡，也知道自己是第几个。
    # 无论哪条分支都要先取一次组：即使是"传 1 不切"，也需要组对象来对齐 rank 语义
    # （后面 MoE 那边还要用它做 all-reduce）。
    tp = get_tp_group()
    # None = 跟随环境里的 TP 组（默认行为）。
    if tp_size is None:
        return tp.size, tp.group_rank
    # 显式 1 = 明确要求「不切分」：此时返回 (1, 0)，所有 rank 都是同一份完整权重。
    if tp_size == 1:
        return 1, 0
    # 传了别的值又与环境组不符时直接报错。
    # 为什么不容忍：这说明调用方以为自己在一个 size=N 的组里，实际不是 ——
    # 继续算下去会得到"每卡都算局部结果、但没人负责拼起来"的静默错误。
    if tp_size != tp.size:
        raise ValueError(f"tp_size={tp_size} does not match process group {tp.size}")
    return tp.size, tp.group_rank


# 【取分片 view】因为支持"显式指定 tp 组"，所以签名长了；
# 语义没变：返回的是 **narrow 出来的 view**，不是拷贝。
def _shard(
    weight: torch.Tensor,
    dim: int,
    *,
    tp_size: int | None = None,
    tp_rank: int | None = None,
) -> torch.Tensor:
    # tp_size==1 直接返回原张量：这是 EP 下共享专家的路径 ——
    # 每卡要有完整权重，切分必须退化成恒等。
    # 返回原张量（而不是 [:,]）还有一个好处：省掉一次包装，且 .data 地址与参数一致。
    """按 TP rank 取指定维上的分片 view。``tp_size==1`` 时返回原张量。"""
    # 没显式给就回落到环境 TP 组。
    if tp_size is None or tp_rank is None:
        size, rank = _tp_meta(None)
    else:
        size, rank = tp_size, tp_rank
    # size==1 时不做 narrow：既没必要，也会让「不切」这条路径多一层视图。
    if size == 1:
        return weight
    # 分片长度 = 总长 // size。除不尽时由调用方（divide / 校验）提前拦下 ——
    # 这里直接整除，除不尽会静默丢掉尾巴，所以上层必须保证整除。
    chunk = weight.shape[dim] // size
    return weight.narrow(dim, chunk * rank, chunk)


# ============================================================================
# 【整份复制的线性层】没有任何切分、也没有任何通信 ——
# 每个 rank 都持有完整的 [out, in] 权重与完整的 bias。
#
# 什么时候需要它：
#     · MoE 的 router（gate）：每个 rank 都要独立打分，且**必须打出完全一样的结果**
#       （否则同一 token 在不同卡上会去找不同专家，出口 all-reduce 拼起来就是错的）；
#     · EP 下的共享专家：每卡一份完整权重，算完再由上层决定 all-reduce 的顺序。
# 代价是显存冗余，收益是零通信 —— 对这两处（小矩阵 / 需要结果一致）是划算的。
# ============================================================================
class ReplicatedLinear(nn.Module):
    # 与列并行的区别：这里既不切权重、也不在 forward 里做任何集合通信。
    """整份复制、无通信。MoE router / EP 下的 shared expert 用这个。"""

    def __init__(self, in_size: int, out_size: int, *, bias: bool = False):
        super().__init__()
        # 权重形状就是完整的 [out, in]，与 HF checkpoint 里的形状一致 ——
        # 所以下面的 weight_loader 可以直接 copy_，不需要任何切片逻辑。
        self.weight = nn.Parameter(torch.empty(out_size, in_size))
        # 设置权重加载器
        # 权重同样挂上加载回调。
        # 仍走统一的 weight_loader 约定：上层按名字取到参数后调它，本类只是「整份拷进去」。
        setattr(self.weight, "weight_loader", self.weight_loader)
        # 初始化偏置，仅持有当前 Rank 负责的偏置部分
        # 列并行的 bias 与输出同维，所以跟着权重一起切。
        if bias:
            # bias 在这里是**整份**保留的（不切）—— 这与列并行不同。
            # 注意：上面那两行注释（"仅持有当前 Rank 负责的偏置部分 / 列并行的 bias 与输出同维"）
            # 是从列并行类抄过来的，对本类不成立：本类不做任何切分，see forward。
            self.bias = nn.Parameter(torch.empty(out_size))
            # 给 bias 挂加载回调：上层从 HF 权重字典读全量张量，回调负责切出本 rank 的那一段。
            setattr(self.bias, "weight_loader", self.weight_loader)
        else:
            # 显式注册 None：让参数遍历能看到「这里有 bias 这个概念，只是当前没有」。
            # 显式注册 None 而不是不注册：让 named_parameters / 遍历逻辑能看到
            # "这里有 bias 这个概念、只是当前没有"，加载器据此决定要不要找 bias 权重。
            self.register_parameter("bias", None)

    # 最简单的加载器：直接整份拷。没有分片偏移、没有 shard_id、没有 tp_rank ——
    # 这正是"复制层"与"并行层"在加载阶段的差别。
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param.data.copy_(loaded_weight)

    # 前向：本 rank 先算局部输出；要不要拼回完整输出由 gather_output 决定。
    # 前向就是一次普通矩阵乘，不做 gather、不做 reduce。
    # 注意：上面那行注释（"要不要拼回完整输出由 gather_output 决定"）以及
    # "返回 [q_local | k_local | v_local]"同样是抄自其它并行层的，对本类不适用 ——
    # 本类没有 gather_output 参数，返回值就是完整的 y。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 返回 [q_local | k_local | v_local]，分片状态继续往下传，切分交给上层注意力。
        # bias 为 None 时 F.linear 会自然跳过它（这也是上面要注册 None 的原因之一）。
        return torch.nn.functional.linear(x, self.weight, self.bias)


# 列并行：权重按**输出维**切，每张卡负责算出输出的一段。
class ColumnParallelLinear(nn.Module):
    """
        输出并行

        【数学原理】
        权重 W 按照输出维度（第0维）切分， W ∈ R^(out x in)，切分为 W_i ∈ R^((out/TP) x in。
        每个 Rank 独立计算局部的  Y_i = X * W_i^T。
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        *,
        bias: bool = False,
        gather_output: bool = True,
        tp_size: int | None = None,
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.gather_output = gather_output
        # 【把 tp 组在构造时钉死】否则每次切分都要去读环境里的 TP 组；
        # 现在允许显式传 tp_size（None=环境组、1=不切），所以在构造时解析一次并记住，
        # 后续加载与 forward 都用这两个数 —— 避免"构造时说不切、加载时又去读了环境组"这种不一致。
        self._tp_size, self._tp_rank = _tp_meta(tp_size)
        self.output_size_per_partition = divide(out_size, self._tp_size)
        # 初始化权重，仅持有当前 Rank 负责的权重部分
        # 权重形状 [out/tp, in]。注意参数**本身就只有分片**，不是「先全量再切片」。
        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_partition, in_size)
        )
        # 初始化偏置，仅持有当前 Rank 负责的偏置部分
        # 行并行的Bias是全量持有
        # bias 与输入维无关、只加在输出上 → 不能切，必须全量持有。
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            setattr(self.bias, "weight_loader", self.weight_loader)
        else:
            self.register_parameter("bias", None)
        # 设置权重加载器
        setattr(self.weight, "weight_loader", self.weight_loader)

    # 权重加载回调：把 HF 的全量权重按输出维切给本 rank。
    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int = 0
    ) -> None:
        param.data.copy_(
            # 加载时把这两个数一并传给切分函数：切谁、切多少必须与构造时的口径一致，
            # 否则会出现"参数按不切分配、权重却按切分写入"的错位（形状对得上也可能数值全错）。
            _shard(loaded_weight, dim=0, tp_size=self._tp_size, tp_rank=self._tp_rank)
        )

    # 前向：可选的输入切片 → 局部 GEMM → **all-reduce** 补回完整输出 → 加 bias。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 局部 GEMM，形状 [.., out/tp]。
        out = torch.nn.functional.linear(x, self.weight, self.bias)

        # 跨 TP 通信，将所有 Rank 的局部输出拼接为完整输出
        # 后接「按输入维切分」的层时可以**不拼**，直接把分片传下去 —— 这是 TP 省通信的关键。
        if self.gather_output:
            return tensor_model_parallel_all_gather(out, dim=-1)
        return out


# 行并行：权重按**输入维**切，每张卡吃一段输入，最后 all-reduce 求和。
class RowParallelLinear(nn.Module):
    """
        输入并行

        【数学原理】
        权重 W 按照输出维度（第1维）切分， W ∈ R^(out x in)，切分为W_i ∈ R^(out x (in/TP))。
        
        【使用场景】
        通常紧跟在 ColumnParallelLinear 之后（例如 FFN 的 down_proj 或 Attention 的 out_proj）。
        输出前 all-reduce 求和，确保各 rank 的输出累加为全局输出。
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        *,
        bias: bool = False,
        input_is_parallel: bool = False,
        reduce_results: bool = True,
        tp_size: int | None = None,
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        # input_is_parallel 仅决定 forward 是否还需 scatter 输入，不影响权重形状。
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results
        self._tp_size, self._tp_rank = _tp_meta(tp_size)
        self.input_size_per_partition = divide(in_size, self._tp_size)
        # 权重形状 [out, in/tp]。输出维是**全量**的 —— 所以行并行的单卡输出只是「部分和」。
        self.weight = nn.Parameter(
            torch.empty(out_size, self.input_size_per_partition)
        )
        # 行并行的Bias是全量持有
        if bias:
            self.bias = nn.Parameter(torch.empty(out_size))
            setattr(self.bias, "weight_loader", self.weight_loader)
        else:
            self.register_parameter("bias", None)
        setattr(self.weight, "weight_loader", self.weight_loader)

    # 加载回调：bias 全量拷，权重按输入维切。
    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int = 0
    ) -> None:
        # 同一个 loader 同时挂给了 bias 与 weight，所以第一步要认领「现在是谁」。
        if param is self.bias:
            param.data.copy_(loaded_weight)
            return
        param.data.copy_(
            _shard(loaded_weight, dim=1, tp_size=self._tp_size, tp_rank=self._tp_rank)
        )

    # 前向：一次融合 GEMM，**不 gather**。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel and self._tp_size > 1:
            shard = self.input_size_per_partition
            x = x.narrow(-1, self._tp_rank * shard, shard).contiguous()
        # 局部计算
        # 局部 GEMM：形状 [.., out]，但每张卡算出来的是**部分和**。
        out = torch.nn.functional.linear(x, self.weight)
        if self.reduce_results and self._tp_size > 1:
            # 跨 TP 通信，将所有 Rank 的局部输出拼接为完整输出
            # 关键一步：各卡的部分和相加才是完整结果。行并行的通信就在这一行。
            out = tensor_model_parallel_all_reduce(out)
        # 加上偏置
        # bias 必须在 all-reduce **之后**加：提前加会被累加 tp 次。
        if self.bias is not None:
            out = out + self.bias
        return out


# QKV 融合 + Head 切分：Q 均分到各卡；卡数多于 KV 头数时，KV 头**复制**到多张卡上。
class QKVParallelLinear(nn.Module):
    """
    融合 QKV 投影层
    
    【设计目的】
    将 Query, Key, Value 的投影矩阵融合为一个大矩阵，减少 Kernel Launch 次数，提高显存带宽利用率。
    
    【特性】
    支持 MHA、MQA 和 GQA。
    当 KV Head 数量小于 TP Size 时，会自动在多个 Rank 之间复制 KV Head。
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        *,
        bias: bool = False,
    ):
        super().__init__()
        tp_size = get_tp_group().size
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.num_heads_total = total_num_heads
        self.num_kv_heads_total = total_num_kv_heads
        # Q 均分到每个 rank
        self.num_heads = divide(total_num_heads, tp_size)
        # 处理 KV Head 数量小于 TP Size 的情况
        # 处理办法不是报错，而是让多张卡**共享同一个 KV 头**（各自持一份相同的权重）。
        if tp_size > total_num_kv_heads:
            self.num_kv_heads = 1
            # 记录每个 KV Head 需要被复制的次数 (例如 2个KV Head, 4个Rank -> 每个复制 2 次)
            # 每个 KV 头要复制给几个 rank：tp=4 / nkv=2 → 每个头 2 份。
            # 记录每个 KV Head 需要被复制的次数 (例如 2个KV Head, 4个Rank -> 每个复制 2 次)
            self.num_kv_head_replicas = divide(tp_size, total_num_kv_heads)
        else:
            self.num_kv_heads = divide(total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        # 本 rank 的 q 段 / kv 段宽度（头数 × head_size）。
        self.q_size = self.num_heads * head_size
        self.kv_size = self.num_kv_heads * head_size

        # 本地持有的融合权重形状: [q_size + 2*kv_size, hidden_size]
        self.weight = nn.Parameter(
            torch.empty(self.q_size + 2 * self.kv_size, hidden_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.q_size + 2 * self.kv_size))
            setattr(self.bias, "weight_loader", self.weight_loader)
        else:
            self.register_parameter("bias", None)
        setattr(self.weight, "weight_loader", self.weight_loader)

    # 从全量 qkv 权重里切出本 rank 的一段。q / k / v 三段用同一套「偏移 + 切片」规则。
    def _shard_qkv(
        self, loaded_weight: torch.Tensor, shard_id: str
    ) -> tuple[int, torch.Tensor]:
        """
        根据 shard_id ('q', 'k', 'v') 从全量单体权重中切出本 Rank 负责的部分
        """

        # 组内序号：下面算 KV 头归属要用它。
        tp_rank = get_tp_group().group_rank
        # q 段排在最前面，按 rank 连续切即可。
        if shard_id == "q":
            begin = 0
            return begin, loaded_weight[
                tp_rank * self.q_size : (tp_rank + 1) * self.q_size
            ]
        # k 与 v 段跟在 q 后面，两段的差别只在起始偏移。
        if shard_id == "k":
            begin = self.q_size
        else:  # "v"
            begin = self.q_size + self.kv_size
        
        # 处理 GQA/MQA 下的 KV Head 复制逻辑
        # 复制模式：本 rank 该拿哪个**全局** KV 头？用整除算归属。
        if self.num_kv_head_replicas > 1:
            # 例：nkv=2, tp=4 时，head0 归 rank0,1 共享、head1 归 rank2,3 共享
            # 通过整除计算当前 Rank 应该获取哪一个全局 KV Head
            # 整除的含义正是「每 replicas 张卡共享同一个头」。
            head_idx = tp_rank // self.num_kv_head_replicas
            start = head_idx * self.kv_size
        else:
            # 正常模式：第 tp_rank 个头。
            start = tp_rank * self.kv_size
        # 返回（在融合权重里的起始位置，切出来的分片）。
        return begin, loaded_weight[start : start + self.kv_size]

    # 融合层的加载回调：靠 shard_id（'q' / 'k' / 'v' / 'bias'）决定切哪一段、写到哪个偏移。
    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: str
    ) -> None:
        # 白名单校验：shard_id 是字符串，写错不会报错，只会静默加载到错误位置。
        assert shard_id in ("q", "k", "v", "bias")
        # bias 走单独路径：它在 HF 里是一整段拼接过的张量。
        if shard_id == "bias":
            self._load_bias(loaded_weight)
            return
        # 先算出本 rank 的段偏移与分片，再按偏移就地写进融合权重。
        begin, shard = self._shard_qkv(loaded_weight, shard_id)
        param.data[begin : begin + shard.shape[0]].copy_(shard)

    # bias 的加载比权重绕：HF 给的是 q/k/v 拼接后的一整段，要先按全量长度拆开，再套同一套切分规则。
    def _load_bias(self, loaded_weight: torch.Tensor) -> None:
        """
        加载合并形式的 QKV Bias。
        假设传入的 loaded_weight 格式为：[q_全量 | k_全量 | v_全量] 拼接而成的单段张量。
        """
        # 全量 q / kv 段长度（用 total 头数算，不是本 rank 的分片）。
        q_total = self.num_heads_total * self.head_size
        kv_total = self.num_kv_heads_total * self.head_size
        # 切分并加载 Q Bias
        # q bias：先取全量 q 段，再按 q 的规则取本 rank 那一段。
        _, q_shard = self._shard_qkv(loaded_weight[:q_total], "q")
        self.bias.data[: self.q_size].copy_(q_shard)
        # k / v 两段跟在 q 后面，各占 kv_total。
        k_seg = loaded_weight[q_total : q_total + kv_total]
        v_seg = loaded_weight[q_total + kv_total : q_total + 2 * kv_total]
        # k 与 v 逻辑相同、只有写入偏移不同，所以用循环避免重复代码。
        for idx, (seg, shard_id) in enumerate(((k_seg, "k"), (v_seg, "v"))):
            _, shard = self._shard_qkv(seg, shard_id)
            lo = self.q_size + idx * self.kv_size
            self.bias.data[lo : lo + self.kv_size].copy_(shard)

    # 前向同样是一次 GEMM，段的切分由上层按 output_sizes 做。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：直接进行融合矩阵乘法，不进行 Gather。
        返回的张量包含当前 Rank 负责的 [q_local | k_local | v_local]，由上层模型负责 Split。
        """
        return torch.nn.functional.linear(x, self.weight, self.bias)


# 多段列融合：把 gate / up 这种「同源不同段」的投影合成一个大 GEMM。
class MergedColumnParallelLinear(nn.Module):
    """
    多列融合并行线性层。
    
    【设计目的】
    将多个独立的列并行层融合为一个巨大的权重矩阵，减少 GPU Kernel 的 Launch 次数，提升硬件计算利用率。
    """

    def __init__(
        self,
        in_size: int,
        output_sizes: list[int],
        *,
        bias: bool = False,
        tp_size: int | None = None,
    ):
        super().__init__()
        self.in_size = in_size
        self.output_sizes = output_sizes
        self._tp_size, self._tp_rank = _tp_meta(tp_size)
        # 每一段各自按 tp 切，再拼成本 rank 的总宽度。
        self.output_sizes_per_partition = [
            divide(s, self._tp_size) for s in output_sizes
        ]
        # 本 rank 的总输出宽度 = 各段分片宽度之和。
        total = sum(self.output_sizes_per_partition)
        # 权重形状 [总宽, in]：一次 GEMM 同时得到 gate 与 up。
        self.weight = nn.Parameter(torch.empty(total, in_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(total))
            setattr(self.bias, "weight_loader", self.weight_loader)
        else:
            self.register_parameter("bias", None)
        setattr(self.weight, "weight_loader", self.weight_loader)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int
    ) -> None:
        # 本段起始偏移 = 前面各段**分片后**宽度之和（不是全量宽度）。
        shard_offset = sum(self.output_sizes_per_partition[:shard_id])
        # 先切出本 rank 的分片。
        shard_size = self.output_sizes_per_partition[shard_id]
        # 再写到融合权重里本段的位置。
        param.data[shard_offset : shard_offset + shard_size].copy_(
            _shard(loaded_weight, dim=0, tp_size=self._tp_size, tp_rank=self._tp_rank)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


# 默认加载器：不切分，直接全量拷贝（单卡层用它）。
def _default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    param.data.copy_(loaded_weight)


# 取参数上挂着的加载器；没挂就用默认的。这是模型层与权重加载逻辑之间的唯一约定。
def get_weight_loader(param: nn.Parameter) -> Callable:
    return getattr(param, "weight_loader", _default_weight_loader)
