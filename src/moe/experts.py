# ============================================================================
# 【本文件是做什么的】MoE 的专家权重：怎么放、怎么从 checkpoint 装进来、
# 以及 EP 下「哪个专家在本卡、哪个在别人卡上」这张对照表。
#
# 【为什么权重是 3D 而不是"一个专家一个 nn.Linear"】
#     w13: [本卡专家数, 2 × 中间维分片, hidden]
#     w2 : [本卡专家数, hidden, 中间维分片]
# 把同一层的专家叠成一个连续张量，fused kernel 才能一次算完所有专家 ——
# 这就是"fused"的前提：专家号只是行索引，不是 60 次 kernel 启动。
# 如果写成 60 个 nn.Linear，光启动开销就把 MoE 的优势吃掉了。
#
# 【w13 里的两段是什么】第 0 段是 up_proj、第 1 段是 gate_proj。
#     HF 的 checkpoint 存的是 [gate, up]，fused kernel 要 [up, gate]，
#     所以要翻一次顺序（_hf_gate_up_to_cutlass 与 _w13_slice 分别管加载的两条路径）。
#
# 【"per_partition" 是什么】TP 下每个专家自己还要再切一刀中间维；
#     EP 下 MoE 的 tp 被塌缩成 1（见 parallel.py），这个分片数就退化成"整份"。
#     同一份代码同时服务两种切法，靠的就是这个字段。
#
# 【expert_map 的作用】global 专家号 → 本卡局部专家号；不在本卡的填 -1。
#     融合核看到 -1 就跳过这一路 —— 于是"单卡只算自己的专家"这件事不需要改 kernel，
#     只是一次查表。代价是每张卡都要读完整 hidden（token 不搬家）。
#
# ⚠️ -1 只是给融合核看的哨兵值，不能拿去当 tensor 索引（torch 里 -1 会绕到最后一个专家）。
# ============================================================================
"""Experts weights, EP map, and fused forward."""

from __future__ import annotations

import torch
from torch import nn

from ..distributed import divide, ensure_divisibility
from .fused import fused_moe_forward
from .parallel import MoEParallelConfig


# 【专家分配公式】num_experts 按 ep_size 均分，第 r 段就是第 r 个 rank 的。
# 8 个专家、ep_size=2 ⇒ 卡 0 拿 [0,4)、卡 1 拿 [4,8)。
# 所以要求 num_experts % ep_size == 0（在 config 校验里卡住）。
def local_expert_range(num_experts: int, ep_size: int, ep_rank: int) -> tuple[int, int]:
    n_local = divide(num_experts, ep_size)
    start = ep_rank * n_local
    return start, start + n_local


# 【全局 → 局部的对照表】形状 [num_experts]（全局长度），本卡持有的段填 0,1,2…，
# 其余全填 -1。融合核用它把 "专家 7" 翻译成 "本卡第 3 个" 或 "不是我的"。
# 注意表长是全局专家数 —— 每个 rank 拿到的表长度一样，只是填的内容不同。
def make_expert_map(
    num_experts: int,
    ep_size: int,
    ep_rank: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    # 本卡专家 → 0..n_local-1；别人卡上的专家 → -1。
    # dtype 默认 int32：融合核的 topk_ids 参数就是 int32，省一次转换。
    """``map[global_id] = local_id``; remote experts are ``-1``."""
    mapping = torch.full((num_experts,), -1, dtype=dtype, device=device)
    start, end = local_expert_range(num_experts, ep_size, ep_rank)
    mapping[start:end] = torch.arange(end - start, dtype=dtype, device=device)
    return mapping


# 【为什么 HF 的 gate/up 要交换】HF 的 MoE 权重把 gate_proj 与 up_proj 拼在一起、
# gate 在前；fused kernel 约定的是 up 在前。这里 chunk 再 cat，把顺序翻过来。
# 只影响加载，不影响计算 —— 但翻错会得到"能跑、结果全错"的静默错误。
def _hf_gate_up_to_cutlass(w13_hf: torch.Tensor) -> torch.Tensor:
    gate, up = w13_hf.chunk(2, dim=1)
    return torch.cat([up, gate], dim=1)


# 【判"这一段属于谁"靠文件名后缀】w13 里 up 占前 I 行、gate 占后 I 行，
# 而 HF 的两个权重文件都叫 gate_proj / up_proj（Qwen3-MoE 里叫 w1 / w3），
# 加载时是分开调进来的，必须知道自己该写哪一段。
# 注意两个习惯相反：w1=gate、w3=up（HF 的编号顺序 ≠ 拼接顺序）。
def _w13_slice(weight_name: str, intermediate_size: int) -> slice:
    if weight_name.endswith(("up_proj.weight", "w3")):
        return slice(0, intermediate_size)
    if weight_name.endswith(("gate_proj.weight", "w1")):
        return slice(intermediate_size, 2 * intermediate_size)
    raise ValueError(f"unknown gate/up weight: {weight_name}")


# 【为什么要容忍转置】不同来源的 fused checkpoint 存的是 [E, 2I, H] 还是 [E, H, 2I]
# 并没有统一约定。与其要求用户改文件，不如这里认两种形状，转置过来就行。
# transpose(1, 2) 之后必须 contiguous()：后面要 copy_ 进参数张量，非连续视图会报错。
def _normalize_fused_weight(
    loaded: torch.Tensor,
    expected_shape: tuple[int, int, int],
    name: str,
) -> torch.Tensor:
    assert loaded.ndim == 3, f"{name} must be 3D, got {tuple(loaded.shape)}"
    if loaded.shape == expected_shape:
        return loaded
    transposed = (expected_shape[0], expected_shape[2], expected_shape[1])
    if loaded.shape == transposed:
        loaded = loaded.transpose(1, 2).contiguous()
    assert loaded.shape == expected_shape, (
        f"{name} shape {tuple(loaded.shape)} != {expected_shape}"
    )
    return loaded


# ============================================================================
# 专家权重容器：本卡持有 [expert_start, expert_end) 这一段专家的完整权重。
# ============================================================================
class Experts(nn.Module):
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        parallel: MoEParallelConfig,
    ) -> None:
        super().__init__()
        ensure_divisibility(intermediate_size, parallel.tp_size)
        ensure_divisibility(num_experts, parallel.ep_size)
        # global_num_experts 是全局专家数（配置里那个数），下面 self.num_experts 是
        # 本卡真正持有的个数 —— 两个数都要留，前者用来建 expert_map，后者用来建权重。
        self.global_num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.tp_size = parallel.tp_size
        self.tp_rank = parallel.tp_rank
        self.ep_size = parallel.ep_size
        self.ep_rank = parallel.ep_rank
        self.fused_backend = parallel.fused_backend
        # 本卡负责的专家区间：EP 下是全局的一段，不开 EP 时就是全部。
        self.expert_start, self.expert_end = local_expert_range(
            num_experts, parallel.ep_size, parallel.ep_rank
        )
        self.num_experts = self.expert_end - self.expert_start
        # 单个专家的中间维分片 = 中间维 / tp_size。EP 下 tp_size=1 ⇒ 恰好是整个中间维。
        self.intermediate_size_per_partition = divide(intermediate_size, parallel.tp_size)
        i = self.intermediate_size_per_partition
        # w13：本卡每个专家的 up 与 gate 拼在一起（前 I 段 up、后 I 段 gate）。
        # 用 empty 而不是 zeros：所有权重位都会在加载时被覆盖，
        # 多一步清零既是浪费、也会让"忘了加载某个专家"这种 bug 更难发现。
        self.w13_weight = nn.Parameter(
            torch.empty(self.num_experts, 2 * i, hidden_size)
        )
        # w2：down_proj，形状 [本卡专家数, hidden, 中间维分片] —— 注意中间维在最后一维。
        self.w2_weight = nn.Parameter(torch.empty(self.num_experts, hidden_size, i))
        setattr(self.w13_weight, "weight_loader", self.weight_loader_w13)
        setattr(self.w2_weight, "weight_loader", self.weight_loader_w2)
        # 把 expert_map 注册成 buffer（而不是普通属性）：它能跟着 .to(device) / state_dict 走，
        # 且 persistent=False 表示不写进 checkpoint（它可以从 ep_rank 现场算出来，没必要存）。
        # 先建在 CPU 上：权重加载阶段不需要它占显存，前向时才搬到 GPU。
        self.register_buffer(
            "expert_map",
            make_expert_map(
                num_experts,
                parallel.ep_size,
                parallel.ep_rank,
                device=torch.device("cpu"),
            ),
            persistent=False,
        )

    # 全局专家号 → 本卡局部号；不在本卡返回 None（调用方据此跳过，不报错）。
    def _local_id(self, expert_id: int) -> int | None:
        if expert_id < self.expert_start or expert_id >= self.expert_end:
            return None
        return expert_id - self.expert_start

    # 【w13 的权重加载器】签名里的 expert_id 是全局专家号，由模型加载器逐个传进来。
    # 一个专家的加载要做三件事：①不是我的就跳过；②TP 时切中间维；③按文件名写进 up 段或 gate 段。
    def weight_loader_w13(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        expert_id: int,
    ) -> None:
        # 不属于本卡的专家：直接返回。这就是 EP 下「加载 60 个专家、每卡只留 30 个」的实现方式。
        local = self._local_id(expert_id)
        if local is None:
            return
        # 本卡要保留的中间维分片大小（EP 下就是整个中间维）。
        shard = self.intermediate_size_per_partition
        # 沿中间维（dim 0）取属于本 rank 的那一片。
        # 注意这里切的是加载进来的完整权重，与本卡是否 EP 无关 —— TP 才需要切。
        piece = loaded_weight.narrow(0, self.tp_rank * shard, shard)
        # 写进目标参数：先选本卡第 local 个专家，再按 up/gate 选前段或后段。
        dest = param.data[local, _w13_slice(weight_name, shard)]
        assert dest.shape == piece.shape, f"{dest.shape} != {piece.shape}"
        dest.copy_(piece)

    # 【w2 的权重加载器】与 w13 唯一的差别是"中间维在第几维"：
    #     w13 是 [2I, H] ⇒ 中间维是 dim 0，narrow(0, ...)
    #     w2  是 [H, I] ⇒ 中间维是 dim 1，narrow(1, ...)
    # 切错维度会得到一个形状对得上、数值全错的权重（assert 也拦不住）。
    def weight_loader_w2(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        expert_id: int,
    ) -> None:
        local = self._local_id(expert_id)
        if local is None:
            return
        shard = self.intermediate_size_per_partition
        # 与 w13 对称：只把维度换成 1。
        piece = loaded_weight.narrow(1, self.tp_rank * shard, shard)
        dest = param.data[local]
        assert dest.shape == piece.shape, f"{dest.shape} != {piece.shape}"
        dest.copy_(piece)

    # 【fused checkpoint 路径】上面两个 weight_loader 是"HF 逐专家喂进来"；
    # 这两条是"已经拼好的 [E, 2I, H] 张量整体喂进来"（fused / 量化权重的装载格式）。
    # 差别在于：这里要自己做"取本卡那一段专家 + 取本卡那一片中间维"的切片。
    def load_fused_w13(self, loaded: torch.Tensor) -> None:
        loaded = _normalize_fused_weight(
            loaded,
            (self.global_num_experts, 2 * self.intermediate_size, self.hidden_size),
            "gate_up",
        )
        # 先翻成 [up, gate]，再取本卡负责的专家段 —— 两步的顺序不能反：
        # 交换是在"全局 2I 维"上做的，先切片的话 up/gate 的边界就对不上了。
        loaded = _hf_gate_up_to_cutlass(loaded)[self.expert_start : self.expert_end]
        shard = self.intermediate_size_per_partition
        start = self.tp_rank * shard
        u = loaded[:, start : start + shard, :]
        g = loaded[
            :,
            self.intermediate_size + start : self.intermediate_size + start + shard,
            :,
        ]
        # 拼成本卡布局 [up 分片, gate 分片]。目的张量形状是 [E_local, 2*shard, H]，正好对上。
        self.w13_weight.data.copy_(torch.cat([u, g], dim=1))

    # w2 的 fused 路径：形状已经是 [E, H, I]（中间维在最后一维），
    # 所以只需要"取本卡专家段 + 沿最后一维取本 rank 的分片"，不需要交换顺序。
    def load_fused_w2(self, loaded: torch.Tensor) -> None:
        loaded = _normalize_fused_weight(
            loaded,
            (self.global_num_experts, self.hidden_size, self.intermediate_size),
            "down",
        )[self.expert_start : self.expert_end]
        shard = self.intermediate_size_per_partition
        start = self.tp_rank * shard
        self.w2_weight.data.copy_(loaded[:, :, start : start + shard])

    # ============================================================================
    # 前向：查表翻译专家号 → 交给 fused kernel → 返回本卡的局部和。
    # 跨卡求和不在这里做，由 block.py 在正确的时机发起 all-reduce。
    # ============================================================================
    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        # 明确拒绝 CPU：fused MoE 是 GPU kernel，CPU 上没有等价实现。
        # 与其在 kernel 里炸出难懂的报错，不如在这里给一句人话。
        if not hidden_states.is_cuda:
            raise RuntimeError("MoE fused kernels require CUDA tensors")
        # 把 expert_map 搬到与 topk_ids 同设备（它是 CPU buffer，见 __init__）。
        # topk_ids.long() 是因为查表要整型索引；查完远端专家变成 -1，融合核会跳过它们。
        # 于是本卡算出的只是"自己那几个专家的贡献"= 局部和 —— 这正是 EP 需要 all-reduce 的原因。
        mapping = self.expert_map.to(device=topk_ids.device, non_blocking=True)
        return fused_moe_forward(
            backend=self.fused_backend,
            hidden_states=hidden_states,
            topk_ids=mapping[topk_ids.long()],
            topk_weights=topk_weights,
            w13_weight=self.w13_weight,
            w2_weight=self.w2_weight,
        )
