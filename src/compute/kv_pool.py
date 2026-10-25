# ============================================================================
# 【本文件是做什么的】分页 KV 池：一张大张量
# `[2, 层, 块数, 页内 token, KV 头, head_dim]`，加上「按剩余显存推算能开多少块」。
#
# 【为什么形状是这个顺序】
#     第 0 维是 K / V —— 写入时两个指针就能覆盖两种缓存；
#     第 1 维是层 —— 每一层各有一片独立空间，attention 按层取；
#     后面三维 [块, 页内 token, KV头 × head_dim] 让「一个 token 的 KV」在内存里
#     是一小块连续数据，正好对上 flash-attn 的 paged 接口。
# 这个布局决定了：写入是"按槽位算地址"，读取是"按页表算地址"，
# 两边都不需要连续，这正是分页的意义。
#
# 【为什么块数要所有 rank 取最小值】每张卡的可用显存不同（PP 不均分时差异更大），
# 但调度器只能按一个数来分块 —— 块号必须在所有 rank 上含义一致。
# 所以自动推算完要 all_gather 取 min，否则某张卡会先 OOM。
#
# 【为什么要先扣掉 CG 足迹】CUDA Graph 的捕获需要一块它自己的显存池。
# 如果不先把它扣掉，KV 池就会把这块吃满，等到捕获时直接 OOM。
# 所以顺序是：量 CG 足迹 → 再算可用显存 → 再估块数。
#
# 【⚠️ 一个口径陷阱】`memory_get_info` 返回的是驱动视角的可用量
# （包含了碎片），与 torch 的 `memory_reserved()` 不是同一个口径。
# 混用两个口径去算"还能开多少块"会得出偏乐观的结论。
#
# 【读这个文件的顺序】
#     get_kv_dims（本 rank 到底要建多大）
#       → KVPool.__init__ / _alloc_pool（开池）
#       → auto_num_blocks（按剩余显存反推块数，多卡取最小）
# ============================================================================

from __future__ import annotations

import torch
from transformers import PretrainedConfig

from ..distributed import all_gather, get_pp_group, get_pp_indices, get_tp_group


# 本 rank 实际要建的 KV 形状：层数被 PP 切，KV 头数被 TP 切。
def get_kv_dims(hf: PretrainedConfig) -> tuple[int, int, int]:
    # TP / PP 两个组都要问：层维归 PP 管，头维归 TP 管。
    tp = get_tp_group()
    pp = get_pp_group()
    # KV 头按 TP 切。max(1, ...) 是兜底 —— GQA 下 KV 头可能少于卡数（那种情况由头复制处理）。
    num_kv_heads = max(1, hf.num_key_value_heads // tp.size)
    # 本 rank 负责的层区间：池只为这几层分配空间。
    start, end = get_pp_indices(hf.num_hidden_layers, pp.group_rank, pp.size)
    # head_dim 优先读配置，没有就用 hidden / heads 推。
    head_dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
    # 返回（层数, KV 头数, head_dim）三元组。
    return end - start, num_kv_heads, head_dim


# 池对象本身很薄：持有大张量 + 记住块大小等元信息。
class KVPool:
    def __init__(
        self,
        hf_config: PretrainedConfig,
        *,
        num_blocks: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        # 块数为 0 直接拒绝：那意味着调度器一个请求都放不进来。
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {num_blocks}")
        self.hf_config = hf_config
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = device
        self.dtype = dtype
        # 一次性把整池开出来（不是按需增长）—— 地址固定对 CG 也是前提。
        self.kv_cache = self._alloc_pool(
            hf_config, num_blocks, block_size, device, dtype
        )

    @staticmethod
    # 开池：先算单块多少字节，再按 [2, 层, 块, 页内, 头, dim] 开一个大张量。
    def _alloc_pool(
        hf: PretrainedConfig,
        num_blocks: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # 形状由 TP 与 PP 共同决定。
        layers, nkv, hd = get_kv_dims(hf)
        # 单块字节 = 2(K/V) × 层 × 页内 token × KV头 × head_dim × 每元素字节数。
        block_bytes = 2 * layers * block_size * nkv * hd * dtype.itemsize
        # 一次开完。这是显存里最大的一块，形状顺序决定了后续的寻址方式。
        kv = torch.empty(
            2, layers, num_blocks, block_size, nkv, hd, dtype=dtype, device=device
        )
        print(
            f"KV pool: {num_blocks} blocks × {block_size} tok "
            f"({num_blocks * block_bytes / (1024**3):.2f} GiB)"
        )
        return kv

    # 给调度器用：池里总共有多少块。
    @property
    def num_blocks_total(self) -> int:
        return self.num_blocks

    # 按剩余显存反推块数。注意先扣掉 CG 足迹再算。
    @classmethod
    def auto_num_blocks(
        cls,
        hf: PretrainedConfig,
        *,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device,
        gpu_memory_utilization: float,
        reserve_bytes: int = 0,
    ) -> int:
        # 先把缓存清干净、同步，否则读到的显存数是虚的。
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        # 复用同一套形状公式算单块字节。
        layers, nkv, hd = get_kv_dims(hf)
        block_bytes = 2 * layers * block_size * nkv * hd * dtype.itemsize
        # 驱动视角的可用 / 总量（与 torch 的 reserved 不是同一个口径）。
        free, total = torch.cuda.mem_get_info(device)
        # 可用量 = 总量 × 利用率上限 − 已占用部分（权重、激活等都算在内）。
        available = int(total * gpu_memory_utilization) - (total - free)
        # 再把 CUDA Graph 要占的那部分扣掉。
        available -= reserve_bytes
        # 至少留 4 块：避免极小显存下算出 0，让引擎根本起不来。
        num_blocks = max(4, available // block_bytes)
        # 取所有 rank 的最小值 —— 块号必须在每张卡上含义一致。
        return int(
            all_gather(torch.tensor([num_blocks], device=device), dim=0).min().item()
        )
