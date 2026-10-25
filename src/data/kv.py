# ============================================================================
# 【本文件是做什么的】driver 侧的 KV **账本**：给序列发页、查前缀命中、用完回收。
# 注意它 **不 import torch** —— 因为真正的显存池在 compute/kv_pool.py。
#
# 【为什么账本与显存池要分成两个文件】
#   这里把「调度」与「计算」拆到了不同进程：
#     账本（本文件）跟调度器同进程（driver 侧），只回答「哪条序列占了哪些块」；
#     显存池（compute/kv_pool.py）在每个 Worker 进程里，持有真正的大张量。
#   两边只需要对齐一个数字 —— **总块数**（由 Worker 报给门面）。
#   这样调度器不需要任何 GPU 上下文，也就能在 driver 侧自由抢占与重排。
#
# 【一条序列的生命周期，看这四个方法就够】
#     can_allocate  能不能放进来（准入，无副作用）
#       → allocate  真正发块（能复用前缀就复用，剩下新分配）
#       → may_append 续写时要不要再补块（跨页那一刻）
#       → deallocate 回收（决定"留着当缓存"还是"还回空闲"）
#
# 【前缀复用的核心：先问索引"最多能复用多少"】
#   `_fit_prefix` 把命中长度与"可回收的其它缓存块"一起算进可用空间 ——
#   这正是 cached 状态存在的意义：缓存块虽然被占着，但随时可以被挤掉让位。
#
# 【⚠️ 三个容易踩的点】
#     ① `publish=False` 时**绝不登记进前缀索引**。被抢占的序列 KV 不完整，
#        发布出去会让后续请求命中「看起来相同但不完整」的 KV —— 静默错误。
#     ② 块的两种状态转移顺序在后端之间是**反的**：hash 先 invalidate 再归还，
#        radix 先归还再 invalidate（radix 的树结构要先释放才能判断哪些页可回收）。
#     ③ `num_cached_tokens` 与 `block_table` 必须同时维护，改一边不改另一边会静默错位。
#
# 【读这个文件的顺序】
#     __init__ / num_free（账本的构成）
#       → _fit_prefix（前缀复用的判断，最关键的一处）
#       → can_allocate / allocate（准入与分配）
#       → can_append / may_append（续写）
#       → sync_prefix / deallocate（发布与回收）
# ============================================================================

from __future__ import annotations

from .block_manager import BlockManager
from .prefix import PrefixBackend, RadixPrefixIndex, make_prefix_index
from .sequence import Sequence

# ==============================================================================
# driver 侧 KV 账本：给序列发逻辑页、查前缀命中、用完回收。
#
# Scheduler 调 can_allocate / allocate / may_append / deallocate。
# GPU 上真正的 cache 张量在 compute/kv_pool.py，本类不 import torch。
# ==============================================================================


# 账本对象：整份只是块号的分配记录，不持有任何张量。
class KVManager:
    # 构造时定下三件事：块大小、前缀索引后端、总块数。
    def __init__(
        self,
        *,
        num_blocks: int,
        block_size: int,
        prefix_backend: PrefixBackend = "hash",
    ) -> None:
        # 0 块直接拒绝：那意味着调度器一个请求都放不进来。
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {num_blocks}")
        # 块大小（一个块装多少个 token）。
        self.block_size = block_size
        self.prefix_backend: PrefixBackend = prefix_backend
        # 总块数。num_blocks_schedulable 是其中可被调度器分配的部分（本版本两者相等）。
        self.num_blocks_total = num_blocks
        self.num_blocks_schedulable = num_blocks
        # 块管理器：负责三态（free / cached / used）与真正的分配动作。
        self.blocks = BlockManager(self.num_blocks_schedulable, block_size)
        # 前缀索引：按后端造一个（hash 或 radix）。
        self.index = make_prefix_index(prefix_backend)
        # 淘汰回调：块被挤出池子时，必须同步把索引里的记录作废，否则会命中已失效的块。
        self.blocks.on_evict = lambda bid: self.index.invalidate(bid, self.blocks)
        # radix 后端多一个能力：从树里回收页，所以要额外挂一个 reclaim 回调。
        if isinstance(self.index, RadixPrefixIndex):
            self.blocks.reclaim_fn = lambda n: self.index.reclaim_pages(n, self.blocks)
        # 供上层做前缀命中统计。
        self.last_num_cached_tokens = 0

    # 给调度器用：还剩多少块可用。
    @property
    def num_free(self) -> int:
        return self.blocks.num_free

    # 前缀复用预检查：先算「最多能复用多少连续前缀块」，再判断空间够不够。
    def _fit_prefix(self, seq: Sequence) -> int:
        """前缀复用预检查：计算当前序列最多可复用多少连续前缀 KV 块。"""
        # 这条序列总共需要多少块。
        need = seq.num_blocks
        # 一条空序列不需要任何块。
        if need == 0:
            return 0
        # 问索引：能命中哪几块（返回顺序就是前缀顺序）。
        hit = self.index.match(seq, self.blocks)
        # 命中数不能超过总需求 —— 序列可能比缓存短。
        max_cached = min(len(hit), need)
        # 命中的块里有多少处于 cached 状态（可被挤掉来腾空间）。
        reclaimable = sum(
            1 for b in hit[:max_cached] if b in self.blocks.cached_ids
        )
        # 池子里**其它** cached 块的数量：它们同样是可回收的候选。
        other_cached = len(self.blocks.cached_ids) - reclaimable
        # 复用之外，还需要新分配多少块。
        new_needed = need - max_cached
        # 关键判断：**自由块 + 可回收的其它缓存块** ≥ 新需求，就能允许复用。
        if self.blocks.num_free + other_cached >= new_needed:
            return max_cached
        # 放不下时返回 -1 作为信号，由调用方决定是拒绝还是抢占。
        return -1

    # 准入判断：能不能放下（不产生任何副作用）。
    def can_allocate(self, seq: Sequence) -> bool:
        return self._fit_prefix(seq) >= 0

    # 真正分块：能复用就复用，剩下新分配。
    def allocate(self, seq: Sequence) -> None:
        # 每次分配都从干净状态开始：块表清空、缓存计数归零。
        seq.block_table.clear()
        seq.num_cached_tokens = 0
        # 这条序列现在需要多少块。
        need = seq.num_blocks
        # 不需要块时也要刷新统计值，避免上层读到上一轮的残留。
        if need == 0:
            self.last_num_cached_tokens = 0
            return

        # 再算一次复用上限。
        max_cached = self._fit_prefix(seq)
        # 到这里还是 -1，说明调用方跳过了准入检查，直接报错比默默算错好。
        if max_cached < 0:
            raise RuntimeError("not enough KV blocks for sequence")
        # 再取一次命中列表：这次要真正用它来登记块。
        hit = self.index.match(seq, self.blocks)

        # 命中的块：引用计数 +1（从 cached 变 used），并写进本序列的块表。
        for bid in hit[:max_cached]:
            self.blocks.acquire_hit(bid)
            seq.block_table.append(bid)

        # 剩下的块新分配。
        for _ in range(max_cached, need):
            seq.block_table.append(self.blocks.allocate_fresh())

        # 缓存长度 = 复用的块数 × 页大小。这就是 prefill 起始 position 的来源。
        seq.num_cached_tokens = max_cached * self.block_size
        # 记给上层做命中统计。
        self.last_num_cached_tokens = seq.num_cached_tokens

        # radix 后端还要把命中的路径「钉住」，防止它在后续步骤里被回收。
        if isinstance(self.index, RadixPrefixIndex) and max_cached > 0:
            self.index.pin_match(seq, max_cached)

    # 续写判断：下一步要不要新块（纯转发给块管理器）。
    def can_append(self, seq: Sequence) -> bool:
        return self.blocks.can_append(seq)

    # 续写时真正补上跨页需要的那一块。
    def may_append(self, seq: Sequence) -> None:
        self.blocks.may_append(seq)

    def can_append_n(self, seq: Sequence, num_new_tokens: int) -> bool:
        """投机解码：一次追加 num_new_tokens 个 token 的预检。

        ⚠️ 投机路径下有两个不同的量，别弄混：
          - **写入 KV 的位置数** = K+1（验证阶段把草稿全写进去）
          - **最终有效的 token 数** = j+1（j 为被接受的草稿数）
        因此调用方必须按 K+1 预留，再在验证结束后把 num_cached_tokens
        推进到 j+1 —— 多余的 K-j 个 slot 留在同一页里，下步覆盖写即可。
        """
        return self.blocks.can_append_n(seq, num_new_tokens)

    def may_append_n(self, seq: Sequence, num_new_tokens: int) -> None:
        """投机解码：一次预留 num_new_tokens 个 token 所需的页。"""
        self.blocks.may_append_n(seq, num_new_tokens)

    # ZOS 专用：按「本步要写的逻辑位置」判断 / 占块（账本落后一步时用）。
    def can_append_at(self, seq: Sequence, pos: int) -> bool:
        return self.blocks.can_append_at(seq, pos)

    def may_append_at(self, seq: Sequence, pos: int) -> None:
        self.blocks.may_append_at(seq, pos)

    def sync_prefix(self, seq: Sequence, *, num_new_tokens: int | None = None) -> None:
        """本步写满的新块登记进前缀索引，供后来的请求命中。

        ``num_new_tokens`` 默认为 ``seq.num_scheduled_tokens``。
        投机解码必须显式传入**实际有效**的新增数（j+1 而不是 K+1）：
        否则会把"写了 KV 但被判定丢弃"的 token 也算进块哈希，
        后来的请求就会命中到与真实文本不符的前缀。
        """
        new_tokens = (
            seq.num_scheduled_tokens if num_new_tokens is None else num_new_tokens
        )
        if self.prefix_backend == "none" or new_tokens <= 0:
            return
        prev = seq.num_cached_tokens - new_tokens
        # 只登记满块：半截块的内容下次可能变，绝不能进索引。
        start = prev // self.block_size
        end = min(seq.num_full_blocks(), seq.num_cached_tokens // self.block_size)
        # 没有新的满块就直接返回。
        if start >= end:
            return
        # 发布。publish 内部只对满块生效 —— 这是前缀索引的核心不变式。
        self.index.publish(seq, end, self.blocks, start=start)

    # 回收：按 publish 决定「留着当缓存」还是「直接还回空闲」。
    def deallocate(self, seq: Sequence, *, publish: bool = True) -> None:
        # publish=False（被抢占）时按 0 算 —— 一块都不留。
        n_full = seq.num_full_blocks() if publish else 0
        # 正常结束时把满块发布进索引，让别人有机会命中。
        if publish and n_full > 0 and self.prefix_backend != "none":
            self.index.publish(seq, n_full, self.blocks)

        # 先撤掉这条序列在索引里的活动记录。
        self.index.release_seq(seq.seq_id)

        # 已发布的块集合：它们会以 cached 状态留在池子里。
        published = set(seq.block_table[:n_full]) if n_full > 0 else set()
        # 逐块归还。
        for bid in list(seq.block_table):
            # 只有「发布过 + 前缀缓存没关」的块才留着当缓存。
            keep = bid in published and self.prefix_backend != "none"
            # hash 后端：先作废索引记录、再归还。
            if not keep and self.prefix_backend == "hash":
                self.index.invalidate(bid, self.blocks)
            # 归还：按 keep 决定进 cached 还是 free。
            self.blocks.release_to_cached_or_free(bid, keep_cached=keep)
            # radix 后端相反：先归还、再作废 —— 树结构要先释放才能判断哪些页可回收。
            if not keep and self.prefix_backend == "radix":
                self.index.invalidate(bid, self.blocks)

        # 序列清空。块表与缓存计数必须一起归零，否则下次分配会错位。
        seq.block_table.clear()
        seq.num_cached_tokens = 0
