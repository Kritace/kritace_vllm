# ==============================================================================
# 逻辑页池：free / used / cached。
#
# KVManager 通过本类给 Sequence.block_table 发页号。
# 页号与 compute/kv_pool.py 里 GPU 张量的 block 下标一一对应，这里不碰 CUDA。
# ==============================================================================


# 一个物理页的元信息。ref_count 是引用计数，hash/token_ids 是它缓存的内容指纹。

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable

# ==============================================================================
# 逻辑页池：free / used / cached。
#
# KVManager 通过本类给 Sequence.block_table 发页号。
# 页号与 compute/kv_pool.py 里 GPU 张量的 block 下标一一对应，这里不碰 CUDA。
# ==============================================================================


# 一个物理页的元信息。ref_count 是引用计数，hash/token_ids 是它缓存的内容指纹。
@dataclass
class Block:
    ref_count: int = 0
    # 内容指纹（链式块哈希）。-1 表示"这个页的内容没有被缓存的价值"。
    hash: int = -1
    token_ids: list[int] = field(default_factory=list)

    # 算完一个块之后把它的内容与哈希记下来，供前缀索引使用。
    def update(self, h: int, token_ids: list[int]) -> None:
        self.hash = h
        self.token_ids = list(token_ids)

    # 清空指纹：页即将被重用，它的旧内容不再有任何意义。
    def clear_meta(self) -> None:
        self.hash = -1
        self.token_ids = []

    # 分配一个新页时的初始化：ref_count 置 1（刚被某条序列认领），并清掉旧指纹。
    def reset(self) -> None:
        self.ref_count = 1
        self.clear_meta()


# 页池。它只管"页"，不管 token；前缀匹配的决策在上层（manager.py）。
class BlockManager:
    # 构造期就把所有页建好，之后只做状态迁移，不再有分配动作。
    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        # 三个集合互斥地描述每个页的状态，见文件头的不变式。
        self.num_blocks = num_blocks
        self.block_size = block_size
        # free_ids 用 deque 而不是 list：分配走 popleft()，是 O(1)；
        # list.pop(0) 是 O(n)，在每步都分配的高频路径上不可接受。
        self.blocks = [Block() for _ in range(num_blocks)]
        # used_ids 用 set：判断"某个页是否在用"、"移除"都是 O(1)。
        self.free_ids: deque[int] = deque(range(num_blocks))
        # cached_ids 用 OrderedDict 是为了 LRU：插入顺序 = 新旧顺序。
        # 若换成 set，就不知道"最旧的页是哪个"，驱逐只能随机选 —— 命中率会明显下降。
        self.used_ids: set[int] = set()
        # 前缀命中后暂时留下的页；最旧的在队头，缺页时先赶走。
        # LRU: oldest at front → evict first under pressure (hash / fallback).
        self.cached_ids: OrderedDict[int, None] = OrderedDict()
        # 页被回收重用时调这个回调，让上层把前缀索引里指向它的条目删掉。
        # 忘了挂这个回调 → 前缀索引会指向内容已变的页 → 结果静默错误。
        self.on_evict: Callable[[int], None] | None = None
        # radix 用：从树上摘冷叶子，返回的页号已经不在索引里，不要再 on_evict。
        # Optional: reclaim ``n`` cached pages (e.g. radix tree.evict). Returns
        # page ids already removed from the index; caller must not notify again.
        # radix 后端的专用回收钩子：它按"树叶 LRU"挑页，比池级 LRU 更准。
        self.reclaim_fn: Callable[[int], list[int]] | None = None

    # ---------- 三个只看不动的计数 ----------
    # 注意 available_slots 把 cached 也算进来了 —— 因为 cached 可以被回收来用。
    # 真正的"绝对可用"只有 num_free；调度器判断能不能跑时要看前者。
    @property
    def num_free(self) -> int:
        return len(self.free_ids)

    @property
    def num_cached(self) -> int:
        return len(self.cached_ids)

    @property
    def available_slots(self) -> int:
        return len(self.free_ids) + len(self.cached_ids)

    # 分配一个全新页：没有命中前缀时走这条路。
    def allocate_fresh(self) -> int:
        # 没空闲页就先腾一个出来（可能触发驱逐）。
        if not self.free_ids:
            self._ensure_free_page()
        bid = self.free_ids.popleft()
        # block.reset() 会把 ref_count 置 1 并清空指纹 —— 这一步同时完成了
        # "认领"和"抹掉旧身份"两件事，漏了任何一件都会出问题。
        block = self.blocks[bid]
        block.reset()
        self.used_ids.add(bid)
        return bid

    # 腾出一个空闲页。这是页池里最微妙的一段，三种策略有优先顺序。
    def _ensure_free_page(self) -> None:
        # 已经有空闲页就什么都不做（幂等）。
        if self.free_ids:
            return
        # Prefer backend-specific reclaim (radix tree leaf LRU).
        # 第一优先级是后端专属回收（radix 的树叶 LRU），而不是池级 LRU。
        # 原因：radix 树知道"哪些页是树的叶子、最近没被访问"，这个信息比
        # "页池里的插入顺序"更准确，能少丢缓存。
        if self.reclaim_fn is not None and self.cached_ids:
            for bid in self.reclaim_fn(1):
                self._take_cached_to_free(bid, notify=False)
            if self.free_ids:
                return
        # Hash (and radix fallback): page-pool LRU.
        # 连一个 cached 页都没有 → 池子真的用尽了。
        # 这是个真实故障（不是暂时性），要抛出来让上层触发抢占。
        if not self.cached_ids:
            raise RuntimeError("KV block pool exhausted")
        # Peek oldest; ``_take_cached_to_free`` removes it from cached_ids.
        # 兜底：取最旧的 cached 页（LRU 队首）淘汰。
        # 这里只是"看一眼"，真正的移除在 _evict_cached 里做（它负责通知索引）。
        bid = next(iter(self.cached_ids))
        self._evict_cached(bid)

    # cached → free 的唯一通道。notify 决定要不要通知上层清索引。
    def _take_cached_to_free(self, bid: int, *, notify: bool) -> None:
        # 幂等保护：不在 cached 集合里就直接返回。
        if bid not in self.cached_ids:
            return
        # 通知只能在真正移出之前发（下面第 88 行才 pop）。
        # 索引清理逻辑需要读到这个页当前的 hash，先 pop 就读不到了。
        if notify and self.on_evict is not None:
            self.on_evict(bid)
        self.cached_ids.pop(bid, None)
        self.blocks[bid].ref_count = 0
        self.blocks[bid].clear_meta()
        self.free_ids.append(bid)

    # 常规驱逐：一定要通知（因为页内容即将被覆盖）。
    def _evict_cached(self, bid: int) -> None:
        self._take_cached_to_free(bid, notify=True)

    # cached → used：某个请求命中前缀，把这个页"认领"回来。
    # 它会同时从 cached 和 free 里摘掉 —— 因为页可能正处在任意一个状态。
    def reclaim_cached(self, bid: int) -> None:
        self.cached_ids.pop(bid, None)
        try:
            self.free_ids.remove(bid)
        except ValueError:
            pass
        self.used_ids.add(bid)
        self.blocks[bid].ref_count = 1

    # used 引用 +1；cached 则拉回 used；free 则直接认领。三种来源都要处理。
    def acquire_hit(self, bid: int) -> None:
        """前缀命中：used 则 ref+1，cached 拉回 used，否则从 free 领走。"""
        # 已在 used：说明另一个请求也在用同一个前缀页，只加计数、不搬动。
        if bid in self.used_ids:
            self.blocks[bid].ref_count += 1
        elif bid in self.cached_ids:
            self.reclaim_cached(bid)
        else:
            self.free_ids.remove(bid)
            self.blocks[bid].ref_count = 1
            self.used_ids.add(bid)

    # 释放一次引用。计数归零不等于页可以立刻给别人用 ——
    # 这里直接放回 free 并清掉指纹，是"不再保留缓存价值"的语义。
    def release_hit(self, bid: int) -> None:
        # 先把计数减掉再看，不能反过来。
        block = self.blocks[bid]
        block.ref_count -= 1
        if block.ref_count == 0:
            # keep_cached 且内容是"算完的整块"（hash != -1）→ 进缓存供后来者复用。
            # hash == -1 的页（比如只写了一半）绝不能进缓存：
            #   它的内容不完整，被别的请求命中就会算错。
            self.used_ids.discard(bid)
            block.clear_meta()
            self.cached_ids.pop(bid, None)
            self.free_ids.append(bid)

    # LRU 触碰：这个 cached 页刚被用到，往后挪（降低被淘汰的概率）。
    def touch_cached(self, bid: int) -> None:
        if bid in self.cached_ids:
            self.cached_ids.move_to_end(bid)

    # 释放时决定去 cached 还是 free —— 正常结束路径都用这个。
    def release_to_cached_or_free(self, bid: int, *, keep_cached: bool) -> None:
        # 引用计数 > 0 说明还有人共享着这个前缀页，直接返回，什么都不做。
        block = self.blocks[bid]
        block.ref_count -= 1
        if block.ref_count > 0:
            return
        self.used_ids.discard(bid)
        if keep_cached and block.hash != -1:
            self.cached_ids[bid] = None
            self.cached_ids.move_to_end(bid)
        else:
            block.clear_meta()
            self.cached_ids.pop(bid, None)
            self.free_ids.append(bid)

    # 判断"再追加一个 token 需不需要新页"。
    # 判据是 `len % block_size == 1`：跨入新块之后的那个 token 才需要新页。
    # 写成 `== 0` 就晚了一格，会导致写越界。这个边界是本项目踩过的坑。
    def can_append(self, seq) -> bool:
        need_new = len(seq) % self.block_size == 1
        return (not need_new) or self.available_slots >= 1

    # 真正占页。注意它和 can_append 必须用同一个判据 ——
    # 两处不一致会造成"检查通过但没分配"或"分配了但没检查"。
    def may_append(self, seq) -> None:
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self.allocate_fresh())

    # ------------------------------------------------------------------
    # 投机解码：一次追加多个 token
    #
    # can_append / may_append 的判据 `len(seq) % block_size == 1` 隐含了
    # "每步只追加 1 个 token"的假设，投机解码一次提交 K+1 个，
    # 可能一次跨过多个页边界，所以另开一对方法而不是改老的。
    # ------------------------------------------------------------------

    def blocks_for(self, num_tokens: int) -> int:
        """容纳 num_tokens 个 token 需要多少页。"""
        if num_tokens <= 0:
            return 0
        return (num_tokens + self.block_size - 1) // self.block_size

    def can_append_n(self, seq, num_new_tokens: int) -> bool:
        """预检：在 seq 之上再追加 num_new_tokens 个 token，页够不够。

        ⚠️ 与 can_append 不同，本方法在**追加之前**调用，
        `num_new_tokens` 表示"将要追加的数量"。
        """
        need = self.blocks_for(len(seq) + num_new_tokens)
        return self.available_slots >= need - len(seq.block_table)

    def may_append_n(self, seq, num_new_tokens: int) -> None:
        """在 seq 之上预留 num_new_tokens 个 token 所需的页。

        ⚠️ 与 can_append_n 语义对齐，同样是**追加之前**调用。
        """
        need = self.blocks_for(len(seq) + num_new_tokens)
        while len(seq.block_table) < need:
            seq.block_table.append(self.allocate_fresh())

    # ------------------------------------------------------------------
    # ZOS 专用：按「本步要写的逻辑位置」判断 / 占块
    #
    # 为什么不能复用上面两个：can_append / may_append 的判据是 len(seq)，
    # 也就是「CPU 账本里已经有几个 token」。ZOS 调度在途序列时，账本比 GPU 落后一步，
    # 要写的槽位在 len(seq) 之后一格 —— 用旧判据会少分配一块，直接写越界。
    # 所以这里改成直接按「要写的逻辑位置 pos」算需要几块：
    #   覆盖 pos 这一格需要 ceil((pos+1)/block_size) 块。
    # ------------------------------------------------------------------

    def can_append_at(self, seq, pos: int) -> bool:
        need = (pos + self.block_size) // self.block_size
        have = len(seq.block_table)
        return need <= have or self.available_slots >= (need - have)

    def may_append_at(self, seq, pos: int) -> None:
        need = (pos + self.block_size) // self.block_size
        while len(seq.block_table) < need:
            seq.block_table.append(self.allocate_fresh())
