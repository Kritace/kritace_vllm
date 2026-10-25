# ==============================================================================
# 前缀索引：hash → 逻辑页号。
#
# KVManager.allocate 调 match；写满一块后 publish。
# 同一哈希被后写的块覆盖；驱逐时 invalidate 删掉映射。
# ==============================================================================

from __future__ import annotations

from typing import TYPE_CHECKING

from .hash_util import compute_block_hash

if TYPE_CHECKING:
    from ..block_manager import BlockManager
    from ..sequence import Sequence

# ==============================================================================
# 前缀索引：hash → 逻辑页号。
#
# KVManager.allocate 调 match；写满一块后 publish。
# 同一哈希被后写的块覆盖；驱逐时 invalidate 删掉映射。
# ==============================================================================


class HashPrefixIndex:
    # 唯一的存储：块哈希 → 物理页号。一个哈希只保留一个页（更早的那个被覆盖）。
    def __init__(self) -> None:
        self.hash_to_block_id: dict[int, int] = {}

    # 【匹配】从第 0 块开始逐块查，一断就停（见文件头对"为什么能提前 break"的说明）。
    def match(self, seq: Sequence, pool: BlockManager) -> list[int]:
        h = -1
        hit: list[int] = []
        # 只遍历**满块**（num_full_blocks 是整除结果）：
        #   不满的块内容不完整，不能作为前缀被匹配。
        for i in range(seq.num_full_blocks()):
            toks = seq.block_token_ids(i)
            # 链式哈希：把上一块的哈希作为前缀掺进来。
            h = compute_block_hash(toks, h)
            # 查索引。用 get(h, -1) 而不是 []：未命中要能区分出"没有"而不是抛异常。
            bid = self.hash_to_block_id.get(h, -1)
            if bid < 0:
                break
            # 哈希命中**不等于**内容相同 —— 必须逐 token 比对确认（防碰撞）。
            # 少了这一步，一旦碰撞就会静默算出错误结果。
            block = pool.blocks[bid]
            if block.token_ids != toks:
                break
            # 还要确认这个页**当前是可用的**：
            #   used    → 正被别的在跑请求占着，共享它会让引用计数错乱
            #    cached  → 可以共享（这正是缓存的意义）
            #    两者都不是 → 页处于 free 或已被回收，它的内容随时会被覆盖
            if bid not in pool.used_ids and bid not in pool.cached_ids:
                break
            hit.append(bid)
        return hit

    # 【发布】把"已经算完的块"登记进索引，供后续请求复用。
    def publish(
        self,
        seq: Sequence,
        num_full_blocks: int,
        pool: BlockManager,
        *,
        start: int = 0,
    ) -> None:
        # 没有新块要发布（本步没跨过块边界）→ 直接返回，这也是最常见的情况。
        if num_full_blocks <= start:
            return
        h = -1
        # 链式哈希要求知道"前一块的哈希"。如果从中间开始，就得先把前一块的
        # 哈希捞出来（block.hash）。
        if start > 0:
            # 捞不到（前一块的 hash 是 -1，说明它自己也没登记）→ 只能退化成从头发布。
            # 不这样处理会导致链式哈希断链，算出来的哈希与真实内容不对应。
            h = pool.blocks[seq.block_table[start - 1]].hash
            if h == -1:
                start = 0
        # 逐块算哈希并登记。
        for i in range(start, num_full_blocks):
            bid = seq.block_table[i]
            toks = seq.block_token_ids(i)
            h = compute_block_hash(toks, h)
            # 顺序：先把哈希与 tokens 记到**页对象**上，再更新字典。
            # 因为页对象上的 hash 是后续"从中间发布"和"invalidate 定位"的依据。
            pool.blocks[bid].update(h, toks)
            old = self.hash_to_block_id.get(h, -1)
            # 只有当"没有旧条目 / 旧的就是自己 / 旧的内容确实相同"时才覆盖。
            # 如果旧条目指向一块内容**不同**的页（哈希碰撞），就不能覆盖 ——
            # 保留原来那个更安全（宁可少命中，也不能命中错）。
            if old < 0 or old == bid or pool.blocks[old].token_ids == toks:
                self.hash_to_block_id[h] = bid

    # 【失效】页被驱逐重用前必须调它，否则索引会指向内容已变的内存。
    def invalidate(self, block_id: int, pool: BlockManager) -> None:
        block = pool.blocks[block_id]
        # 两个条件都要满足：
        #   ① 这个页确实登记过（hash != -1）
        #    ② 索引里当前指向的**正是这个页**（可能已经被别人覆盖成了别的页号，
        #       那种情况下删掉就会误伤别人）
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]

    # hash 后端没有"每序列的状态"要清理 —— 索引是全局的，靠引用计数与驱逐维护。
    # 这个空实现是为了与 radix 后端保持接口一致。
    def release_seq(self, seq_id: int) -> None:
        return
