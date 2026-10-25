# ==============================================================================
# 前缀索引：把链式块哈希插进压缩 radix 树，值为逻辑页号。
#
# 比 HashPrefixIndex 能共享任意公共前缀。活着的序列 pin 树上的节点，
# 没人引用的叶子才能被 BlockManager 挤掉。
# ==============================================================================


# 对外的 radix 门面。它自己不实现树，全部委托给 RadixTree。

from __future__ import annotations

from typing import TYPE_CHECKING

from .hash_util import compute_block_hash
from .radix_tree import RadixTree

if TYPE_CHECKING:
    from ..block_manager import BlockManager
    from ..sequence import Sequence

# ==============================================================================
# 前缀索引：把链式块哈希插进压缩 radix 树，值为逻辑页号。
#
# 比 HashPrefixIndex 能共享任意公共前缀。活着的序列 pin 树上的节点，
# 没人引用的叶子才能被 BlockManager 挤掉。
# ==============================================================================


# 对外的 radix 门面。它自己不实现树，全部委托给 RadixTree。
class RadixPrefixIndex:
    # 构造很轻：一棵空树 + 一张"序列 → 节点"的映射表。
    def __init__(self) -> None:
        self.tree = RadixTree()
        # 映射表必须在下面所有操作里保持同步（见文件头说明）。
        # 它存在的理由：树节点只有 ref_count，不知道"是哪些序列在钉着我"；
        # 释放某条序列时要精确定位到它的节点，只能靠这张表。
        self._seq_node: dict[int, object] = {}

    # 把前 n 个块换算成链式哈希列表 —— match / pin_match 都要用，抽出来避免算两遍。
    def _block_hashes(self, seq: Sequence, n: int) -> list[int]:
        h = -1
        # guard 是死循环保护。必需：如果树的状态异常
        # （比如所有叶子都有引用计数、摘不掉），没有这个上限会永远转下去。
        out: list[int] = []
        for i in range(n):
            # 逐块链式哈希（上一块的哈希掺进下一块）。
            h = compute_block_hash(seq.block_token_ids(i), h)
            out.append(h)
        return out

    # 【匹配】先用哈希链在树上找，再把结果校验一遍后转成页号列表。
    def match(self, seq: Sequence, pool: BlockManager) -> list[int]:
        # 只考虑满块（不满的块不能作为前缀，理由同 hash 后端）。
        n_full = seq.num_full_blocks()
        if n_full <= 0:
            return []
        # 一次算完所有块的哈希链 —— 树的接口是按"整条 key"匹配的，不是逐块喂。
        key = self._block_hashes(seq, n_full)
        # 树上找最长前缀，返回的 cached_ids 是该路径上每个节点的页号。
        cached_ids, _ = self.tree.match_prefix(key)
        hit: list[int] = []
        # 树给出的是"结构上匹配"，还必须逐项校验内容与页状态 —— 见下面三个 break。
        for i, bid in enumerate(cached_ids):
            if bid < 0 or i >= n_full:
                break
            # 取这一块的 token 做逐 token 比对（防哈希碰撞）。
            toks = seq.block_token_ids(i)
            block = pool.blocks[bid]
            # 两层校验：哈希要对上、内容也要对上。只信一是标准做法（哈希做索引、内容做判据）。
            if block.hash != key[i] or block.token_ids != toks:
                break
            # 还要确认页当前可用（cached 可共享；used 会让引用计数错乱；两者都不是=已被回收）。
            if bid not in pool.used_ids and bid not in pool.cached_ids:
                break
            hit.append(bid)
        return hit

    # 【发布 + 去重】本文件最复杂的方法，分四段读：
    #     ① 算哈希并写进页对象（第 55-66 行）
    #     ② 插入树，拿到"匹配上的旧长度"与叶子节点（第 68-70 行）
    #     ③ 把重复的页 remap 到树上的规范页（第 73-82 行）
    #     ④ 维护引用计数与序列映射（第 84-88 行）
    def publish(
        self,
        seq: Sequence,
        num_full_blocks: int,
        pool: BlockManager,
        *,
        start: int = 0,
    ) -> None:
        # 与 hash 后端同一套"从中间开始发布要先捞前一块哈希"的处理。
        if num_full_blocks <= start:
            return
        h = -1
        if start > 0:
            h = pool.blocks[seq.block_table[start - 1]].hash
            if h == -1:
                # 逐块算哈希并写到页对象上 —— 树与校验都要用这个哈希。
                start = 0
        for i in range(start, num_full_blocks):
            bid = seq.block_table[i]
            # 顺序：先写页对象的 hash，再统一读出来构造 key。
            # 如果边算边构造 key，链式哈希的依赖关系容易写错。
            toks = seq.block_token_ids(i)
            h = compute_block_hash(toks, h)
            pool.blocks[bid].update(h, toks)

        # 树的 key 是"哈希链"，value 是"对应的页号链" —— 两者一一对应。
        key = [pool.blocks[seq.block_table[i]].hash for i in range(num_full_blocks)]
        value = list(seq.block_table[:num_full_blocks])
        # matched_len 是"与树中已有路径重合的块数"：
        #     >0 说明这条序列有一部分内容树上已经有更早的副本了 → 需要去重（见下）。
        matched_len, last = self.tree.insert(key, value)

        # Dedup: remap onto canonical tree blocks when a longer shared prefix exists.
        # ---- 第 ③ 段：去重 ----
        # 只有在"与已有路径重合"时才需要处理，否则页是全新的，不可能重复。
        if matched_len > 0:
            tree_ids, _ = self.tree.match_prefix(key)
            # 三重上限取 min 是必须的：树的返回长度、序列的实际块数、
            # 重合长度，三者都可能更短，越界会写坏 block_table。
            for i in range(min(matched_len, len(tree_ids), num_full_blocks)):
                old_bid = seq.block_table[i]
                new_bid = tree_ids[i]
                # 已经指向同一页 → 无需搬动。
                if old_bid == new_bid:
                    continue
                # 顺序必须是"先松开旧的、再认领新的"，而且注意这里传的是页号：
                #   release_hit 会把旧页的引用计数减 1（减到 0 就回 free 或 cached），
                #    acquire_hit 给规范页加 1。两边必须成对，只做一半就是引用计数泄漏。
                pool.release_hit(old_bid)
                pool.acquire_hit(new_bid)
                seq.block_table[i] = new_bid

        # ---- 第 ④ 段：引用计数 ----
        # 先把这条序列上一次钉的节点松开（可能它之前跑过，节点换了）。
        old = self._seq_node.pop(seq.seq_id, None)
        if old is not None:
            self.tree.dec_ref(old)  # type: ignore[arg-type]
        # 再钉住这次的叶子节点，并记录进映射表。一松一紧必须成对。
        self.tree.inc_ref(last)
        self._seq_node[seq.seq_id] = last

    # 【失效】页被驱逐重用前调它，把"独占这个页的零引用节点"摘掉。
    def invalidate(self, block_id: int, pool: BlockManager) -> None:
        del pool
        # 与 hash 后端的关键差异：hash 是"删字典条目"（真的删），
        # radix 是"摘树叶"（真的从树上摘掉）。只删条目不摘树叶的实现是假回收，
        # 这里补上了 —— 否则树会一直挂着已经失效的叶子，浪费内存且干扰 LRU。
        self.tree.detach_leaf_for_block(block_id)

    # 序列释放时松开它钉住的节点。与 invalidate 分工不同：
    # invalidate 处理"页被回收"，release_seq 处理"序列结束了"。
    def release_seq(self, seq_id: int) -> None:
        node = self._seq_node.pop(seq_id, None)
        if node is not None:
            self.tree.dec_ref(node)  # type: ignore[arg-type]

    # 【钉住】分配命中前缀之后调用，把命中的节点引用计数加 1。
    # 为什么需要它：树上节点的 ref_count 决定"能不能被回收"。
    # 正在使用这段前缀的序列必须显式钉住，否则它还在跑，节点却可能被驱逐掉。
    def pin_match(self, seq: Sequence, num_cached: int) -> None:
        """allocate 命中前缀后钉住对应节点，避免页还在用就被 evict。"""
        # 没命中任何前缀 → 没有节点可钉。
        if num_cached <= 0:
            return
        key = self._block_hashes(seq, num_cached)
        # 用的是"重新算一遍哈希链 + 再匹配一次"，而不是复用 allocate 时的结果。
        # 代价是多算一次，换来的是不依赖调用顺序（allocate 与 pin_match 之间可能插了别的操作）。
        _ids, node = self.tree.match_prefix(key)
        old = self._seq_node.pop(seq.seq_id, None)
        if old is not None:
            self.tree.dec_ref(old)  # type: ignore[arg-type]
        self.tree.inc_ref(node)
        # 【按树叶回收】给 BlockManager 的专用回收钩子（见 block_manager._ensure_free_page）。
        # 返回的是"已经被摘掉索引、可以安全转 free 的页号"。
        self._seq_node[seq.seq_id] = node

    # 这里没有做"每个叶节点只摘一片"的限制 ——
    # 一个叶节点可能覆盖很多页（压缩树的节点是一段路径），摘它就一次释放很多页。
    # 所以返回值可能多于 num_pages，调用方要能接受这个偏差。
    def reclaim_pages(self, num_pages: int, pool: BlockManager) -> list[int]:
        """丢掉没人引用的冷叶子，返回可以送回 free 的页。一片叶子上的页必须一起走。"""
        # 请求 0 页 → 什么都不做。
        if num_pages <= 0:
            return []
        out: list[int] = []
        guard = 0
        # 每次至少摘 1 片（max(1,...)），否则 num_pages-len(out)==0 时会传 0 进去空转。
        while len(out) < num_pages and guard < 64:
            guard += 1
            # 一片都没摘到 → 树上已经没有被回收的了，退出。
            freed = self.tree.evict(max(1, num_pages - len(out)))
            if not freed:
                break
            for bid in freed:
                if bid in pool.cached_ids:
                    out.append(bid)
        return out
