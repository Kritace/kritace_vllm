# ==============================================================================
# 压缩 radix 树：key 是链式块哈希，value 是逻辑页号。
#
# RadixPrefixIndex 用来 match / insert / 按叶子 LRU evict。
# 节点带 ref_count；只有 ref=0 的叶子才能被摘掉。
# ==============================================================================


# 求两个序列的公共前缀长度。树的"压缩"与"分裂"全靠它。
# 用 while 手写而不是 zip+循环：这是热路径（每次匹配/插入都会调）。

from __future__ import annotations

import heapq
import time
from collections.abc import Sequence as AbcSequence

# ==============================================================================
# 压缩 radix 树：key 是链式块哈希，value 是逻辑页号。
#
# RadixPrefixIndex 用来 match / insert / 按叶子 LRU evict。
# 节点带 ref_count；只有 ref=0 的叶子才能被摘掉。
# ==============================================================================


# 求两个序列的公共前缀长度。树的"压缩"与"分裂"全靠它。
# 用 while 手写而不是 zip+循环：这是热路径（每次匹配/插入都会调）。
def _prefix_len(a: AbcSequence[int], b: AbcSequence[int]) -> int:
    n = 0
    m = min(len(a), len(b))
    while n < m and a[n] == b[n]:
        n += 1
    return n


# 树节点。
class RadixNode:
    # __slots__ 省内存：树可能有几十万个节点，每个节点省一个 __dict__ 很可观。
    # 代价是不能动态加属性 —— 加字段必须同步改这一行，否则赋值静默失败。
    __slots__ = ("children", "parent", "key", "value", "ref_count", "access_time")

    # key 和 value 用 tuple 而不是 list：节点一旦插入树，它的 key/value
    # 就不该被原地修改（改了会让 children 的分叉依据失效）。tuple 从类型上阻止这件事。
    def __init__(
        self,
        *,
        parent: RadixNode | None = None,
        key: tuple[int, ...] = (),
        value: tuple[int, ...] = (),
        # 分叉依据：用路径的第一段哈希做键。所以"同一起点的两条路径"天然共享父节点。
        ref_count: int = 0,
    ) -> None:
        # 从父节点到本节点的路径。注意它可能是一段（压缩）而不是一段哈希。
        self.children: dict[int, RadixNode] = {}
        # 与 key 一一对应的页号。长度必须与 key 相等 —— 这是本文件最重要的一条不变式。
        self.parent = parent
        # 含义是"有多少条活着的序列经过我"（含经过的子路径），不是"有多少页"。
        self.key = key
        # 单调时钟：用 time.monotonic() 而不是 time.time()，不受系统时间调整影响。
        self.value = value
        self.ref_count = ref_count
        self.access_time = time.monotonic()

    # 实现 __lt__ 是为了能被丢进 heapq 做 LRU（堆按访问时间排序）。
    def __lt__(self, other: RadixNode) -> bool:
        return self.access_time < other.access_time


# 树本体。root 是一个"占位节点"，它自己不对应任何路径。
class RadixTree:
    # 构造：根节点引用计数固定为 1（永远不回收），并建好反向索引。
    def __init__(self) -> None:
        # root 的 ref_count=1 是刻意的：这样 inc_ref/dec_ref 循环到 root 就停，
        # 不需要判空指针。root 永不被摘。
        self.root = RadixNode(ref_count=1)
        self._block_to_node: dict[int, RadixNode] = {}

    # 反向索引的查询接口。BlockManager 用它判断"这个页在树上有没有主人"。
    def contains_block(self, block_id: int) -> bool:
        return block_id in self._block_to_node

    # 【最长前缀匹配】从根往下走，能走多深走多深。
    # 返回值：(路径上所有节点的 page 列表, 停止处的节点)。
    def match_prefix(self, key: list[int]) -> tuple[list[int], RadixNode]:
        # 空 key 直接返回根。
        if not key:
            return [], self.root
        # 逐层下行。每行一步就把 key 缩短（key = key[pl:]，见第 68 行）。
        node = self.root
        # values 是沿途累积的页号。注意它是"拼起来"的，不是某一层的。
        values: list[int] = []
        # 分叉依据永远是"剩余 key 的第一段"。
        child_key = key[0]
        # 走到无子节点可走就停 —— 这就是"最长"的体现。
        while key and child_key in node.children:
            child = node.children[child_key]
            child.access_time = time.monotonic()
            # 公共前缀长度。
            pl = _prefix_len(child.key, key)
            # 关键分支：pl < len(child.key) 说明在当前节点内部就分叉了。
            # 这时必须把它劈成两半（见 _split），否则这一段路径没法表达两个不同的走向。
            if pl < len(child.key):
                new_node = self._split(child, pl)
                # 劈开之后，本节点（前半段）就是匹配到的部分，返回它。
                values.extend(new_node.value)
                return values, new_node
            # 整段都对上 → 把这一段的页号收进结果，继续往下走。
            values.extend(child.value)
            node = child
            key = key[pl:]
            child_key = key[0] if key else -1
        return values, node

    def insert(self, key: list[int], value: list[int]) -> tuple[int, RadixNode]:
        """插入 key→页号。返回（已有公共前缀长度, 叶子）。"""
        if not key:
            return 0, self.root
        node = self.root
        child_key = key[0]
        matched = 0
        while key and child_key in node.children:
            child = node.children[child_key]
            child.access_time = time.monotonic()
            pl = _prefix_len(child.key, key)
            matched += pl
            key = key[pl:]
            value = value[pl:]
            child_key = key[0] if key else -1
            # 同样要在节点内部劈开。
            if pl < len(child.key):
                node = self._split(child, pl)
                break
            node = child
        # 走到这里 key 还有剩余 → 说明树里没有这条路径，需要新建节点接上去。
        last = node
        if key:
            last = self._add(node, tuple(key), tuple(value))
        return matched, last

    # 【引用计数】加 1。一路爬到根，见文件头说明。
    def inc_ref(self, node: RadixNode) -> None:
        while node is not self.root:
            node.ref_count += 1
            node = node.parent or self.root

    # 减 1。与 inc_ref 必须严格配对，任何一处少减/多减都会导致节点永远回收不掉，
    # 或反过来被提前摘掉（前者浪费内存，后者是正确性错误）。
    def dec_ref(self, node: RadixNode) -> None:
        while node is not self.root:
            node.ref_count -= 1
            node = node.parent or self.root

    # 因"页被回收"而摘节点。
    # 三重检查缺一不可：节点存在、不是根、引用计数为 0。
    # 少了最后一条就会把正在被使用的路径摘掉 → 别的序列还在跑却没了索引 → 缓存错乱。
    def detach_leaf_for_block(self, block_id: int) -> list[int]:
        """页被池子丢掉时，若所属节点无人引用，整段从树上摘掉。"""
        node = self._block_to_node.get(block_id)
        if node is None or node is self.root or node.ref_count > 0:
            return []
        freed = list(node.value)
        self._remove(node)
        return freed

    # 【LRU 淘汰】按访问时间从最久的叶子开始摘，直到凑够 num_slots 个页。
    # 它和 detach_leaf_for_block 的区别：那个是"某个页要失效了，去树里找它的主人"，
    # 这个是"页池缺页了，主动挑几个最冷的叶子牺牲"。
    def evict(self, num_slots: int) -> list[int]:
        """按 access_time 丢掉无人引用的叶子，直到凑够页数。"""
        # 每次都重新收集叶子并建堆（O(n)），而不是增量维护。
        # 这是刻意的取舍：增量维护堆与树的同步很容易出错，
        # 而淘汰本身不频繁（只在页池真的要满时发生）。
        leaves = self._leaves()
        heapq.heapify(leaves)
        # 收集被摘掉的页号。
        freed: list[int] = []
        # 循环条件：凑够片数 或 堆空。
        while len(freed) < num_slots and leaves:
            node = heapq.heappop(leaves)
            # 堆里可能有过期的条目（节点已经被别人摘了，或引用计数又涨上去了），
            # 所以弹出来还要再检查一遍，不能直接用。
            if node is self.root or node.ref_count > 0:
                continue
            # 一个叶子可能覆盖多个页 → 一次摘掉全部（它们必须一起走，见文件头）。
            freed.extend(node.value)
            # 摘掉孩子之后，如果父节点也没了孩子，父节点就"变成叶子"了 ——
            # 下面把它重新压进堆，这样"压缩"的效果才是充分的（不会留一串单孩子节点）。
            parent = node.parent
            self._remove(node)
            if parent is not None and parent is not self.root and not parent.children:
                heapq.heappush(leaves, parent)
        return freed

    # ---------- 反向索引的同步 ----------
    # 这两个必须与树的结构改动成对调用。
    # 漏了 _index → 页失效时找不到节点；漏了 _unindex → 摘掉的节点还挂在索引里。
    def _index(self, node: RadixNode) -> None:
        for bid in node.value:
            self._block_to_node[bid] = node

    def _unindex(self, node: RadixNode) -> None:
        for bid in node.value:
            if self._block_to_node.get(bid) is node:
                del self._block_to_node[bid]

    # 收集所有叶子（用于 LRU）。它走的是整棵树，O(n)。
    def _leaves(self) -> list[RadixNode]:
        out: list[RadixNode] = []

        # 递归遍历。根节点不算叶子（它不承载任何路径）。
        def walk(n: RadixNode) -> None:
            if not n.children:
                if n is not self.root:
                    out.append(n)
                return
            for c in n.children.values():
                walk(c)

        # 从根开始走。
        walk(self.root)
        return out

    # 从树上摘掉一个节点。它不改引用计数，只改结构 ——
    # 调用方要先确认这个节点确实可以被摘（零引用）。
    def _remove(self, node: RadixNode) -> None:
        if node.parent is None or not node.key:
            return
        self._unindex(node)
        node.parent.children.pop(node.key[0], None)

    # 新建一个孩子节点并接上去（插入路径的尾部）。
    def _add(
        self, parent: RadixNode, key: tuple[int, ...], value: tuple[int, ...]
    ) -> RadixNode:
        node = RadixNode(parent=parent, key=key, value=value)
        parent.children[key[0]] = node
        self._index(node)
        return node

    # 【分裂】压缩树最核心的操作：把一个节点在 split_at 处劈成两半。
    #
    # 分裂前：  parent → node (key 长度 10, 页 1..10)
    # 分裂后：  parent → new (key 长度 4, 页 1..4)
    #                       └→ node (key 长度 6, 页 5..10)
    #
    # 四件事必须全做对，否则树会损坏：
    #   ① 前半段成为新节点，后半段留在原节点（key/value 都要切）
    #    ② 原节点改成新节点的孩子
    #    ③ 新节点要接到原父节点下面（别忘，漏了就是整棵子树从树上掉下来）
    #    ④ 反向索引要重新指向（原索引指向 node，现在前半段的页归 new 了）
    #
    # 引用计数直接继承（ref_count=node.ref_count），不重新计算 ——
    # 因为分裂不改变"有多少序列经过这段路径"。
    def _split(self, node: RadixNode, split_at: int) -> RadixNode:
        self._unindex(node)
        new_node = RadixNode(
            parent=node.parent,
            key=node.key[:split_at],
            value=node.value[:split_at],
            ref_count=node.ref_count,
        )
        new_node.children = {node.key[split_at]: node}
        node.parent = new_node
        node.key = node.key[split_at:]
        node.value = node.value[split_at:]
        if new_node.parent is not None and new_node.key:
            new_node.parent.children[new_node.key[0]] = new_node
        self._index(new_node)
        self._index(node)
        return new_node
