# ==============================================================================
# n-gram 草稿：不引入第二个模型，直接从序列自身的历史里找可复用的后缀。
#
# 命中条件：当前结尾的 n 个 token 之前出现过。
# 适用：摘要 / 代码 / few-shot 这类含重复片段的输入；
#      对无重复的自然文本基本无效。
#
# 它的真正价值是"零成本验证整条投机链路"—— 不需要第二个模型，
# 就能把 KV 账本、接受判定、回滚逻辑全部跑通，
# 之后再换成 draft 模型时，上层代码一行都不用改。
# ==============================================================================
from __future__ import annotations

__all__ = ["NgramDrafter"]


class NgramDrafter:
    """基于 n-gram 后缀匹配的草稿生成器。

    ``max_n`` 越大匹配越精确、命中率越低；``min_n`` 是允许的最短后缀。
    匹配时从 ``max_n`` 递减，取第一个在历史上出现过的后缀。

    示例（``max_n=3``）::

        history = [5, 9, 2, 7, 5, 9, 2]
        propose(history, k=2) -> [7, 5]
        理由：后缀 [5, 9, 2] 上一次出现在 index 0，后继是 7；
              接上 7 后后缀 [9, 2, 7] 上次出现在 index 1，后继是 5。

    ⚠️ 本实现是朴素扫描（每次匹配 O(n·L)），只为把链路跑通。
    生产环境应预建 n-gram → 后继列表的索引，把匹配降到 O(n)。
    """

    # 【两个参数的取舍】max_n 大 ⇒ 匹配更精确但命中率更低（长后缀更难重复出现）；
    # min_n 是兜底：极端情况下哪怕只匹配 1 个 token 也可以给出一个草稿。
    # 实测里 max_n=4 是重复文本与自然文本之间的折中（v6 的默认值就是这个）。
    def __init__(self, *, max_n: int = 4, min_n: int = 1) -> None:
        # 参数自检：min_n 至少为 1，且不能大于 max_n（否则下面的 range 是空的、永远匹配不上）。
        if min_n < 1 or max_n < min_n:
            raise ValueError(
                f"需要 1 <= min_n <= max_n, 得到 min_n={min_n} max_n={max_n}"
            )
        self.max_n = max_n
        self.min_n = min_n

    # 【草稿是"边匹配边延长"的】每匹配出一个 token 就把它接到 history 尾巴上，
    # 于是下一轮匹配能看到刚刚产生的这个 token —— 这才是"在重复片段里连续延长"。
    # 如果只匹配一次、然后把同一个后继重复 k 遍，草稿质量会差很多。
    def propose(self, token_ids: list[int], k: int) -> list[int]:
        """返回最多 ``k`` 个候选 token；找不到任何匹配时返回空列表。

        生成方式是"匹配一个 → 接上 → 再匹配一个"，因此草稿自身也参与匹配，
        可以在重复片段里连续延长。
        """
        # 两个前置条件：k>0（否则不必生成），以及历史长度至少 min_n+1
        # （要匹配 n 个历史 token 并取出它的后继，至少得有 n+1 个 token）。
        if k <= 0 or len(token_ids) < self.min_n + 1:
            return []

        # 返回给上层的草稿（最多 k 个；匹配不上就提前结束，长度可以不足 k）。
        out: list[int] = []
        # 复制一份 history 用于**追加**匹配到的草稿 token。
        # 不能直接改调用方传进来的 list（调用方的序列状态不能被草稿影响）。
        history = list(token_ids)
        # 每轮尝试「匹配一个 token → 接上 → 再匹配下一个」。
        for _ in range(k):
            # 取不到后继（历史里没有这个后缀）就停：宁可少给草稿，也不要瞎猜一个。
            nxt = self._match_once(history)
            if nxt is None:
                break
            # 一个个累积成草稿列表。
            out.append(nxt)
            # ★ 关键一步：把草稿接进 history，让下一轮匹配能利用它。
            # 没有这一行，多 token 草稿就会退化成"同一个后继重复 K 次"。
            history.append(nxt)
        return out

    # ============================================================================
    # 取一个后继：用**最长**的可匹配后缀。
    # 从 max_n 递减到 min_n，第一个在历史上出现过的后缀就采用它的后继 ——
    # 后缀越长，条件越强、命中的概率越低，但一旦命中，"下一个 token 就该是它"的
    # 可信度也越高。所以优先长后缀。
    # ============================================================================
    def _match_once(self, history: list[int]) -> int | None:
        """用最长的可匹配后缀取一个后继 token，取不到返回 None。"""
        # 递减遍历：range(max_n, min_n - 1, -1) 生成 max_n, max_n-1, ..., min_n。
        # 这样第一个成功的匹配就是"最长的那个后缀"。
        for n in range(self.max_n, self.min_n - 1, -1):
            if len(history) < n + 1:
                continue
            # 取当前序列结尾的 n 个 token 作为匹配键。
            key = tuple(history[-n:])
            # 在历史里找这个键最后一次出现的位置。
            # 用"最后一次"而不是第一次：离当前越近的重复片段，越可能延续同样的后续
            # （真实文本里"最近用过什么"比"最早出现过什么"更相关）。
            pos = self._last_occurrence(history, key)
            if pos is not None:
                # 命中：紧跟在那个位置的键之后的下一个 token 就是草稿。
                # 注意 pos 是**起点**，所以后继下标是 pos + n。
                return history[pos + n]
        return None

    @staticmethod
    def _last_occurrence(history: list[int], key: tuple[int, ...]) -> int | None:
        """找 ``key`` 在 history 中最后一次出现的起始下标。

        搜索范围刻意排除序列末尾 —— 否则当前后缀会匹配到自己，
        取出的"后继"根本不存在。
        """
        # 键的长度 —— 也就是「要跳过多少个位置才可能读到后继」。
        n = len(key)
        # ★ 搜索上界刻意排除了末尾：最晚只搜到 len(history)-n-1。
        # 否则当前后缀会匹配到**它自己**，然后 pos+n 直接越界（或者取到一个不存在的位置），
        # 这是这类"用历史预测未来"的实现最经典的越界来源。
        for i in range(len(history) - n - 1, -1, -1):
            if tuple(history[i : i + n]) == key:
                # 倒着找 → 遇到的第一处就是最后一次出现。
                return i
        return None
