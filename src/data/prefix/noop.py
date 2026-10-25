# ============================================================================
# 【本文件是做什么的】前缀缓存的空实现（`prefix_backend=none`）：
#     match 永远返回空列表、publish / invalidate / release_seq 什么都不做。
#
# 【为什么需要它】三种后端（none / hash / radix）要满足同一套接口，
#     KVManager 只拿到"一个满足约定的对象"，不关心是哪种。
#     有 none 这个实现，才能：
#         ① 用配置关掉前缀缓存做对照实验（性能对比的一项变量）；
#         ② 让"没有前缀缓存"这条路径不写满 `if backend is None` 的分支。
#
# 【为什么方法体是空的而不是 raise】调用方（KVManager）无需知道自己拿的是哪种后端，
#     关掉缓存时它照样调 publish —— 空实现让调用点保持统一。
#     `del seq, num_full_blocks, ...` 只是显式声明这些参数被有意忽略，
#     避免被 linter 报"未使用参数"。
# ============================================================================

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..block_manager import BlockManager
    from ..sequence import Sequence

# prefix_backend=none：match 永远落空，publish / invalidate 什么都不做。


class NoopPrefixIndex:
    # 永远不命中 —— 这就是"关闭前缀缓存"的全部含义。
    def match(self, seq: Sequence, pool: BlockManager) -> list[int]:
        return []

    # 所有写操作都丢弃：没有索引，也就没有需要维护的内容。
    def publish(
        self,
        seq: Sequence,
        num_full_blocks: int,
        pool: BlockManager,
        *,
        start: int = 0,
    ) -> None:
        # 显式声明这些参数被有意忽略（见文件头说明）。
        del seq, num_full_blocks, pool, start
        return

    def invalidate(self, block_id: int, pool: BlockManager) -> None:
        return

    def release_seq(self, seq_id: int) -> None:
        return
