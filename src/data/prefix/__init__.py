# ============================================================================
# 【本文件是做什么的】前缀缓存后端的选择点：三种实现（none / hash / radix）
# 在这里按配置选一个。KVManager 只拿到一个满足约定接口的对象，不关心是哪种。
#
# 【三种后端的分工】
#     none  —— 不做前缀复用（调试、或对比实验用）
#     hash  —— 按块内容哈希建索引：实现简单，命中率略低
#     radix —— 前缀树：命中更精确（能复用部分匹配的前缀），但回收逻辑更复杂
#
# 【为什么用鸭子类型而不是抽象基类】三种实现只要满足同样几个方法即可，
# 少一层继承，以后加新后端也不用改基类。
#
# 【⚠️ 一个容易踩的点】后端名写错不会报错：下面的工厂有兜底分支，
# 会静默返回 Noop（等于悄悄关掉前缀缓存）。所以 `EngineConfig.__post_init__`
# 里先做了一次名字校验 —— 这类"静默降级"必须靠上游拦住。
# ============================================================================

from __future__ import annotations

from typing import Literal

from .hash_index import HashPrefixIndex
from .noop import NoopPrefixIndex
from .radix_index import RadixPrefixIndex

# KVManager 按 config.prefix_backend 在这里选实现。
PrefixBackend = Literal["none", "hash", "radix"]


# 工厂函数。返回的对象只要满足那四个方法的契约即可（本项目用的是鸭子类型，没有抽象基类）。
def make_prefix_index(backend: PrefixBackend):
    # hash 后端：按块内容哈希建索引，实现最简单。
    if backend == "hash":
        return HashPrefixIndex()
    # radix 后端：前缀树，能复用部分匹配的前缀，回收也更复杂。
    if backend == "radix":
        return RadixPrefixIndex()
    # 兜底 = Noop（不做前缀复用）。名字写错也会落到这里，所以配置层要先校验。
    return NoopPrefixIndex()


# 对外契约：后端类型与工厂函数。
__all__ = ["PrefixBackend", "make_prefix_index"]
