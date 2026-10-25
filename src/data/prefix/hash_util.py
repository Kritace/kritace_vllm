# ============================================================================
# 【本文件是做什么的】前缀缓存的**块哈希**：把一段 token 变成一个整数键。
#
# 【为什么要带 prefix 参数】这是前缀缓存的全部要点：
#     哈希不只覆盖本块的 token，还要把**前一块的哈希**混进来（prefix 就是它）。
#     于是"同样的 token、但前面内容不同"会得到不同哈希 ——
#     否则两个只是局部相同的请求会被误判成同一个前缀，复用出错。
#
# 【为什么用 struct + numpy】token id 是 int32 序列，要按**二进制**喂给哈希
#     （不能先转成文本再哈希：那样慢且可能出现编码歧义）。
#     struct.pack 负责把 C 级别的整数序列打包成字节串，喂给 Python 的哈希。
#
# ⚠️ 哈希只用来做**索引**，不承担正确性：真正的复用还要比对块内容
#   （见 radix_index.py / hash_index.py 的匹配逻辑）。把它当"相等性判据"会出错 ——
#   哈希碰撞会静默地复用错块。
# ============================================================================

from __future__ import annotations

import struct

import numpy as np

try:
    import xxhash
except ImportError:  # pragma: no cover
    xxhash = None  # type: ignore

# 整块 token 的链式哈希：h_i = H(本块 token | h_{i-1})，h_{-1} = -1。
# HashPrefixIndex / RadixPrefixIndex 用它当块的身份。


def compute_block_hash(token_ids: list[int] | np.ndarray, prefix: int = -1) -> int:
    if xxhash is not None:
        h = xxhash.xxh64()
        # 前缀的字节序/符号必须固定（little + unsigned）：
        #   换个平台或换种编码方式得到的就是另一个哈希，索引会全部对不上。
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little", signed=False))
        # 统一成 int32 再取字节串 —— 这是"把列表变成一个确定字节序列"的关键一步。
        arr = np.asarray(token_ids, dtype=np.int32)
        h.update(arr.tobytes())
        return int(h.intdigest())
    import hashlib

    h = hashlib.blake2b(digest_size=8)
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little", signed=False))
    # 末尾按 little + unsigned 转回整数，与 xxhash 路径保持一致的口径。
    h.update(struct.pack(f"<{len(token_ids)}i", *token_ids))
    return int.from_bytes(h.digest(), "little", signed=False)
