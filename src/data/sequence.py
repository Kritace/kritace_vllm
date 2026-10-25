# ==============================================================================
# 一条请求在 CPU 上的状态。
#
# Engine.add_request / Scheduler / KVManager 读写本对象。
# 发给 Worker 前由 ScheduledBatch.to_wire() 收成 SeqSchedule。
# ==============================================================================


# 三态：等 KV、正在跑、已结束。调度器的所有判断都基于它。

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import count

from ..sampling_params import SamplingParams

# 全局递增的序列 id。进程级计数器，多引擎实例会共享它（本项目只有单引擎，够用）。
_SEQ_IDS = count(1)

# ==============================================================================
# 一条请求在 CPU 上的状态。
#
# Engine.add_request / Scheduler / KVManager 读写本对象。
# 发给 Worker 前由 ScheduledBatch.to_wire() 收成 SeqSchedule。
# ==============================================================================


# 三态：等 KV、正在跑、已结束。调度器的所有判断都基于它。
class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


# 【逻辑位置 → 物理槽位】给一整段 [start, end) 批量换算，而不是逐 token 调。
def slots_for_range(
    block_table: list[int],
    start: int,
    end: int,
    block_size: int,
) -> list[int]:
    """[start, end) 每个 token 对应的物理 slot。按块 stretch range，避免逐 token 做 // 和 %。"""
    # 长度为零的段直接返回空（构造空 range 也没问题，提前返回更省）。
    n = end - start
    # 起始点所在的块号与块内偏移。
    if n == 0:
        return []
    b = start // block_size
    off = start - b * block_size
    # 快路径：整段落在**同一个块**里 → 一次 range 就够，不需要循环。
    # 这是 decode 以及小 chunk 走的路径，也是最常见的。
    if off + n <= block_size:
        base = block_table[b] * block_size
        return list(range(base + off, base + off + n))
    # 慢路径：跨多个块，只能逐块拼。
    slots: list[int] = []
    # 最后一块的块号，作为循环边界。
    last = (end - 1) // block_size
    # lo/hi 的边界处理是这段的关键（本项目踩过坑）：
    #   第一块要从 start 开始（不是块首），最后一块到 end 结束（不是块尾），
    #    中间的块才是整块。写成"每块都取整块"就会多算 token。
    while b <= last:
        lo = start if b == start // block_size else b * block_size
        hi = end if b == last else (b + 1) * block_size
        base = block_table[b] * block_size
        slots.extend(range(base + lo - b * block_size, base + hi - b * block_size))
        b += 1
    return slots


# 一条请求。字段很多，但可以按上面的三个角色分类看。
@dataclass
class Sequence:
    # ---------- 角色①：内容 ----------
    # token_ids 是唯一的"真相来源"，其余长度类字段都是从它推出来的。
    token_ids: list[int]
    block_size: int
    seq_id: int = 0
    status: SequenceStatus = SequenceStatus.WAITING
    block_table: list[int] = field(default_factory=list)
    num_cached_tokens: int = 0
    num_scheduled_tokens: int = 0
    # 本步 prefill 是否已经覆盖完整 prompt，需要采样。
    need_sample: bool = False
    # is_prefill 这个字段语义比较模糊（v2 遗留），实际判断以 need_sample 为准。
    is_prefill: bool = True
    num_prompt_tokens: int = 0
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    max_tokens: int = 128
    ignore_eos: bool = False
    last_token: int = 0

    # --- 投机解码（v6）---
    # 本步由草稿生成的候选 token；为空表示退化为普通 decode。
    spec_draft_ids: list[int] = field(default_factory=list)
    # 验证后被接受的草稿数 j；本步实际产出 j + 1 个 token。
    spec_accepted: int = 0
    # 待写回的 token（j 个草稿 + 1 个修正 token）。
    # _update_batch 优先消费它，这样"一步产出多个 token"就不必改方法签名。
    spec_emitted: list[int] = field(default_factory=list)

    # --- 零开销调度（ZOS）---
    # 只在「本步要跑、但上一步的结果还没回灌」时被调度器写上；每步 schedule 前都会清回 -1，
    # 所以它们不是持久状态，只允许 compute 层读。
    #   zos_remap >= 0 ：本行是「在途行」——输入 token 此刻还不知道，要等 GPU 端
    #                    从上一步的采样结果里按这个下标取（见 compute/batch.py 的 build_decode）。
    #   zos_start_pos  ：在途行的逻辑位置。CPU 侧账本落后一步，所以它是「已缓存数 + 1」。
    zos_remap: int = -1
    zos_start_pos: int = -1

    # 唯一的构造入口。参数用 keyword-only（*），避免调用方把顺序写错 ——
    # 这条序列的字段太多，位置参数很容易张冠李戴。
    @classmethod
    def from_prompt(
        cls,
        token_ids: list[int],
        *,
        block_size: int,
        sampling: SamplingParams | None = None,
    ) -> Sequence:
        sp = sampling or SamplingParams()
        ids = copy(token_ids)
        # 空 prompt 直接拒绝（空列表在后续所有长度计算里都是陷阱）。
        if not ids:
            raise ValueError("empty prompt")
        # seq_id 从全局计数器取，保证唯一。它是索引里登记/释放的键。
        return cls(
            token_ids=ids,
            block_size=block_size,
            seq_id=next(_SEQ_IDS),
            num_prompt_tokens=len(ids),
            last_token=ids[-1],
            temperature=sp.temperature,
            top_p=sp.top_p,
            top_k=sp.top_k,
            min_p=sp.min_p,
            max_tokens=sp.max_tokens,
            ignore_eos=sp.ignore_eos,
        )

    # ---------- 让 Sequence 像列表一样用 ----------
    # 实现 __len__ / __getitem__ 之后，len(seq) 与 seq[i] 都能直接用，
    # 调度器和页池管理里大量依赖这两个操作。
    def __len__(self) -> int:
        # 总 token 数（prompt + 已生成）。
        return len(self.token_ids)

    # 注意 __getitem__ 直接透传到 token_ids —— 这意味着切片返回的是 list 而不是 Sequence。
    def __getitem__(self, key):
        return self.token_ids[key]

    # ---------- 角色③ + 派生长度 ----------
    # 下面这一串都是只读派生属性，不存状态，每次现算。
    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    # 结束判断用状态而不是长度 —— 因为"达到 max_tokens"和"采到 EOS"是两种结束方式，
    # 但都体现为状态变成 FINISHED。
    @property
    def is_finished(self) -> bool:
        # 已生成 token 数 = 总长 − prompt 长。注意是"生成"不是"已缓存"。
        return self.status == SequenceStatus.FINISHED

    # 需要多少页 = 总长按块大小向上取整。它用的是**总长**（含未算的部分），
    # 所以调度器能提前知道"就算全生成完，页够不够"。
    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    # 减 1 是因为块号从 0 开始：(num_blocks-1) 才是最后一块的编号。
    @property
    def num_blocks(self) -> int:
        n = self.num_tokens
        if n == 0:
            return 0
        return (n + self.block_size - 1) // self.block_size

    # 先判 num_blocks==0，否则下面的乘法会算出一个负数。
    @property
    def last_block_num_tokens(self) -> int:
        if self.num_blocks == 0:
            return 0
        # 已填满的块数。**只有满块才能进前缀索引**，所以这个函数是发布逻辑的前置条件。
        # 用的是整除（向下取整），不是向上取整 —— 这是"不满的块不算"的体现。
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    # 取第 i 块的 token id 列表（末块可能不满）。
    def num_full_blocks(self) -> int:
        return self.num_tokens // self.block_size

    # 追加一个采样出来的 token。
    # 必须同时更新 last_token —— decode 的输入是"最后一个 token"，
    # 忘了更新就会一直重复输入同一个 token（输出变成循环重复）。
    def block_token_ids(self, i: int) -> list[int]:
        assert 0 <= i < self.num_blocks
        start = i * self.block_size
        end = min(start + self.block_size, self.num_tokens)
        return self.token_ids[start:end]

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        self.last_token = token_id
