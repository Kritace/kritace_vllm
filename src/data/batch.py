# ============================================================================
# 【本文件是做什么的】控制面与计算面之间的 CPU 数据结构。
#
# 【它在一条请求的路径上处在哪】
#     Scheduler.schedule  →  ScheduledBatch（本进程内，直接持 Sequence 引用）
#         → to_wire()     →  SchedulerOutput（纯数据，可跨进程 msgpack）
#             → Worker / ModelRunner 的 build_decode / build_prefill
#     Scheduler.update    →  EngineCoreOutput（增量 token）→ 回到门面 Engine
#
# 【为什么要有 to_wire 这一步】单进程时，调度器给的 Sequence 可以直接被
# 计算层读；多进程下 Sequence 里带着整条 token 历史、块表、引用计数等，
# 既大又不该跨进程传。所以在这里裁剪出前向真正需要的字段，
# 顺带把"能共享的就共享"（块表不拷贝，见下）。
#
# 【⚠️ 两个容易踩的点】
#     ① `block_table` 是共享引用而不是拷贝 —— GPU 侧靠 seq_id 做行缓存、
#        增量 H2D。一旦哪一层就地改了它，增量同步的判断就会失效。
#     ② SeqSchedule / SchedulerOutput 是 msgspec 结构体，字段顺序即编码顺序，
#        跨进程协议，别随意调整字段次序或类型。
#
# 【读这个文件的顺序】
#     ScheduledBatch（调度器的产出）+ to_wire（裁剪）
#       → _decode_schedule / _prefill_schedule（两种裁剪方式）
#       → SeqSchedule / SchedulerOutput / EngineCoreOutput（三个传输结构）
# ============================================================================

from __future__ import annotations

import os
from dataclasses import dataclass, field

import msgspec

from .sequence import Sequence

# 诊断开关：KRITACE_SPEC_DEBUG=1 时打印组包前的序列状态。
# 用途：投机序列的 KV 账本一旦被破坏（例如被抢占后账本清零、草稿却还挂着），
# 组包侧的断言会失败；此时必须能看到序列的身份与账本才能定位。
_SPEC_DEBUG = bool(os.environ.get("KRITACE_SPEC_DEBUG"))

# ==============================================================================
# 控制面与计算面之间的 CPU 数据结构。
#
# Scheduler.schedule 产出 ScheduledBatch（仍持有本进程 Sequence）。
# Executor.run 调用 to_wire()，得到可 msgspec 编码的 SchedulerOutput，
# 交给 Worker / ModelRunner 的 build_decode、build_prefill。
# Scheduler.update 产出 EngineCoreOutput，经 Client 回到 Engine。
#
# GPU 上的静态输入缓冲在 compute/batch.py。
# ==============================================================================


# 调度器 → 执行器之间唯一的接口结构。刻意做成"纯数据"：执行层不需要懂 Sequence 内部。
@dataclass
class ScheduledBatch:
    """本步要跑的 prefill / decode 序列。Scheduler 与 EngineCore 使用；Worker 只吃 to_wire()。"""

    # 本步要跑的两类序列：prefill（含 chunked）与 decode。
    prefills: list[Sequence] = field(default_factory=list)
    decodes: list[Sequence] = field(default_factory=list)

    # 引擎用它判断"这步是不是白跑了"（是就要累计空转次数，见 Engine._MAX_IDLE_STEPS）。
    def is_empty(self) -> bool:
        return not self.prefills and not self.decodes

    def to_wire(self) -> "SchedulerOutput":
        """抽出本步前向需要的字段，不把整条 token 历史发给 Worker。"""
        return SchedulerOutput(
            prefills=[_prefill_schedule(s) for s in self.prefills],
            decodes=[_decode_schedule(s) for s in self.decodes],
        )


# 四个采样参数打包成一个元组 —— 三个 schedule 函数都要用。
def _sampling_fields(s: Sequence) -> tuple[float, float | None, int | None, float | None]:
    return s.temperature, s.top_p, s.top_k, s.min_p


def _decode_schedule(s: Sequence) -> "SeqSchedule":
    """decode：一个新 token、缓存长度、块表（与 Sequence 同一 list，不拷贝）。

    投机解码下本步输入的 token 是 ``[last_token, *spec_draft_ids]``，长度 $K+1$；
    但**序列本身只长了 1**（草稿尚未被接受，还没写进 token_ids），
    所以下面的不变量不变：``len(token_ids) == num_cached_tokens + 1``。
    """
    # 两条路，区别只在「这一步的输入 token 现在知不知道」：
    #   普通行（zos_remap < 0）：CPU 账本已回灌，last_token / num_cached_tokens 都是准的。
    #   在途行（zos_remap >= 0）：CPU 还没拿到上一步的采样结果，token id 发不出去，
    #       位置要按「已缓存数 + 1」乐观推进，token id 留空、由 GPU 端 gather 补上。
    temperature, top_p, top_k, min_p = _sampling_fields(s)
    if s.zos_remap >= 0:
        return SeqSchedule(
            seq_id=s.seq_id,
            block_table=s.block_table,
            # 乐观位置：写这个 token 时的逻辑位置（= 已缓存数 + 1）。
            num_cached_tokens=s.zos_start_pos,
            # 空 list = 「本步的输入 token 由 GPU 自己从上一批的采样结果里取」。
            token_ids=[],
            need_sample=False,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            remap=s.zos_remap,
        )
    # 不变式：decode 时 token 历史长度 = 已缓存数 + 1（这个新 token 还没写进 KV）。
    assert len(s.token_ids) == s.num_cached_tokens + 1, (
        f"decode invariant: len(token_ids)={len(s.token_ids)} vs "
        f"num_cached_tokens+1={s.num_cached_tokens + 1}"
    )
    return SeqSchedule(
        seq_id=s.seq_id,
        # 块表与 Sequence 共享同一个 list（不拷贝）—— GPU 侧靠 seq_id 做行缓存增量 H2D。
        block_table=s.block_table,
        num_cached_tokens=s.num_cached_tokens,
        token_ids=[s.last_token, *s.spec_draft_ids],
        need_sample=False,
        spec_draft_ids=list(s.spec_draft_ids) if s.spec_draft_ids else None,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
    )


def _prefill_schedule(s: Sequence) -> "SeqSchedule":
    """prefill：只发本步 scheduled 的 token 切片，不含已缓存前缀和尚未调度的尾巴。

    ⚠️ 投机验证的序列走的也是这里 —— 只有 prefill 通道的 varlen attention
    能一次算 K+1 个 query，所以调度器把带草稿的序列挪进了 prefills。
    但它的本步输入不是"已调度切片"：草稿尚未被接受、还没写进 ``token_ids``，
    所以 ``token_ids[start:end]`` 只会切出 last_token 一个元素，前向只算
    1 个位置，验证结果的第 2..K+1 位会留在采样缓冲区的初值 0 上，
    然后被当成"额外 token"发出去（现象：序列在第 2 个 token 就开始偏离）。
    必须显式拼成 ``[last_token, *草稿]``，与 _decode_schedule 保持一致。
    """
    if s.spec_draft_ids:
        if _SPEC_DEBUG:
            print(
                f"[wire] seq={s.seq_id} status={s.status} tokens={len(s.token_ids)} "
                f"prompt={s.num_prompt_tokens} cached={s.num_cached_tokens} "
                f"blocks={len(s.block_table)} scheduled={s.num_scheduled_tokens} "
                f"draft={s.spec_draft_ids}",
                flush=True,
            )
        assert len(s.token_ids) == s.num_cached_tokens + 1, (
            f"spec prefill invariant 被破坏 seq={s.seq_id} status={s.status} "
            f"len(token_ids)={len(s.token_ids)} num_prompt_tokens={s.num_prompt_tokens} "
            f"num_cached_tokens={s.num_cached_tokens} blocks={len(s.block_table)} "
            f"draft={s.spec_draft_ids}"
        )
        temperature, top_p, top_k, min_p = _sampling_fields(s)
        return SeqSchedule(
            seq_id=s.seq_id,
            block_table=s.block_table,
            num_cached_tokens=s.num_cached_tokens,
            token_ids=[s.last_token, *s.spec_draft_ids],
            need_sample=False,
            spec_draft_ids=list(s.spec_draft_ids),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )

    # 本步要算的 token 区间：从已缓存处开始。
    start = s.num_cached_tokens
    # 长度是本步调度的量 —— chunked prefill 下可能不足一整条 prompt。
    end = start + s.num_scheduled_tokens
    temperature, top_p, top_k, min_p = _sampling_fields(s)
    return SeqSchedule(
        seq_id=s.seq_id,
        block_table=s.block_table,
        num_cached_tokens=s.num_cached_tokens,
        # 只发这一段切片：已缓存的前缀与尚未调度的尾巴都不发。
        token_ids=s.token_ids[start:end],
        need_sample=s.need_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
    )


# 单条序列的前向输入。用 msgspec 是为了能直接 msgpack 编码、跨进程传。
class SeqSchedule(msgspec.Struct):
    """
        单条序列的前向输入（Gloo / 本进程都可以用）。

        token_ids：decode 为 [last_token]；prefill 为本步切片。
        num_cached_tokens：本步之前已写入 KV 的长度，用来算 position / cache_seqlen。
        block_table：与 Sequence 共享引用；GPU 侧按 seq_id 做行缓存，增量 H2D。
    """

    # 下面这些字段顺序即编码顺序，别随意调整。
    seq_id: int
    block_table: list[int]
    num_cached_tokens: int
    token_ids: list[int]
    # 本步是否需要消费采样结果（chunked prefill 的中间段为 False）。
    need_sample: bool = False
    # 投机解码：本步输入的草稿（token_ids 里已经带上，这里单独留一份
    # 方便 Worker 侧直接拿到草稿做验证，不必再切一次切片）。
    spec_draft_ids: list[int] | None = None
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    # ZOS 在途行专用：本行的输入 token 要取「上一批采样结果」的第几个。
    # -1 = 普通行（token_ids 里就是真 token）。放在最后是为了不动跨进程编码顺序。
    remap: int = -1


# to_wire 的结果，也是 Worker.execute_model 的入参。到这里已经是纯数据了。
class SchedulerOutput(msgspec.Struct):
    """to_wire 的结果：Worker.execute_model 的入参。"""

    prefills: list[SeqSchedule] = msgspec.field(default_factory=list)
    decodes: list[SeqSchedule] = msgspec.field(default_factory=list)


# 本步的增量输出：经 Client 回到 Engine。
class EngineCoreOutput(msgspec.Struct):
    """
        本步一条序列的增量，给 Engine / MPClient。

        new_token_ids 为空：本步未采样（chunked prefill 中间段，或 max_tokens 已用尽）。
    """

    seq_id: int
    # 增量 token（可能为空）与是否完成。
    new_token_ids: list[int] = msgspec.field(default_factory=list)
    is_finished: bool = False
