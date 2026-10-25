# ============================================================================
# 【本文件是做什么的】调度器：决定"这一步跑哪些序列"，并管 KV 块的分配与回收。
#     schedule()          每步调用一次，产出 ScheduledBatch
#     _schedule_decode    续跑已在跑的序列（decode 优先，保住延迟）
#     _pull_prefill_chunk 按 token 预算从等待队列拉新请求（chunked prefill）
#     preempt()           显存不够时把某个序列踢回等待队列
#
# 【为什么要混合调度（mix_prefill_decode）】只跑 decode 时 GPU 用不满（batch 小），
#     只跑 prefill 时正在生成的请求被饿死（延迟炸掉）。
#     把两者塞进同一步是同时喂饱算力与保住延迟的常规做法 ——
#     代价是 attention 里要处理"同一批里既有 prefill 又有 decode"的元信息。
#
# 【为什么 prefill 要切 chunk】一个长 prompt 的 prefill 会独占一步很久，
#     期间所有 decode 请求都卡住（尾延迟）。按 token 预算切块后，
#     长 prefill 可以分几步做完，中间穿插 decode。
#
# ⚠️ 调度器是唯一决定"谁能上、谁被抢占"的地方；
#   显存账（是否需要抢占）由它配合 BlockManager 判断，模型侧不参与。
# ============================================================================

from __future__ import annotations

import os
from dataclasses import dataclass
from collections import deque

from ..data import EngineCoreOutput, KVManager, ScheduledBatch, Sequence, SequenceStatus
from ..spec import NgramDrafter, greedy_accept_length

# 投机路径的逐序列诊断开关（KRITACE_SPEC_DEBUG=1 打开）。
#
# 为什么需要它：投机与基线不一致时，只看最终 token 无法区分三种成因 ——
#   ① 草稿本身错（n-gram 匹配到错位置）
#   ② 验证前向的位置对错（KV 写入偏移）
#   ③ 接受判定错（seg 取值越界）
# 必须同时打印 draft / seg / j / cached 才能分开。默认关闭，避免刷屏。
_SPEC_DEBUG = bool(os.environ.get("KRITACE_SPEC_DEBUG"))


# 调度器自身的状态很少：两条队列 + 一个 KV 管理器。真正复杂的都在下面几个私有方法里。
# 队列的顺序是有语义的，改顺序会改变行为：
#   waiting 是 FIFO（保证公平、不被饿死）；
#    running 的队尾是抢占时优先牺牲的位置。
# ZOS 账本用到的两个小记录。只是"快照"，不持有任何 GPU 资源。
@dataclass
class _StepSeq:
    """一批里的一条序列：谁、这一步算了几个 token。"""

    seq: Sequence
    num_tokens: int


@dataclass
class _InFlight:
    """已经发射、但采样结果还没回灌的那一批。"""

    decodes: list[_StepSeq]
    prefills: list[_StepSeq]
    n_p: int
    n_d: int
    # 这一批用的是哪一块锁页缓冲 / 哪个 cudaEvent（双缓冲的槽位）。
    slot: int
    # 这一步的指标（prefill token 数 + decode 条数），回灌时一起返回。
    metric: int


class Scheduler:
    """请求调度器（并行无关）。

    只管理 waiting / running 队列与 KV 逻辑块预算，产出 ScheduledBatch。
    TP/PP/DP 均不在此体现：同一份 schedule 由 Executor 广播（TP/PP）或由各 DP
    副本各自独立调度（DP）。
    """

    # 构造只保存配置与队列，不做任何工作。
    def __init__(
        self,
        kv: KVManager,
        *,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        eos_token_id: int,
        mix_prefill_decode: bool = True,
        spec_k: int = 0,
        spec_ngram_max_n: int = 4,
    ) -> None:
        # KV cache 管理器，负责块分配、释放、前缀匹配与回收。
        self.kv = kv
        # 单步最多调度的序列数。
        self.max_num_seqs = max_num_seqs
        # 单步最多处理的 token 总数。
        self.max_num_batched_tokens = max_num_batched_tokens
        # EOS token id，用于判断生成是否结束。
        self.eos = eos_token_id
        # 是否允许 prefill 与 decode 在同一步混合调度。
        self.mix_prefill_decode = mix_prefill_decode
        # 投机解码：n-gram 草稿只靠 token 历史，在 CPU 侧零成本生成，
        # 因此 drafter 留在调度器里，不必下沉到 Worker。
        # draft model 方案则需要写在 Worker 侧（它要跑前向）。
        self.spec_k = spec_k
        self.spec_drafter = NgramDrafter(max_n=spec_ngram_max_n) if spec_k > 0 else None
        # 等待队列：新请求或被抢占后需要重新分配 KV 的序列。
        self.waiting: deque[Sequence] = deque()
        # 运行队列：已开始生成或已完成 prefill、可参与 decode 的序列。
        self.running: deque[Sequence] = deque()

        # ---------- ZOS 在途账本（见文件尾部"在途账本的登记与回灌"）----------
        # 为 None 表示"当前没有已发射未回灌的批" —— 同步模式下它永远为 None。
        self._pending: _InFlight | None = None
        # 在途序列 → 采样下标 / 那一步的 token 数。
        self._inflight_slots: dict[int, int] = {}
        self._inflight_tokens: dict[int, int] = {}
        # 在途、但这一步拿不到可用采样结果的序列（中间块 prefill）。
        self._blocked: set[int] = set()
        # 已完成、但 KV 还要再压一步才归还的序列。
        self._deferred_release: list[Sequence] = []

    # 两条队列都空 = 没有在等的、也没有在跑的 → 引擎主循环可以退出了。
    def is_finished(self) -> bool:
        # ZOS：还有一批在途时也不算结束 —— 它的 token 还没取回来。
        return not self.waiting and not self.running and self._pending is None

    # 新请求一律先进 waiting；状态改成 WAITING 是给外部（如 postprocess）判断用的。
    def add(self, seq: Sequence) -> None:
        # publish=False 是关键：被抢占的序列并没有真正结束，后面还要重来一遍。
        # 此刻把它的块发布进前缀索引没有意义 —— 反而可能让别的请求命中一段"马上要被释放"
        # 的内容。只有真正 FINISHED 的序列才值得发布。
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    # 唯一对外的调度入口：按配置选一种策略，把结果包成 ScheduledBatch 交给执行器。
    # 两种策略的返回值契约不同 —— prefill 优先模式可能返回"只有 decodes、没有 prefills"。
    def schedule(self) -> ScheduledBatch:
        if self.mix_prefill_decode:
            p, d = self._schedule_mixed()
        else:
            p, d = self._schedule_prefill_first()
        # 有草稿的序列必须走 prefill 通道：只有 varlen attention 能一次
        # 算 K+1 个 query；decode 通道的 flash_attn_with_kvcache 每序列只收 1 个。
        spec = [s for s in d if s.spec_draft_ids]
        if spec:
            d = [s for s in d if not s.spec_draft_ids]
            p = [*p, *spec]
        return ScheduledBatch(prefills=p, decodes=d)

    # 这里只管序列数预算，不管 token 预算 —— token 预算由各循环自己逐条累加判断。
    # 分开的原因是：prefill 吃的 token 数是不确定的（可能几十上百），
    # decode 恰好是 1，没法在统一出口上判。
    def _room(self, prefills: list[Sequence], decodes: list[Sequence]) -> bool:
        return len(prefills) + len(decodes) < self.max_num_seqs

    # "尝试"两字的含义：它可能失败（页池不够），失败时不抛异常也不部分分配。
    # 先看 block_table 是否已有内容：非空说明这条序列的 KV 还在（刚才被调度过），
    # 直接复用。若重复分配，既会漏页，也会把已有的前缀丢掉。
    def _try_alloc(self, seq: Sequence) -> bool:
        if seq.block_table:
            return True
        if not self.kv.can_allocate(seq):
            return False
        self.kv.allocate(seq)
        return True

    # ==================== decode 调度 ====================
    def _schedule_decode(
        self,
        prefills: list[Sequence],
        decodes: list[Sequence],
        tokens_used: int,
    ) -> int:
        # 循环条件要连着下面一起看：序列数没超、且下面还有 token 预算检查。
        while self.running and self._room(prefills, decodes):
            # 从队头取（FIFO）。取出来先不放回去 —— 万一它自己也被抢占，
            # 就该留在 waiting 里，不能同时存在于 running。
            seq = self.running.popleft()
            sid = seq.seq_id

            # ZOS：这条序列上一批跑过、但结果还没回灌（chunked prefill 的中间块）。
            # 它的账本是"半截"的，同一步里再排一次就会把同一段 chunk 算两遍。
            if sid in self._blocked:
                self.running.appendleft(seq)
                break

            # ZOS 账本：它在上一批的采样下标（None = 不在途）、上一批算了几个 token。
            slot = self._inflight_slots.get(sid)

            if slot is not None:
                # ---- 在途行 ----
                # CPU 账本比 GPU 落后一步：num_cached_tokens 还是"上一批之前"的值，
                # 所以要写的逻辑位置 = 已缓存数 + 上一批算掉的 token 数。
                pos = seq.num_cached_tokens + self._inflight_tokens.get(sid, 1)
                # 在途行永远只跑 1 个 token：ZOS 与投机互斥（见 core.py 的开关判定），
                # 这里清掉草稿只是把不变式写死，防止上层误开。
                seq.spec_draft_ids = []
                # decode 每条固定吃 1 个 token 预算 → 预算不够就停。
                if tokens_used + 1 > self.max_num_batched_tokens:
                    self.running.appendleft(seq)
                    break
                # KV 账也要按 pos 算：旧判据看 len(seq)，账本落后一步会少算一块。
                # 在途序列的 KV 正被 GPU 写，这一步绝不能抢它 —— 抢了就是把正在写的页
                # 还回池子。宁可本步少跑一条，也不能让它的数据被踩。
                if not self.kv.can_append_at(seq, pos):
                    self.running.appendleft(seq)
                    break
                self.kv.may_append_at(seq, pos)
                seq.zos_remap = slot
                seq.zos_start_pos = pos
                seq.num_scheduled_tokens = 1
                decodes.append(seq)
                tokens_used += 1
                continue

            # ---- 普通行（刚进 decode，或上一步的结果已经回灌）----
            # 投机：先出草稿（n-gram 是纯 CPU，不占 GPU），再按 K+1 预留 KV。
            draft: list[int] = []
            if self.spec_drafter is not None:
                draft = self.spec_drafter.propose(seq.token_ids, self.spec_k)
            n_new = len(draft) + 1

            if tokens_used + n_new > self.max_num_batched_tokens:
                self.running.appendleft(seq)
                break

            # KV 不足时，从队尾开始抢占其他运行序列，为当前序列腾出空间。
            while not self.kv.can_append_n(seq, n_new) and self.running:
                # 队尾若是在途序列（KV 正被写），本步不再抢 —— 直接放弃这次腾挪。
                if self.running[-1].seq_id in self._inflight_slots:
                    break
                self.preempt(self.running.pop())

            if not self.kv.can_append_n(seq, n_new):
                self.preempt(seq)
                break

            seq.spec_draft_ids = draft
            seq.num_scheduled_tokens = n_new
            self.kv.may_append_n(seq, n_new)
            seq.zos_remap = -1
            seq.zos_start_pos = -1
            decodes.append(seq)
            tokens_used += n_new

        # 把本轮真正跑成的序列放回 running。
        # extendleft 是往队首逐个塞，顺序会翻转 → 所以要先反着迭代一次。
        if decodes:
            self.running.extendleft(reversed(decodes))

        return tokens_used

    # ==================== prefill 调度（chunked prefill 的入口）====================
    def _pull_prefill_chunk(
        self,
        prefills: list[Sequence],
        decodes: list[Sequence],
        tokens_used: int,
        *,
        allow_trailing_partial: bool = True,
    ) -> int:
        # 没得拉、或者序列数已经满了 → 直接返回，什么也不做。
        if not self.waiting or not self._room(prefills, decodes):
            return tokens_used

        # 每轮最多只允许一个序列处于"半截 prefill"状态。
        # 否则页池会被一群"都还没产出第一个 token、但都已经占着页"的序列塞满，
        # 结果是所有人都在等、没人能出 token（活锁）。
        # 记录已产生的部分 prefill chunk 数量。
        # 每轮最多只允许一个序列处于未完成 prefill 状态，避免过多半截 prefill。
        n_partial = 0
        # list(...) 取快照：循环体里会从 waiting 中 remove，直接遍历原队列会出问题。
        for seq in list(self.waiting):
            if not self._room(prefills, decodes):
                break

            # ZOS：这条序列上一批的 chunk 还没回灌。不能给它排新 chunk ——
            # 否则同一段 token 会被算两遍，账本也会被加两次。FIFO 语义要求 break。
            if seq.seq_id in self._blocked:
                break

            # 每轮重新算剩余预算 —— 前面几条序列可能已经吃掉一部分了。
            remaining = self.max_num_batched_tokens - tokens_used
            if remaining <= 0:
                break

            # 页池不够就不能开始 → 直接停。
            # 这里用 break 而不是 continue：队列是 FIFO 的，
            # 前面这条都进不来，后面的只会更差（它也需要至少一页）。
            if not self._try_alloc(seq):
                break

            # need = "这条序列还需要算多少 token" = 总长度 − 已缓存长度。
            # 前缀命中越多，need 越小；极端情况可能为 0（见下面的分支）。
            # 该序列本次还需要计算的 token 数。
            need = seq.num_tokens - seq.num_cached_tokens
            # 整条 prompt 都在前缀缓存里 → 一次 prefill 都不必做，
            # 直接转成 running 去参加 decode。这是前缀缓存最理想的效果。
            if need <= 0:
                # 前缀完全命中：无需执行 prefill，直接进入运行队列作为 decode 候选。
                seq.status = SequenceStatus.RUNNING
                self.waiting.remove(seq)
                self.running.append(seq)
                continue

            # 两种模式的差别就体现在这里（allow_trailing_partial）：
            #   prefill 优先模式：本轮已经有 prefill 了，就不允许再塞一条会被切开的
            #   混合模式：允许最后一条被切开，尽量把 token 预算填满、不浪费这一步
            # 如果不允许在已有 prefill 后追加部分 chunk，且当前预算不足，则停止拉取。
            if remaining < need and prefills and not allow_trailing_partial:
                break

            # chunk 就是本轮为这条序列真正要算的 token 数：
            # 预算够 → 一次算完 need；预算不够 → 只算到预算边界。
            # 切在哪完全由 token 预算决定，这也是 chunked prefill 名称的来源。
            # 计算本步为该序列调度的 chunk 大小。
            chunk = min(need, remaining)
            seq.num_scheduled_tokens = chunk
            prefills.append(seq)
            # prefill 行永远发真 token，不带在途标记。
            seq.zos_remap = -1
            seq.zos_start_pos = -1
            tokens_used += chunk

            # 这一块算完恰好把 prompt 走完 → 本序列本轮需要采样（要吐第一个 token）。
            # 本步完成后 prompt 全部进入缓存：转入运行队列，本步需采样。
            if seq.num_cached_tokens + chunk == seq.num_tokens:
                seq.need_sample = True
                seq.status = SequenceStatus.RUNNING
                self.waiting.remove(seq)
                self.running.append(seq)
            else:
                # 只算了一块：本轮不采样（还没有可信的"最后一个 token"），
                # 序列继续留在 waiting 里，等下一轮接着算。
                # 该序列本步只完成部分 prefill，仍留在等待队列中。
                seq.need_sample = False
                n_partial += 1
                if n_partial >= 1:
                    break

        return tokens_used

    # ==================== 策略 A：prefill 优先 ====================
    # 先把 waiting 里的新请求喂饱，实在没有 prefill 可做才去跑 decode。
    # 特点：TTFT 好（新请求不用排队等老请求生成完）；
    #       代价是 ITL 会变差（正在生成的请求被晾在一边）。
    def _schedule_prefill_first(self) -> tuple[list[Sequence], list[Sequence]]:
        prefills: list[Sequence] = []
        decodes: list[Sequence] = []
        # allow_trailing_partial=False 是这个模式的灵魂：
        #   一旦本轮已经有 prefill 了，就绝不再塞"会被切开"的序列。
        #    保证一轮里的 prefill 都是完整的，行为可预测、延迟好估算。
        # prefill 优先模式下，不允许在已有 prefill 后追加部分 chunk。
        self._pull_prefill_chunk(prefills, decodes, 0, allow_trailing_partial=False)
        # 有 prefill 可做 → 本轮只做 prefill，decode 一个都不跑（注意返回的是空 decodes）。
        if prefills:
            return prefills, []
        self._schedule_decode(prefills, decodes, 0)
        return [], decodes

    # ==================== 策略 B：混合调度（v3 默认）====================
    # 先尽力安排 decode，再用剩下来的 token 预算去拉 prefill。
    # 特点：decode 的 ITL 稳定（老请求优先），长 prompt 靠剩余预算一点点推进 ——
    #       既不会被饿死，也不会把 decode 挤走。这是连续批处理想要的那个效果。
    def _schedule_mixed(self) -> tuple[list[Sequence], list[Sequence]]:
        prefills: list[Sequence] = []
        decodes: list[Sequence] = []
        # 这里给 decode 传的预算是 0，不是"没用预算"的意思 ——
        # decode 是"先做、不设上限"的，真正的账由下面把 used 交给 prefill 去算。
        used = self._schedule_decode(prefills, decodes, 0)
        # 有 decode 跑成了 → 把剩下的 token 预算花在 prefill 上，不浪费这一轮。
        if decodes:
            # 投机验证的 batch 布局是定宽 [bs, K+1]（见 model_runner._spec_verify），
            # 混入普通 prefill 会让定宽切片错位。所以有草稿的这一步不再拉新请求，
            # 等验证结束、草稿被消费后，下一步再正常拉取。
            if not any(s.spec_draft_ids for s in decodes):
                self._pull_prefill_chunk(prefills, decodes, used)
            return prefills, decodes
        # 一条 decode 都没跑成（比如刚启动、或者 running 空了）→ 退化成 prefill 优先。
        return self._schedule_prefill_first()

    # ==================== 抢占 ====================
    # 语义 = "把这条序列的 KV 全部丢掉，让它回 waiting 从头再来"。
    # v3 只实现了 recompute 这一种做法（没有 swap 到 CPU 的实现），依据是：
    #     重算代价 = 重新读一遍 prompt 的 prefill（compute-bound，且能和其他请求并行）
    #     换出代价 = 页数据拷进拷出，拷回来时还要等 PCIe 带宽（且期间还占着 CPU 内存）
    # 短序列重算更划算；序列很长且机器 PCIe 很宽时这个判断会反过来。
    def preempt(self, seq: Sequence) -> None:
        # 已完成的不再抢占：它的 KV 已经归还、并且已经发布进前缀缓存了。
        # 再来一次 deallocate 会重复归还同一批块 → 引用计数错乱。
        if seq.status == SequenceStatus.FINISHED:
            return
        seq.status = SequenceStatus.WAITING
        self.kv.deallocate(seq, publish=False)
        # 被抢占的序列不再持有任何在途标记：它要整条重算。
        seq.zos_remap = -1
        seq.zos_start_pos = -1
        # 抢占把 KV 账本清零（deallocate 会重置 num_cached_tokens / block_table），
        # 所以必须一并丢掉草稿：否则这条序列会带着 decode 语义的草稿回到 waiting，
        # 重新 prefill 时草稿长度与已缓存前缀对不上。
        seq.spec_draft_ids = []
        self.waiting.appendleft(seq)

    @staticmethod
    def _spec_width(seqs: list[Sequence]) -> int:
        """投机验证的定宽：每条序列在扁平数组里占多少个元素。

        取 ``max(len(草稿)) + 1`` 而不是各序列自己的长度 —— 不同序列的
        n-gram 命中情况不同，草稿长度会不一样，必须 pad 到同一个宽度，
        才能让 ``i*width : (i+1)*width`` 这种切片对齐（短的那条后面是 pad -1）。
        """
        return max(len(s.spec_draft_ids) for s in seqs) + 1

    @staticmethod
    def prefill_samples(prefills: list[Sequence]) -> int:
        """本步 prefill 通道会回传多少个**元素**（注意：不是请求数）。

        ``read_samples`` 是按元素数切片的：普通 prefill 每个请求恰好 1 个样本，
        所以两者相等、区别长期被掩盖；投机验证一次回传 K+1 个位置的 argmax，
        就变成了 ``请求数 × 定宽``。算错的后果很隐蔽 —— 只会拿到前 1 个，
        接受长度被钳到 1，投机的收益全部消失。

        两种情形不会混批（见 _schedule_mixed），所以判断第一个即可。
        """
        if not prefills:
            return 0
        if prefills[0].spec_draft_ids:
            return Scheduler._spec_width(prefills) * len(prefills)
        return sum(s.num_scheduled_tokens for s in prefills)

    def _update_spec_batch(
        self, seqs: list[Sequence], flat_ids: list[int]
    ) -> list[EngineCoreOutput]:
        """投机验证的回写。

        ``flat_ids`` 是扁平数组：第 i 条序列占 ``[i*W, (i+1)*W)``，W = max_draft + 1，
        每段是验证位置 0..K 上的 argmax，pad 位为 -1。

        与普通 prefill 的关键差别：**有效长度是 j+1，不是 num_scheduled_tokens**。
        验证阶段写进 KV 的是 K+1 个位置，但只有前 j+1 个是真实的，
        因此 num_cached_tokens 只推进 j+1；其余 K-j 个 slot 留在同一页里，
        下一步的写入会自然覆盖它们（见 block_manager.may_append_n）。

        另外不必把被接受的 token 过网络回传：它们**就是草稿的前 j 个**，
        而草稿本来就在调度器手里。
        """
        width = self._spec_width(seqs)
        outputs: list[EngineCoreOutput] = []
        finished: set[int] = set()

        for i, seq in enumerate(seqs):
            seg = flat_ids[i * width : (i + 1) * width]
            draft = seq.spec_draft_ids
            j = greedy_accept_length(draft, seg[: len(draft)])
            emitted = [*draft[:j], int(seg[j])]

            if _SPEC_DEBUG:
                # 定位"投机与基线不一致"必须同时看到草稿、验证输出与接受数：
                # 只看 emitted 无法区分"草稿错"和"验证位错"。
                print(
                    f"[spec] seq={seq.seq_id} L={len(seq.token_ids)} "
                    f"cached={seq.num_cached_tokens} sched={seq.num_scheduled_tokens} "
                    f"width={width} draft={draft} seg={seg} j={j} "
                    f"emitted={emitted}",
                    flush=True,
                )

            seq.num_cached_tokens += j + 1
            seq.num_scheduled_tokens = 0
            self.kv.sync_prefix(seq, num_new_tokens=j + 1)
            seq.spec_draft_ids = []
            seq.spec_accepted = j

            new_ids: list[int] = []
            is_done = False
            for tok in emitted:
                if seq.num_completion_tokens >= seq.max_tokens:
                    is_done = True
                    break
                seq.append_token(tok)
                new_ids.append(tok)
                if (not seq.ignore_eos) and tok == self.eos:
                    is_done = True
                    break

            if is_done:
                seq.status = SequenceStatus.FINISHED
                # 完成序列释放 KV cache，并发布前缀供后续请求复用。
                self.kv.deallocate(seq, publish=True)
                finished.add(seq.seq_id)
            outputs.append(
                EngineCoreOutput(
                    seq_id=seq.seq_id, new_token_ids=new_ids, is_finished=is_done
                )
            )

        # 与 _update_batch 末尾同样的记账，不能省：
        # 序列结束后必须从 running 里摘掉。少了这一步，FINISHED 的序列会被
        # 下一步重新调度、重新挂草稿，而它的 KV 账本已在 deallocate 里清零
        # （num_cached_tokens=0、block_table 为空）—— 组包侧的
        # "len(token_ids) == num_cached_tokens + 1" 断言随即失败。
        # 这类 bug 在单请求短序列上不易出现，一旦 max_tokens 被触发就会冒出来。
        # 已完成序列需要从运行队列中移除。
        # 未完成的 partial prefill 已在上面跳过，不会出现在 finished 中。
        # 用集合 + 重建队列，而不是边走边 remove：
        #   running 里同一条序列可能有多处引用，逐个 remove 容易漏。
        if finished:
            self.running = deque(s for s in self.running if s.seq_id not in finished)
        return outputs

    # ==================== ZOS：在途账本的登记与回灌 ====================
    # 【为什么需要这本账】ZOS 下 CPU 只领先 GPU 一步：第 t 步发射完 batch_t，
    # 才回头去收 batch_{t-1} 的采样结果。所以 schedule(batch_t) 那一刻，CPU 对
    # batch_{t-1} 一无所知 —— 既不知道它采出的 token，也不知道它是否让某条序列结束。
    # 这本账把"在途"显式记下来，供三处使用：
    #   _inflight_slots   在途序列 → 它在采样缓冲里的下标（GPU 端 gather 用）
    #   _inflight_tokens  在途序列 → 上一批给它算了几个 token（算乐观位置用）
    #   _blocked          在途、但这一批拿不到可用采样的序列（中间块 prefill）
    #   _deferred_release 已完成、但 KV 要再压一步才归还的序列（见 _release_later）
    #
    # 【三条不变式，改这个文件时别破】
    #   ① 同一序列不会在同一批里出现两次（所以"多调度一步"最多只多一步）；
    #   ② 在途序列的 KV 绝不能在它那一批落地前归还（否则正在写的页会被别人复用）；
    #   ③ 在途行发出去的 token_ids 必须是空 list —— 有值就说明账本已经回灌了。
    def swap_pending(
        self,
        sched: ScheduledBatch,
        n_p: int,
        n_d: int,
        slot: int,
        metric: int,
    ) -> "_InFlight | None":
        """把刚发射的一批登记为在途，返回被替换下来的上一批（可能为 None）。"""
        prev = self._pending
        self._install(sched, n_p, n_d, slot, metric)
        return prev

    def take_pending(self) -> "_InFlight | None":
        """摘走当前在途批（尾段收尾用），之后 _pending 为空。"""
        prev = self._pending
        self._clear()
        return prev

    def _clear(self) -> None:
        self._pending = None
        self._inflight_slots = {}
        self._inflight_tokens = {}
        self._blocked = set()

    def _install(
        self, sched: ScheduledBatch, n_p: int, n_d: int, slot: int, metric: int
    ) -> None:
        # 快照：num_scheduled_tokens 会在下一次 schedule 被改写，
        # 而回灌发生在下一次 schedule 之后，所以必须在这里存下来。
        dec = [_StepSeq(s, s.num_scheduled_tokens) for s in sched.decodes]
        pre = [_StepSeq(s, s.num_scheduled_tokens) for s in sched.prefills]
        slots: dict[int, int] = {}
        tokens: dict[int, int] = {}
        blocked: set[int] = set()
        for i, r in enumerate(dec):
            # decode 一条一定采样：采样缓冲里 decode 段在前。
            slots[r.seq.seq_id] = i
            tokens[r.seq.seq_id] = r.num_tokens
        for i, r in enumerate(pre):
            tokens[r.seq.seq_id] = r.num_tokens
            if r.seq.need_sample:
                # prefill 段在采样缓冲里接在 decode 后面。
                slots[r.seq.seq_id] = n_d + i
            else:
                # 中间块 prefill 不采样：它这一步没有可用 token，下一批不能再排它。
                blocked.add(r.seq.seq_id)
        self._pending = _InFlight(dec, pre, n_p, n_d, slot, metric)
        self._inflight_slots = slots
        self._inflight_tokens = tokens
        self._blocked = blocked

    # 回灌一批的采样结果：推进序列状态、追加 token、判断结束。
    def finish(
        self, prev: "_InFlight", p_ids: list[int], d_ids: list[int]
    ) -> list[EngineCoreOutput]:
        # 先还掉上一步登记的待释放。调用点保证：此刻 prev 这一批已经算完
        # （read_samples 已同步），所以"上一步登记的那些"更早就落地了，还块是安全的。
        self._flush_deferred()
        outputs: list[EngineCoreOutput] = []
        if prev.decodes:
            outputs.extend(self._apply(prev.decodes, d_ids, is_prefill=False))
        if prev.prefills:
            outputs.extend(self._apply(prev.prefills, p_ids, is_prefill=True))
        return outputs

    def _flush_deferred(self) -> None:
        if not self._deferred_release:
            return
        for seq in self._deferred_release:
            self.kv.deallocate(seq, publish=True)
        self._deferred_release = []

    def _release_later(self, seq: Sequence) -> None:
        """为什么要推迟一轮还块：ZOS 下判断出"完成"时，**下一批已经在 GPU 上跑了**，
        而那一批里很可能还有这条序列 —— 它还要往自己最后那一格里写 token。
        当场 deallocate 等于把正在写的页交还池子，下一个请求拿到它就是数据踩踏。
        所以统一压一步，等下一批落地（read_samples 返回）之后才真正 release。
        """
        self._deferred_release.append(seq)

    def _apply(
        self,
        recs: list["_StepSeq"],
        token_ids: list[int],
        *,
        is_prefill: bool,
    ) -> list[EngineCoreOutput]:
        # 投机验证走 prefill 通道，但它的回写规则与普通 prefill 完全不同：
        # 普通 prefill 按 num_scheduled_tokens 推进，投机的有效长度却是 j+1。
        # ZOS 与投机互斥（core.py 里判定），这条分支只为同步模式保留。
        if is_prefill and any(r.seq.spec_draft_ids for r in recs):
            return self._update_spec_batch([r.seq for r in recs], token_ids)

        finished: set[int] = set()
        outputs: list[EngineCoreOutput] = []

        # zip 依赖两边顺序完全一致。这个顺序是在执行器里约定的
        # （decode 的 id 在前、prefill 的在后）。顺序错了不会报错，
        # 只会把 token 装到错误的序列上 —— 这类 bug 极难定位。
        for r, token_id in zip(recs, token_ids):
            seq = r.seq
            # 防御：这一批发射时它还活着，回灌时已经收尾（ZOS 的"多调度一步"）。
            # 它那一格 KV 写就写了，反正块要还；不能再追加 token、也不能重复结束。
            if seq.status == SequenceStatus.FINISHED:
                continue
            # 用快照里的 token 数，而不是 seq.num_scheduled_tokens ——
            # 后者可能已被下一批的 schedule 改写（ZOS 才有这种情况）。
            seq.num_scheduled_tokens = r.num_tokens
            # 先把"本步写进去的 token"记进账本，再同步前缀索引 —— sync_prefix 内部
            # 用 num_cached_tokens - num_new_tokens 反推本步写入的起点，顺序反了就会发布错的块。
            seq.num_cached_tokens += r.num_tokens
            self.kv.sync_prefix(seq)
            seq.num_scheduled_tokens = 0

            # 未完成 prompt 的 prefill chunk 本步不采样，也不判断是否完成。
            if is_prefill and seq.num_cached_tokens < seq.num_prompt_tokens:
                outputs.append(EngineCoreOutput(seq_id=seq.seq_id))
                continue

            # max_tokens<=0 表示只处理 prompt、不生成新 token。
            if seq.max_tokens <= 0:
                seq.status = SequenceStatus.FINISHED
                self._release_later(seq)
                finished.add(seq.seq_id)
                outputs.append(EngineCoreOutput(seq_id=seq.seq_id, is_finished=True))
                continue

            # 追加本步采样得到的 token，并判断是否完成。
            seq.append_token(token_id)

            hit_eos = (not seq.ignore_eos) and token_id == self.eos
            hit_len = seq.num_completion_tokens >= seq.max_tokens
            if hit_eos or hit_len:
                seq.status = SequenceStatus.FINISHED
                self._release_later(seq)
                finished.add(seq.seq_id)

            outputs.append(
                EngineCoreOutput(
                    seq_id=seq.seq_id,
                    new_token_ids=[token_id],
                    is_finished=seq.seq_id in finished,
                )
            )

        if finished:
            self.running = deque(s for s in self.running if s.seq_id not in finished)
        return outputs
