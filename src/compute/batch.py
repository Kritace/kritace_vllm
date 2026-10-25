# ============================================================================
# 【本文件是做什么的】把调度器给出的序列，写进常驻 GPU 缓冲。
# decode 走 build_decode，prefill 走 build_prefill；两者共用同一套缓冲。
#
# 【为什么要三块内存：numpy 视图 → pinned CPU 缓冲 → GPU】
#   每步要送十几个小张量上去，如果逐个走 Python→tensor→copy，光"造张量"这件事
#   就是每步固定成本。这里的做法是：
#     ① 常驻的 pinned CPU 缓冲 各配一个 numpy 视图 —— 填数据时直接写 numpy
#        （切片赋值、np.arange），比逐元素建张量快得多；
#     ② 每个张量只做一次 `copy_(..., non_blocking=True)` 上去；
#     ③ 因为缓冲是锁页的，这次拷贝可以和 GPU 计算重叠。
#
# 【为什么要"指纹"与"快照"】(这两处是这套代码里最抠的优化)
#     * 页表：decode 每步只有"跨页"那一瞬间会变。所以每行记一个
#       (首页, 尾页, 长度) 指纹；没变就整行跳过，只多了一页就只补最后一格。
#     * 采样参数：用 (seq_id, 四个参数) 做快照，没变就一个字节都不传。
#
# 【为什么要 pad_to】CUDA Graph 是按档位录图的（1/2/4/8/16/…）。
#   真实并发数落在一档之内时，多余的槽位也会被图算一遍，
#   所以必须把它们写成安全值（见下）。
#
# 【⚠️ 三个容易踩的点】
#     ① padding 区必须是安全值：槽位 -1（写入内核跳过）、长度 0（不读 KV）、
#        页表 -1（不访问物理页）。CG 重放不会重写这些区域，
#        一旦是非法值就会静默污染别的序列的 KV。
#     ② numpy 视图与 pinned 张量共享内存：写 numpy 就是写那块缓冲，
#        但必须在 `copy_` 之前完成；同时视图要被长期持有，不能被 GC 掉。
#     ③ 页表的增量更新只在「长度 +1 且首尾页指纹吻合」时成立（页表只会追加）。
#        换批、回退、抢占都会让它失效 —— 所以 `same` 判断不能省。
#
# 【读这个文件的顺序】
#     _pinned / _h2d / _gpu（三种内存的小工具）
#       → Batch.__init__（一整套常驻缓冲长什么样）
#       → build_decode / build_prefill（两种组包）
#       → _sync_block_tables / _sync_sampling（两个"能省则省"的同步）
# ============================================================================

from __future__ import annotations

import numpy as np
import os

import torch

from ..data.batch import SeqSchedule
from ..data.sequence import slots_for_range


# 造一块锁页的 CPU 缓冲：H2D 能不能异步，取决于它是不是锁页内存。
def _pinned(
    shape: int | tuple[int, ...],
    dtype: torch.dtype,
    fill: int | float | None = None,
) -> torch.Tensor:
    # 允许传 int 或 tuple，统一成 tuple。
    if isinstance(shape, int):
        shape = (shape,)
    # fill=None 表示只分配、不初始化 —— 省掉一次填充的开销（马上会被覆盖）。
    if fill is None:
        return torch.empty(shape, dtype=dtype, pin_memory=True)
    return torch.full(shape, fill, dtype=dtype, pin_memory=True)


# 把前 n 行拷上 GPU。缓冲区按上限开，但每步只用到前面一段。
def _h2d(dst: torch.Tensor, src: torch.Tensor, n: int) -> None:
    # non_blocking=True 是关键：CPU 提交完立刻返回，拷贝与 GPU 计算重叠。
    dst[:n].copy_(src[:n], non_blocking=True)


# GPU 侧常驻缓冲：一次按上限分配，之后永不重新分配（地址固定也是 CG 的前提）。
def _gpu(
    shape: int | tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    fill: int | float = 0,
) -> torch.Tensor:
    if isinstance(shape, int):
        shape = (shape,)
    # 用 full 而不是 empty：没填到的区域必须是安全值，不能是垃圾数据。
    return torch.full(shape, fill, dtype=dtype, device=device)


# 一整套常驻缓冲 + 每步的组包逻辑。decode / prefill 共用同一份。
class Batch:
    def __init__(
        self,
        *,
        max_bs: int,
        max_tokens: int,
        max_blocks: int,
        vocab_size: int,
        device: torch.device,
        block_size: int,
    ) -> None:
        # 三个上限来自配置：max_bs（并发）/ max_tokens（单步最大 token 数）/ max_blocks（最长序列的页数）。
        self.max_bs = max_bs
        self.max_tokens = max_tokens
        self.max_blocks = max_blocks
        self.vocab_size = vocab_size
        self.device = device
        self.block_size = block_size

        # 下面是 GPU 侧常驻输入，形状全部按上限开好。
        self.input_ids = _gpu(max_tokens, torch.int64, device)
        self.positions = _gpu(max_tokens, torch.int64, device)
        # 槽位初值 -1 = 无效槽位：写入内核遇到它会跳过。
        self.slot_mapping = _gpu(max_tokens, torch.int32, device, fill=-1)
        self.cache_seqlens = _gpu(max_bs, torch.int32, device)
        self.cu_seqlens_q = _gpu(max_bs + 1, torch.int32, device)
        self.cu_seqlens_k = _gpu(max_bs + 1, torch.int32, device)
        # 页表补 -1：attention 遇到无效页会跳过，不会越界访问。
        self.block_tables = _gpu((max_bs, max_blocks), torch.int32, device, fill=-1)

        # ZOS：本行输入 token 该取「上一批采样结果」的第几个。-1 = 普通行，直接用 input_ids。
        # 它必须留在 GPU 上 —— ZOS 的全部意义就是不让 CPU 知道那个 token。
        self.remap = _gpu(max_bs, torch.int32, device, fill=-1)
        # 本批 decode 的最大 KV 长度（写完当前 token 之后的长度）。
        # 回退 attention 用它一次性批量 gather，避免在 GPU 前向上做 host 同步。
        self.max_cache_len = 1

        # 采样参数也常驻：四组一起拷，避免每步多次小块传输。
        self.temperatures = _gpu(max_bs, torch.float32, device)
        self.top_ps = _gpu(max_bs, torch.float32, device)
        self.top_ks = _gpu(max_bs, torch.int32, device)
        self.min_ps = _gpu(max_bs, torch.float32, device)

        # 下面是「整批形态」的汇总标志，用来让采样器选最省的分支。
        self.all_greedy = True
        # 这些全是纯 CPU 计算，零同步开销。
        self.any_greedy = False
        self.use_top_k = False
        self.use_top_p = False
        self.use_min_p = False
        self.max_top_k = 0

        # CPU 侧锁页暂存做两份，每步换一份（A/B 交替）。
        # 只换 CPU 暂存，GPU 侧的输入张量仍然只有一份、地址不变 —— 所以 CUDA Graph 不受影响。
        # 为什么两份就够：第 t 步的 H2D 排在第 t 步前向之前，而下一轮 read_samples(t) 必须等
        # 第 t 步的 D2H 兑现，那时流早已过了 H2D(t)；两轮后才复用同一槽位，必然安全。
        self._bufs = [self._make_pinned_set(max_bs, max_tokens, max_blocks) for _ in range(2)]
        self._buf_slot = 0
        self._use_slot(0)
        # 每行页表的指纹 (首页, 尾页, 长度)：用来判断「这一行到底变了没有」。
        self._row_fp: list[tuple[int, int, int]] = [(-1, -1, -1)] * max_bs
        # 上一次组包的序列 id 序列：换了一批序列，页表必须整表重发。
        self._gpu_bt_ids: tuple[int, ...] = ()
        # 采样参数快照：没变就跳过整段拷贝。
        self._samp_key: tuple | None = None


    # decode 组包：每条序列只需要 4 个标量。pad_to 是 CG 档位补齐的目标。
    # decode 组包：每条序列只需要 4 个标量。pad_to 是 CG 档位补齐的目标。
    #
    # ZOS 让这一层多了第三种输入：`prev_sampled`（上一批的采样结果，还在显存里）。

    # 在途行（remap>=0）的 token id 此刻 CPU 根本不知道，只能先占位，
    # 再用一次 gather 在 GPU 上把它换成"上一批真采出来的那个 token"。
    def _make_pinned_set(self, max_bs: int, max_tokens: int, max_blocks: int) -> dict:
        # 一套锁页暂存：decode 按 max_bs 开，prefill 按 max_tokens 开。
        d = {
            "dec_ids": _pinned(max_bs, torch.int64),
            "dec_pos": _pinned(max_bs, torch.int64),
            "dec_slot": _pinned(max_bs, torch.int32, fill=-1),
            "dec_csl": _pinned(max_bs, torch.int32),
            "dec_remap": _pinned(max_bs, torch.int32, fill=-1),
            "pre_ids": _pinned(max_tokens, torch.int64),
            "pre_pos": _pinned(max_tokens, torch.int64),
            "pre_slot": _pinned(max_tokens, torch.int32, fill=-1),
            "pre_cuq": _pinned(max_bs + 1, torch.int32),
            "pre_cuk": _pinned(max_bs + 1, torch.int32),
            "h_temp": _pinned(max_bs, torch.float32),
            "h_top_p": _pinned(max_bs, torch.float32),
            "h_top_k": _pinned(max_bs, torch.int32),
            "h_min_p": _pinned(max_bs, torch.float32),
            "bt_pin": _pinned((max_bs, max_blocks), torch.int32, fill=-1),
        }
        # 每个张量配一个 numpy 视图：填数据走 numpy，拷贝走 copy_，共享同一块内存。
        for k in list(d):
            d[k + "_np"] = d[k].numpy() if d[k].dim() <= 2 else None
        d["bt_cpu"] = d["bt_pin"].numpy()
        return d

    def _use_slot(self, slot: int) -> None:
        # 把当前槽位的缓冲挂到属性名上，组包代码不用关心槽位。
        d = self._bufs[slot]
        for k, v in d.items():
            if k.endswith("_np") and v is None:
                continue
            setattr(self, "_" + k, v)
        # 页表那一份的视图名字是 _bt_cpu（不是自动推导出来的），单独处理。
        self._bt_pin = d["bt_pin"]
        self._bt_cpu = d["bt_cpu"]

    def begin_step(self) -> None:
        # 每步开头翻一次槽位；同一个 step 里 decode / prefill 两次组包共用同一个槽位。
        self._buf_slot ^= 1
        self._use_slot(self._buf_slot)

    def build_decode(
        self,
        seqs: list[SeqSchedule],
        *,
        pad_to: int | None = None,
        prev_sampled: torch.Tensor | None = None,
    ) -> None:
        # 真实并发数（可能小于 CG 档位）。
        bs = len(seqs)
        ids_np, pos_np = self._dec_ids_np, self._dec_pos_np
        slot_np, csl_np = self._dec_slot_np, self._dec_csl_np
        remap_np = self._dec_remap_np
        bsz = self.block_size
        # 本批里有没有在途行：没有就整段跳过 gather，普通路径零额外开销。
        any_remap = False
        max_csl = 0
        # 这个循环是每步 CPU 侧的主要工作：逐条算 4 个标量。
        for i, s in enumerate(seqs):
            # 注意它是每步都要跑的 Python 循环 —— batch 很大时，它本身就会变成 CPU 瓶颈。
            pos = s.num_cached_tokens
            bt = s.block_table
            # 只取最后一个 token：decode 每次只喂一个新 token。
            if s.remap >= 0:
                # 在途行：token 还不知道，先写 0 占位，位置与槽位照常算（它们是 CPU 账本）。
                ids_np[i] = 0
                remap_np[i] = s.remap
                any_remap = True
            else:
                ids_np[i] = s.token_ids[-1]
                remap_np[i] = -1
            pos_np[i] = pos
            # 槽位 = 页号 × 页大小 + 页内偏移。分页布局下 token 不连续，必须每步现算。
            slot_np[i] = bt[pos // bsz] * bsz + pos % bsz
            # attention 要读的历史长度 = 写完这个 token 之后的长度。
            csl_np[i] = pos + 1
            if pos + 1 > max_csl:
                max_csl = pos + 1
        self.max_cache_len = max(max_csl, 1)
        # 4 个小张量一次性拷上去（都是 non_blocking）。
        _h2d(self.input_ids, self._dec_ids, bs)
        _h2d(self.positions, self._dec_pos, bs)
        _h2d(self.slot_mapping, self._dec_slot, bs)
        _h2d(self.cache_seqlens, self._dec_csl, bs)

        # ---- ZOS：把占位行换成上一批真采出来的 token ----
        # 这里不需要任何同步：上一批的采样核排在同一条 CUDA 流的前面，
        # gather 读到的必然是它的结果。这正是"CPU 不知道 token 也能接着跑"的关键。
        n_row = pad_to if pad_to is not None else bs
        if any_remap and prev_sampled is not None:
            # 补齐行没有 remap：写 -1，让 where 保留原值。
            if n_row > bs:
                remap_np[bs:n_row] = -1
                # 补齐行的 input_ids 也清 0：embedding 查表要求 id 合法。
                self.input_ids[bs:n_row].zero_()
            _h2d(self.remap, self._dec_remap, n_row)
            idx = self.remap[:n_row]
            valid = idx >= 0
            # clamp 只是为了索引合法；无效行最后会被 where 丢掉。
            picked = prev_sampled[idx.clamp(min=0)]
            self.input_ids[:n_row].copy_(
                torch.where(valid, picked, self.input_ids[:n_row])
            )

        # 序列 id 序列：下面两个同步函数靠它判断「换批了没有」。
        sid = tuple(s.seq_id for s in seqs)
        self._sync_block_tables(seqs, sid)
        self._sync_sampling(seqs)
        # CG 档位补齐：多出来的槽位要显式写成安全值。
        if pad_to is not None and pad_to > bs:
            # 槽位 -1（写入跳过）、长度 0（不读 KV）、页表 -1（不访问物理页）。
            self.slot_mapping[bs:pad_to].fill_(-1)
            self.cache_seqlens[bs:pad_to].zero_()
            self.block_tables[bs:pad_to].fill_(-1)


    # prefill 组包：紧凑打包（不是 padding），并给出 Q / K 两条累积边界。
    def build_prefill(self, seqs: list[SeqSchedule]) -> tuple[int, int, int, bool]:
        ids = self._pre_ids_np
        positions = self._pre_pos_np
        slots = self._pre_slot_np
        cu_q = self._pre_cuq_np
        cu_k = self._pre_cuk_np
        # CSR 的第 0 项固定为 0。
        cu_q[0] = 0
        cu_k[0] = 0
        max_q = 0
        max_k = 0
        n_tokens = 0
        # 逐序列把真实 token 首尾相接塞进同一段缓冲 —— 不按最长序列 padding。
        for i, s in enumerate(seqs):
            # 起始位置 = 已缓存前缀长度。不是 0 —— 这是 chunked prefill / 前缀命中下的关键点。
            start = s.num_cached_tokens
            # 本步这条序列要算多少 token。
            n = len(s.token_ids)
            end = start + n
            # 整段切片赋值：走 numpy，比 Python 逐元素循环快得多。
            ids[n_tokens : n_tokens + n] = s.token_ids
            # 位置从 start 开始连续排。
            positions[n_tokens : n_tokens + n] = np.arange(start, end)
            # 槽位：把一个区间内的逻辑位置批量换算成物理槽位。
            slots[n_tokens : n_tokens + n] = slots_for_range(
                s.block_table, start, end, self.block_size
            )
            # Q 边界：只累计本步真正要算的 token。
            cu_q[i + 1] = cu_q[i] + n
            # K 边界：把已缓存的前缀也累计进去 —— attention 必须看得到全部历史。
            cu_k[i + 1] = cu_k[i] + end
            # 记下最长 Q / 最长 K：flash-attn 需要它们来切块。
            max_q = max(max_q, n)
            max_k = max(max_k, end)
            n_tokens += n

        bs = len(seqs)
        _h2d(self.input_ids, self._pre_ids, n_tokens)
        _h2d(self.positions, self._pre_pos, n_tokens)
        _h2d(self.slot_mapping, self._pre_slot, n_tokens)
        n_cu = bs + 1
        _h2d(self.cu_seqlens_q, self._pre_cuq, n_cu)
        _h2d(self.cu_seqlens_k, self._pre_cuk, n_cu)

        # K 总量 > Q 总量 ⇒ 这批有历史缓存 ⇒ 要走 paged 路径并上传页表。
        has_cache = int(cu_k[bs]) > int(cu_q[bs])
        sid = tuple(s.seq_id for s in seqs)
        # 没有历史缓存就不需要页表；同时把 id 清空，强制下次整表重发。
        if has_cache:
            self._sync_block_tables(seqs, sid)
        else:
            self._gpu_bt_ids = ()
        self._sync_sampling(seqs)
        return n_tokens, max_q, max_k, has_cache

    # 页表同步。这是全套代码里最「抠」的一处：用指纹判断整行是否变化，尽量只拷变化的部分。
    def _sync_block_tables(
        self, seqs: list[SeqSchedule], sid: tuple[int, ...]
    ) -> None:
        bs = len(seqs)
        # 序列集合没变，才谈得上复用上一批的指纹。
        same = sid == self._gpu_bt_ids
        dirty: list[int] = []
        # 三个本地引用：循环里要反复访问，先取出来省属性查找。
        fp = self._row_fp
        cpu = self._bt_cpu
        # 逐行判断：没变就整行不发，只多了一页就只补那一格。
        for i, s in enumerate(seqs):
            bt = s.block_table
            k = len(bt)
            # 上一批这一行的指纹。
            first, last, prev_k = fp[i]
            # 换批了 → 指纹作废，强制重发。
            if not same:
                prev_k = -1
            # 长度与首尾页都没变 ⇒ 中间不可能变（页表只会追加）⇒ 跳过。
            elif prev_k == k and (k == 0 or (bt[0] == first and bt[-1] == last)):
                continue
            if (
                prev_k == k - 1
                and k > 0
                and (k == 1 or (first == bt[0] and last == bt[k - 2]))
            ):
                # 只写最后一格，不动整行。
                cpu[i, k - 1] = bt[-1]
            else:
                cpu[i, :k] = bt
                cpu[i, k:] = -1
            # 更新指纹，供下一步比对。
            fp[i] = (bt[0] if k else -1, bt[-1] if k else -1, k)
            dirty.append(i)
        self._gpu_bt_ids = sid
        if not dirty:
            return
        # 脏行多、或换过批：一次整表拷贝更划算（避免几十次小拷贝）。
        if len(dirty) > 8 or not same:
            self.block_tables[:bs].copy_(self._bt_pin[:bs], non_blocking=True)
            return
        # 脏行少：逐行拷代价更低。
        for i in dirty:
            self.block_tables[i].copy_(self._bt_pin[i], non_blocking=True)

    # 采样参数同步：先用 (seq_id, 四个参数) 做快照比对，完全没变就整段跳过。
    def _sync_sampling(self, seqs: list[SeqSchedule]) -> None:
        key = tuple(
            (s.seq_id, s.temperature, s.top_p, s.top_k, s.min_p) for s in seqs
        )
        # 快照一致 ⇒ 这一步一个字节都不用传。
        if key == self._samp_key:
            return
        self._samp_key = key
        bs = len(seqs)
        vocab = self.vocab_size
        # 四组参数各自的 CPU 暂存视图。
        temps = self._h_temp_np
        top_ps = self._h_top_p_np
        top_ks = self._h_top_k_np
        min_ps = self._h_min_p_np
        # 逐条归一化：把 None / 缺省翻译成 kernel 里能直接比较的数值。
        for i, s in enumerate(seqs):
            temps[i] = s.temperature
            # top_p 缺省 = 1.0（不裁剪）。
            top_ps[i] = 1.0 if s.top_p is None else s.top_p
            # top_k 缺省用 vocab 表示「不限制」—— 这样采样 kernel 里不用再判 None。
            top_ks[i] = (
                vocab if (s.top_k is None or s.top_k <= 0) else min(s.top_k, vocab)
            )
            # min_p 缺省 = 0.0（不裁剪）。
            min_ps[i] = 0.0 if s.min_p is None else s.min_p
        # 四组参数一次性拷上去。
        _h2d(self.temperatures, self._h_temp, bs)
        _h2d(self.top_ps, self._h_top_p, bs)
        _h2d(self.top_ks, self._h_top_k, bs)
        _h2d(self.min_ps, self._h_min_p, bs)

        # 下面用 numpy 在 CPU 上算「整批形态」，供采样器选分支。
        t = temps[:bs]
        # 纯 CPU 计算，零同步开销。
        k = top_ks[:bs]
        p = top_ps[:bs]
        m = min_ps[:bs]
        # 整批都贪心 → 可以直接 argmax，跳过全部概率计算。
        self.all_greedy = bool(np.all(t <= 1e-5))
        # 有任意一条贪心（混合批时采样器要按行分流）。
        self.any_greedy = bool(np.any(t <= 1e-5))
        # 只有真的有人限制了 top_k，才需要走 top-k 那段逻辑。
        self.use_top_k = bool(np.any(k < vocab))
        # 同理：top_p / min_p 各自独立判断。
        self.use_top_p = bool(np.any(p < 1.0))
        self.use_min_p = bool(np.any(m > 0.0))
        # 整批里最大的 top_k：给 kernel 定 workspace 上界。
        self.max_top_k = int(k.max()) if self.use_top_k else 0
