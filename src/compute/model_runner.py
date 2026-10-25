# ============================================================================
# 【本文件是做什么的】把「一次调度结果」真正跑成一次前向：装输入 → 跑模型 → 采样。
#     输入装配（numpy 视图 → pinned 缓冲 → GPU 张量，见 compute/batch.py）
#     attention 元信息构造（块表、seqlens、位置）
#     prefill / decode 两条前向路径 + 采样
#     CUDA Graph 的分档捕获与重放（piecewise：PP 的收发留在图外）
#
# 【为什么 prefill 与 decode 分开两条路径】两者形状与瓶颈完全不同：
#     prefill 是"一次算很多 token、算力受限"，走 varlen attention；
#     decode  是"一次算一个 token、访存受限"，走 paged KV，且最适合 CUDA Graph。
#     混在一条路径里会让两边都做不好的取舍。
#
# 【CUDA Graph 为什么要分档】图捕获时形状必须固定（batch 大小写进图里）。
#     于是按若干个 batch 档位各捕一张图，运行时不匹配的步就补到最近的档位
#     （多算几行 padding 换"不用重新捕获"）。
#
# ⚠️ PP 场景下 recv/send 必须在图**外**：通信与计算的重叠是流水线效率的来源，
#   把通信录进图会让它的发生时机被固定，失去重叠机会。
# ============================================================================

from __future__ import annotations

import os

import torch

from ..config import EngineConfig
from ..data import SchedulerOutput, SeqSchedule
from ..distributed import (
    IntermediateTensors,
    get_pp_group,
    get_tp_group,
    get_world_group,
    recv_tensors,
    send_tensors,
)
from .batch import Batch
from .cuda_graph import CUDAGraphRunner
from .kv_pool import KVPool
from .layers.attention import AttnInputs, AttnMetadata, attention_context
from .sampler import Sampler

# 投机验证的对齐诊断开关（KRITACE_SPEC_LOGITS=1 打开）。
#
# 用途：投机与基线不一致时，仅看 argmax 无法区分两种成因 ——
#   (a) 某个位置的 KV/上下文确实是错的（结构性 bug）
#   (b) kernel 或 bf16 的数值差异把 top-2 之间极小的差距翻转了
# 打印每个 query 位置的 top-3 logits 就能分开：差距极小说明是 (b)。
_SPEC_LOGITS = bool(os.environ.get("KRITACE_SPEC_LOGITS"))


# ---------------------------------------------------------------------------
# ZOS 观测（默认全关，关掉时只多一次 bool 判断）
#
#   KRITACE_ZOS_STATS=1  记"read_samples 到达时那份数据是不是已经好了"：
#                        ready 比例高 → 同步点在到达时已满足，等待确实被下一步盖住了
#   KRITACE_ZOS_GPU=1    每步在 run_forward 前后各记一个 cudaEvent，
#                        bubble = 步墙钟 - 平均 GPU 忙时
# ---------------------------------------------------------------------------
_STATS_ON = os.environ.get("KRITACE_ZOS_STATS", "0").strip().lower() not in ("0", "", "false", "no")
_STATS: dict[str, float] = {"calls": 0, "ready": 0, "wait_us": 0.0}

_GPU_ON = os.environ.get("KRITACE_ZOS_GPU", "0").strip().lower() not in ("0", "false", "no", "")
_GPU_EVENTS: list = []


def gpu_probe_begin():
    if not _GPU_ON:
        return None
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    _GPU_EVENTS.append([ev, None])
    return len(_GPU_EVENTS) - 1


def gpu_probe_end(idx):
    if idx is None:
        return
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    _GPU_EVENTS[idx][1] = ev


def gpu_stats_dump() -> None:
    if not _GPU_ON or not _GPU_EVENTS:
        return
    total = 0.0
    n = 0
    for pair in _GPU_EVENTS:
        if pair[1] is None:
            continue
        pair[1].synchronize()
        total += pair[0].elapsed_time(pair[1])
        n += 1
    if n:
        print(f"[gpu-stats] steps={n} gpu_total={total:.1f}ms avg_gpu={total / n:.2f}ms", flush=True)


def _note_ready(evt: torch.cuda.Event) -> None:
    """先问"到了没有"，再同步 —— ready 的比例就是"气泡被盖住"的直接证据。"""
    import time

    _STATS["calls"] += 1
    if evt.query():
        _STATS["ready"] += 1
        return
    t0 = time.perf_counter()
    evt.synchronize()
    _STATS["wait_us"] += (time.perf_counter() - t0) * 1e6


def zos_stats_dump() -> str:
    calls = int(_STATS["calls"])
    ready = int(_STATS["ready"])
    if not calls:
        return "zos-stats: no samples read"
    wait_ms = _STATS["wait_us"] / 1000.0
    return (
        f"[zos-stats] calls={calls} ready={ready} ({100.0 * ready / calls:.1f}% 无需等待) "
        f"total_wait={wait_ms:.1f}ms avg_wait={wait_ms / calls:.3f}ms"
    )


# 一轮前向的执行器。构造期就把三类重活干完（预分配 / 捕获 graph / 编译 prefill），
# 这样进入推理期后不再有任何"第一次"的开销，延迟才不会抖。
class ModelRunner:
    # 构造期做完三件事：预分配输入张量、捕获 CUDA Graph、编译 prefill 图。
    # 顺序无所谓，但**都必须在第一个请求进来之前完成** ——
    # 捕获 graph 要几百毫秒、编译要几百毫秒到几秒，放到推理期就是一次巨大卡顿。
    def __init__(
        self,
        model,
        kv: KVPool,
        sampler: Sampler,
        *,
        config: EngineConfig,
        device: torch.device,
    ) -> None:
        # ---------- 依赖注入 ----------
        # model 是已经加载好、可能已被 compile/trace 包装过的模型；
        # kv 提供页池与页表（block_size / 页数 / dtype 都从它身上问出来）。
        self.model = model
        self.kv = kv
        self.sampler = sampler.to(device=device)
        self.config = config
        self.device = device
        self.block_size = kv.block_size
        self.kv_cache = kv.kv_cache
        self.dtype = kv.dtype
        self.hidden_size = config.hf_config.hidden_size
        # Attention 需要三样东西才跑得起来：KV cache 本体、页表（谁在第几页）、以及
        # "本轮是 prefill 还是 decode"等元信息。前两样在张量里传，第三样走全局上下文。
        self._meta = AttnMetadata(kv_cache=self.kv_cache)

        pp = get_pp_group()
        tp = get_tp_group()
        self._pp_group = pp
        self.is_first_pp = pp.is_first_rank
        self.is_last_pp = pp.is_last_rank
        self.is_driver = self.is_last_pp and tp.group_rank == 0

        # ---------- 输入张量的预分配（v3「输入管线」优化的落点）----------
        # 三行分别算：最大序列数、单序列最多占多少页、以及下面 Batch 的总容量。
        # max_bs 要 min(max_num_seqs, 256) —— 因为 CUDA Graph 要为**每个** batch size
        # 各捕获一张图。上限开太大（比如 1024）会把启动时间与显存拖垮。
        max_bs = min(config.max_num_seqs, 256)
        max_blocks = (config.context_len + self.block_size - 1) // self.block_size
        self.max_bs = max_bs
        # 一次把整个输入缓冲池建好：input_ids / positions / slot_mapping / 页表 / 采样参数
        # 全部是预分配的持久张量。之后每轮只是往这些张量里**写**数据，不再有分配动作。
        self.batch = Batch(
            max_bs=max_bs,
            max_tokens=max(config.max_num_batched_tokens, max_bs),
            max_blocks=max_blocks,
            vocab_size=config.hf_config.vocab_size,
            device=device,
            block_size=self.block_size,
        )

        # piecewise CUDA Graph（对齐 vLLM #35162）：
        #   图外 recv → 图内 forward（读写静态 buffer）→ 图外 send
        # 与「整段 recv+层+send 录进一张图」相比：P2P 更稳，调度可与图解耦。
        self._pp_recv = self._make_pp_buffer(max_bs) if not self.is_first_pp else None
        self._pp_send = self._make_pp_buffer(max_bs) if not self.is_last_pp else None
        self._decode_out = (
            torch.zeros(max_bs, self.hidden_size, dtype=self.dtype, device=device)
            if self.is_last_pp
            else None
        )

        # 采样结果异步 D2H：独立 copy stream + pinned buffer，采样 kernel 与拷贝解耦。
        # ---------- 采样结果的异步回传 ----------
        # 采样在 GPU 上算完只得到几十个 token id，但 CPU 必须拿到它才能推进请求状态。
        # 朴素写法（直接同步取回）会让整条流水线停下来等 GPU 算完采样。
        # 这里的做法：专用 copy stream + 锁页内存，让"拷回"与"下一轮前向"重叠起来。
        # ZOS：采样结果必须**双缓冲**。
        #   slot=0 是"下一步要收的"，slot=1 是"刚发射、正在 GPU 上算的"。
        # 只有一块的话，第 t 步发射时会把第 t-1 步还没取走的 token 覆盖掉。
        self._copy_event = [torch.cuda.Event(), torch.cuda.Event()]
        self._sample_cpu = [
            torch.empty(max_bs * 2, dtype=torch.int64, pin_memory=True),
            torch.empty(max_bs * 2, dtype=torch.int64, pin_memory=True),
        ]
        # 下一批采样结果该写进哪一块。每次 _stage_samples 后翻转。
        self._stage_slot = 0
        # GPU 上"上一批的采样结果"。ZOS 在途行要按 remap 从这里 gather 输入 token，
        # 所以它必须留在显存里，而且必须在下一批 build_decode 之前不被覆盖。
        self._sample_dev = torch.empty(max_bs * 2, dtype=torch.int64, device=device)

        # ---------- CUDA Graph ----------
        # enforce_eager=True 时完全不捕获（调试/降级用）。捕获在 __init__ 里就做完。
        self.cuda_graph_runner: CUDAGraphRunner | None = None
        if not config.enforce_eager:
            # capture 路径含 TP all-reduce / PP 边界，各 rank 先对齐再录，
            # 避免先到的 rank 卡在 warmup 的 NCCL 集合通信。
            world = get_world_group()
            if world.size > 1:
                world.barrier()
            self.cuda_graph_runner = CUDAGraphRunner(
                CUDAGraphRunner.make_bs_list(max_bs), device
            )
            self.cuda_graph_runner.capture(self._decode_body)
            if world.size > 1:
                world.barrier()

        self.prefill_fn = None
        if config.torch_compile:
            self.prefill_fn = self._build_compiled_prefill()

    # PP 边界的中转缓冲：一段 stage 算完，把 hidden 与 residual 一起交给下一段。
    def _make_pp_buffer(self, n: int) -> IntermediateTensors:
        # 两个张量都要传 —— Transformer 的残差是两条线，少传一条下一段就算错。
        return IntermediateTensors(
            hidden_states=torch.empty(
                n, self.hidden_size, dtype=self.dtype, device=self.device
            ),
            residual=torch.empty(
                n, self.hidden_size, dtype=self.dtype, device=self.device
            ),
        )

    # 下游 / 上游的**组内**序号：NCCL 点对点认的是这个口径，不是 WORLD rank。
    def _pp_dst(self) -> int:
        return self._pp_group.next_group_rank

    # 上游同理。
    def _pp_src(self) -> int:
        return self._pp_group.prev_group_rank

    # 把本段的输出推给下一段。
    def _send_intermediate(self, out: IntermediateTensors) -> None:
        send_tensors(
            (out["hidden_states"], out["residual"]),
            self._pp_dst(),
            self._pp_group,
        )

    # 接收上游的中间结果。static=True 时复用常驻缓冲（CG 路径要求地址固定）。
    def _recv_intermediate(self, n: int, *, static: bool = False) -> IntermediateTensors:
        # CG 重放必须用同一块内存 → 复用捕获时的缓冲；eager 路径可以现开。
        buf = self._pp_recv if static else self._make_pp_buffer(n)
        h = buf["hidden_states"][:n]
        r = buf["residual"][:n]
        # 阻塞式点对点接收（内部对每个张量逐个 wait）。
        recv_tensors((h, r), self._pp_src(), self._pp_group)
        return IntermediateTensors(hidden_states=h, residual=r)

    @property
    def sample_dev(self) -> torch.Tensor:
        """显存里的「上一批采样结果」。ZOS 在途行的 gather 源，也是 PP 广播的对象。"""
        return self._sample_dev

    @torch.inference_mode()
    def run_forward(self, sched: SchedulerOutput):
        """一步 sync 前向：decode 再 prefill；PP 为 lockstep 切层，非 1F1B。

        非末 PP stage 只做本段计算 + P2P，返回 None；末 stage 异步 D2H 采样 id。
        """
        # prefill 与 decode 可以同时存在（连续批处理的体现）：新请求进 prefill、
        # 老请求接着 decode，两批分别跑，最后把返回的 id 合到一起。
        # 两批分开跑：deocde 每条 1 个 token、prefill 每条多个 token，形状本就不同。
        _probe = gpu_probe_begin()
        d = self._run_decode(sched.decodes) if sched.decodes else None
        # 同一轮里 prefill 与 decode 各跑一次 —— 这就是连续批处理的体现。
        p = self._run_prefill(sched.prefills) if sched.prefills else None
        # 非末段没有输出（logits 与采样都在末段），直接返回 None。
        if not self.is_last_pp:
            return None
        # 末段才做异步 D2H：这里只排队，真正取数交给 read_samples。
        out = self._stage_samples(p, d)
        gpu_probe_end(_probe)
        return out

    # decode 的 attention 输入：槽位 + 每条的 KV 长度 + 页表。
    def _decode_attn(self, bs: int) -> AttnInputs:
        return AttnInputs(
            slot_mapping=self.batch.slot_mapping[:bs],
            is_prefill=False,
            cache_seqlens=self.batch.cache_seqlens[:bs],
            block_tables=self.batch.block_tables[:bs],
        )

    # CG 图里执行的**纯计算体**：不含 PP 的收 / 发，收发留在图外。
    def _decode_body(self, bs: int):
        """仅本地 compute；PP recv/send 在 replay 前后执行。"""
        # 非首段要先拿到上游的 hidden 与 residual。
        it = None
        # 收上游：复用捕获时的静态缓冲。
        if not self.is_first_pp:
            assert self._pp_recv is not None
            it = IntermediateTensors(
                hidden_states=self._pp_recv["hidden_states"][:bs],
                residual=self._pp_recv["residual"][:bs],
            )
        # 把 attention 上下文挂上：模型内部的 attention 层从全局取 KV cache 和页表，
        # 而不是每层都接一遍参数（省掉一整条透传链）。
        # 用全局上下文把 KV cache 与页表交给模型内部，避免层层透传参数。
        with attention_context(self._meta):
            out = self.model(
                self.batch.input_ids[:bs],
                self.batch.positions[:bs],
                self._decode_attn(bs),
                it,
            )
        # 末段：把输出写进常驻输出缓冲 —— 图里的写入地址必须固定。
        if self.is_last_pp:
            self._decode_out[:bs] = out
            return self._decode_out[:bs]
        # 非末段：把本段结果拷进发送缓冲，等图外发送。
        assert self._pp_send is not None
        # 拷进常驻缓冲 —— 收发都在图外做，所以这里只能是 memcpy，不能换指针。
        self._pp_send["hidden_states"][:bs].copy_(out["hidden_states"])
        self._pp_send["residual"][:bs].copy_(out["residual"])
        return out

    # decode 一步。三条路径：CG 重放 / eager / 空批。
    def _run_decode(self, seqs: list[SeqSchedule]):
        if not seqs:
            return None
        bs = len(seqs)
        # 先算 CG 档位：真实 bs 要向上取整到已录制的档位。
        pad_to = (
            # 档位只影响 padding，不改语义。
            self.cuda_graph_runner._pad_bs(bs)
            if self.cuda_graph_runner is not None and bs <= self.max_bs
            else None
        )
        # 组包时顺手把 padding 区写成安全值。
        self.batch.build_decode(seqs, pad_to=pad_to, prev_sampled=self._sample_dev)

        # CG 路径：收 → 重放 → 发，三步里只有中间那步在图里。
        if self.cuda_graph_runner is not None and bs <= self.max_bs:
            if not self.is_first_pp:
                self._recv_intermediate(bs, static=True)
            # 一次重放跑完整步。
            hidden, _ = self.cuda_graph_runner.replay(bs)
            # 非末段：把结果发给下一段就结束，不参与采样。
            if not self.is_last_pp:
                self._send_intermediate(
                    IntermediateTensors(
                        hidden_states=self._pp_send["hidden_states"][:bs],
                        residual=self._pp_send["residual"][:bs],
                    )
                )
                return None
            # 只取真实的那几行 —— 图是按档位跑的，padding 部分要丢掉。
            hidden = hidden[:bs]
        else:
            # eager 路径：现开缓冲收上游。
            it = self._recv_intermediate(bs) if not self.is_first_pp else None
            with attention_context(self._meta):
                # eager 前向：与 graph 路径**同一个 model 调用**，输入切片也一样。
                hidden = self.model(
                    self.batch.input_ids[:bs],
                    self.batch.positions[:bs],
                    self._decode_attn(bs),
                    it,
                )
            # 非末段：发完就返回。
            if not self.is_last_pp:
                self._send_intermediate(hidden)
                return None

        # 只有末段会走到这里：出 logits 或直接 argmax。
        return self._sample_logits(hidden, bs)

    # prefill 的 attention 输入：多两条 CSR 边界 + 最长 Q/K（flash-attn 要用它们切块）。
    def _prefill_attn(
        self,
        bs: int,
        n_tokens: int,
        max_q: int,
        max_k: int,
        has_cache: bool,
    ) -> AttnInputs:
        return AttnInputs(
            slot_mapping=self.batch.slot_mapping[:n_tokens],
            is_prefill=True,
            cu_seqlens_q=self.batch.cu_seqlens_q[: bs + 1],
            cu_seqlens_k=self.batch.cu_seqlens_k[: bs + 1],
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            # 有前缀命中才需要页表；纯首 chunk 没有历史页。
            block_tables=self.batch.block_tables[:bs] if has_cache else None,
        )

    # prefill 一步：紧凑打包 →（编译图 / eager）→ 末段采样。
    def _run_prefill(self, seqs: list[SeqSchedule]):
        if not seqs:
            return None
        # 输入管线返回 4 个量：总 token 数、最长 Q、最长 K、以及是否有请求命中了前缀。
        # "最长 K"与"最长 Q"是两个不同的值 —— 只要有一个请求命中前缀，K 就会长于 Q。
        n_tokens, max_q, max_k, has_cache = self.batch.build_prefill(seqs)
        bs = len(seqs)
        # 非首段先收上游。
        it = self._recv_intermediate(n_tokens) if not self.is_first_pp else None

        # 走编译图需要同时满足两个条件：
        #   ① 配置里开了 torch.compile 并且已经构建成功
        #   ② 这批请求**没有任何前缀缓存命中**
        # 第 ② 条是关键约束：一旦命中前缀，Q 的长度（本轮新增）和 K 的长度（含前缀）
        # 就不再相等，而编译图把形状写死了（见 _prefill_forward）→ 会算出错误结果。
        # 走编译图的两个前提，第 ② 条这里再强调一次：**没有任何前缀命中**。
        # 走编译图的两个前提：开了 compile，且**这一批没有任何前缀命中**。
        if self.prefill_fn is not None and all(
            s.num_cached_tokens == 0 for s in seqs
        ):
            with attention_context(self._meta):
                # 编译图只收 5 个参数，没有 cu_seqlens_k / block_tables ——
                # 因为它的语义就是"Q 与 K 等长、没有历史页"，多传反而暴露它不支持。
                hidden = self.prefill_fn(
                    self.batch.input_ids[:n_tokens],
                    self.batch.positions[:n_tokens],
                    self.batch.slot_mapping[:n_tokens],
                    self.batch.cu_seqlens_q[: bs + 1],
                    max_q,
                    it,
                )
        else:
            with attention_context(self._meta):
                hidden = self.model(
                    self.batch.input_ids[:n_tokens],
                    self.batch.positions[:n_tokens],
                    self._prefill_attn(bs, n_tokens, max_q, max_k, has_cache),
                    it,
                )

        # 非末段：把结果发下去就结束 —— 它不持有 lm_head，出不了 token。
        if not self.is_last_pp:
            self._send_intermediate(hidden)
            return None

        # 投机验证：需要**每个位置**的输出，而不是只取最后一个。
        # 走 prefill 通道是必然的 —— 只有 varlen attention 能一次算 K+1 个 query；
        # decode 通道的 flash_attn_with_kvcache 每序列只接受 1 个。
        if any(s.spec_draft_ids for s in seqs):
            return self._spec_verify(hidden, seqs)

        # 有些序列本轮只是"凑数"的（比如 chunked prefill 的中间块），
        # 它们不需要出 token，就不用浪费一次词表投影 + 采样。
        if not any(s.need_sample for s in seqs):
            # 非 driver rank（TP>1 时的其它卡）不重复产出 id。
            if not self.is_driver:
                return None
            # 不生成 token
            return torch.zeros(bs, dtype=torch.int64, device=self.device)
        # 只取每条序列的最后一个 token 出 logits：prefill 的中间 token 后面还有上下文，
        # 不需要预测下一个词。cu_seqlens 是累加边界，减 1 就落在各自最后一位上。
        last = self.batch.cu_seqlens_q[1 : bs + 1] - 1
        return self._sample_logits(hidden[last], bs)

    def _spec_verify(self, hidden: torch.Tensor, seqs: list[SeqSchedule]):
        """投机验证：取每条序列每个位置的 argmax，返回**一维**扁平张量。

        位置对齐（详见 ``spec/engine.py`` 顶部注释）：
        输出位置 i 对应"输入位置 i 的下一个 token"的预测，于是

            j = 第一个与草稿不匹配的下标        （全匹配则 j = K）
            emitted = draft[:j] + [argmax[j]]

        各序列的草稿长度可能不同（n-gram 命中情况不同），统一 pad 到 K+1，
        pad 位填 -1 —— 它不等于任何真实 token，贪心判定会在那里自然停下。

        ⚠️ 返回的是一维（``[bs * width]``）而不是 ``[bs, width]``：
        下游 ``_stage_samples`` 按 ``numel()`` 做 D2H 拷贝、
        ``read_samples`` 按**元素数**切片，两者都只认扁平布局。
        调度器侧用同样的 ``width`` 做定宽切片（``i*width : (i+1)*width``），
        两端必须一致。

        ⚠️ 本批**必须全是投机序列**。定宽切片依赖"每条序列占 width 个元素"
        这一不变量，混入普通 prefill（每序列只回传 1 个样本）会让布局错位。
        由调度器保证：有草稿的那一步不再拉入新请求。
        """
        if not self.is_driver:
            return None
        assert all(s.spec_draft_ids for s in seqs), (
            "spec verify 的 batch 里混入了非投机序列，定宽扁平布局会错位"
        )
        k = max(len(s.spec_draft_ids) for s in seqs)
        width = k + 1
        out = torch.full(
            (len(seqs), width), -1, dtype=torch.int64, device=hidden.device
        )
        cu = self.batch.cu_seqlens_q
        if _SPEC_LOGITS:
            n_tok = int(cu[len(seqs)])
            print(
                f"[spec-dbg] bs={len(seqs)} width={width} "
                f"cu_q={self.batch.cu_seqlens_q[: len(seqs) + 1].tolist()} "
                f"cu_k={self.batch.cu_seqlens_k[: len(seqs) + 1].tolist()} "
                f"pos={self.batch.positions[:n_tok].tolist()} "
                f"slots={self.batch.slot_mapping[:n_tok].tolist()} "
                f"cache_seqlens={[s.num_cached_tokens for s in seqs]}",
                flush=True,
            )
        for i, s in enumerate(seqs):
            n = len(s.token_ids)
            seg = hidden[cu[i] : cu[i + 1]]
            assert seg.shape[0] == n, (
                f"spec verify: hidden 段长 {seg.shape[0]} 与输入长度 {n} 不符"
            )
            if _SPEC_LOGITS:
                # 只在这一步打印：如果某个位置的 top-2 差距极小，说明是 kernel
                # 数值差异；如果差距很大，说明上下文（KV）真的是错的。
                lg = self.model.compute_logits(seg).float()
                top = torch.topk(lg, 3, dim=-1)
                print(f"[spec-dbg] seq={s.seq_id} tokens={s.token_ids}", flush=True)
                for r in range(seg.shape[0]):
                    pair = "  ".join(
                        f"{int(v)}@({float(w):+.3f})"
                        for w, v in zip(top.values[r], top.indices[r])
                    )
                    print(f"[spec-dbg]   row{r}: {pair}", flush=True)
            ids = self.model.argmax(seg)
            take = min(n, width)
            out[i, :take] = ids[:take]
        return out.view(-1)

    # 末段的出口：整批贪心走 argmax，否则算 logits 再采样。
    def _sample_logits(self, hidden: torch.Tensor, bs: int) -> torch.Tensor | None:
        # 整批贪心时直接 argmax，跳过整个词表投影与概率计算。
        if self.batch.all_greedy:
            # 这是 decode 的默认路径，省下的正是最大的一次矩阵乘。
            ids = self.model.argmax(hidden)
            if not self.is_driver:
                return None
            return ids
        # 只需要"每序列最后一个 token"的 hidden 去出 logits —— 词表投影是个大矩阵乘
        # （hidden × vocab），单独拎出来也是为了让它可以被独立融合/优化。
        logits = self.model.compute_logits(hidden)
        if not self.is_driver:
            return None
        return self._sample(bs, logits)

    # ======================== 采样 ========================
    # 采样参数（温度 / top-p / top-k / min-p）**按请求**打包成张量一起送进 kernel，
    # kernel 内部按行各取各的参数 —— 好处是不用"按参数分组跑多趟"。
    # 另外，"这批是不是全都 greedy"这类判断改由 CPU 侧预先算好一个布尔量传进来，
    # 避免为了在 forward 里做分支而触发 GPU→CPU 同步（一次同步能吃掉几十微秒）。
    def _sample(self, bs: int, logits: torch.Tensor) -> torch.Tensor:
        # 一次性把 logits 和**每请求各自的**采样参数传进去。
        return self.sampler(
            logits,
            self.batch.temperatures[:bs],
            top_ps=self.batch.top_ps[:bs],
            top_ks=self.batch.top_ks[:bs],
            min_ps=self.batch.min_ps[:bs],
            all_greedy=self.batch.all_greedy,
            any_greedy=self.batch.any_greedy,
            use_top_k=self.batch.use_top_k,
            use_top_p=self.batch.use_top_p,
            use_min_p=self.batch.use_min_p,
            max_top_k=self.batch.max_top_k,
        )

    # ---------- 采样结果的两段式处理：先异步拷出，用时再同步读回 ----------
    # 拆两段的目的就是让"拷出"和"下一轮计算"重叠：run() 只把拷贝排进 stream 立刻返回，
    # 直到 Engine 真的需要这些 id 去推进请求状态时，才在 read_samples 里阻塞。
    def _stage_samples(
        self, p: torch.Tensor | None, d: torch.Tensor | None
    ) -> tuple[int, int, int]:
        n_p = 0 if p is None else p.numel()
        n_d = 0 if d is None else d.numel()
        need = n_p + n_d
        slot = self._stage_slot
        buf = self._sample_cpu[slot]
        # 单步真正的 token 数可能超过预分配容量（大批 prefill），不够就扩容。
        if need > buf.numel():
            buf = torch.empty(need, dtype=torch.int64, pin_memory=True)
            self._sample_cpu[slot] = buf

        # ① 先留一份在显存里：下一批的在途行要从这里 gather 自己的输入 token。
        #    这一步必须在主流上、且在采样之后 —— 和采样核天然同流，不用额外同步。
        if need:
            if need > self._sample_dev.numel():
                self._sample_dev = torch.empty(need, dtype=torch.int64, device=self.device)
            if d is not None:
                self._sample_dev[:n_d].copy_(d)
            if p is not None:
                self._sample_dev[n_d : n_d + n_p].copy_(p)

        # ② 再拷一份到锁页内存，供 CPU 读。
        # 【必须留在主流上】用独立拷贝流时，下一步的前向会和这次 D2H 并发，而 d/p
        # 指向的显存缓冲下一步就被复写 —— 拷到一半被改，读出来就是垃圾 token。
        # ZOS 下"下一步"离得特别近，所以这个冒险从"几乎撞不上"变成"每步必撞"。
        if d is not None:
            buf[:n_d].copy_(d, non_blocking=True)
        if p is not None:
            buf[n_d : n_d + n_p].copy_(p, non_blocking=True)
        # 打一个事件，供 read_samples 那边等它。
        self._copy_event[slot].record()

        # 本块待取，下一批换另一块 —— 两块轮转，保证"刚发射的"不会覆盖"还没取走的"。
        self._stage_slot = 1 - slot
        return n_p, n_d, slot

    # 与上一个函数配对：这里才是"真的要用了"的时刻，才阻塞等拷贝完成。
    def read_samples(self, n_p: int, n_d: int, slot: int) -> tuple[list[int], list[int]]:
        # 统计口径：先问"到了没有"，再同步。
        if _STATS_ON:
            _note_ready(self._copy_event[slot])
        # 真正的阻塞点：等对应那一块的事件完成，数据一定已经在 CPU 的锁页内存里。
        self._copy_event[slot].synchronize()
        buf = self._sample_cpu[slot]
        # 布局要和拷入时一致：decode 在前、prefill 在后（见 _stage_samples 的切片顺序）。
        p_ids = buf[n_d : n_d + n_p].tolist() if n_p else []
        d_ids = buf[:n_d].tolist() if n_d else []
        return p_ids, d_ids

    # 有 TP 时不能用 cudagraph 模式：compile 自带的 cudagraph 会和 TP 集合通信打架。
    def _resolve_compile_mode(self) -> str:
        """单卡 / 仅 PP 用用户 mode；含 TP 时用 max-autotune-no-cudagraphs。"""
        # 单卡或纯 PP 才用配置里的 mode。
        if get_tp_group().size == 1:
            return self.config.compile_mode
        # 含 TP：换成 max-autotune-no-cudagraphs。
        return "max-autotune-no-cudagraphs"

    # 编译图的"内核"：torch.compile 实际包起来的就是这一段前向。
    # 它只适用于"没有前缀命中"的 prefill —— 因为这里 Q 与 K 长度相等，
    # 形状被编译图写死；一旦命中前缀，K 会更长，这张图就不再正确。
    def _prefill_forward(
        self,
        input_ids,
        positions,
        slot_mapping,
        cu_seqlens,
        max_seqlen,
        intermediate_tensors,
    ):
        # 正常 model 前向路径
        # eager 路径要自己组 AttnInputs；graph 路径里这份输入是捕获时固化好的那一份。
        attn = AttnInputs(
            slot_mapping=slot_mapping,
            is_prefill=True,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
        )
        return self.model(input_ids, positions, attn, intermediate_tensors)

    # ======================== prefill 编译图 ========================
    # 【它到底省掉了什么】prefill 一轮要下发几十上百个 kernel，每个 kernel 在 Python 侧
    # 都要走一遍框架开销（取参数、查 dispatch、建积木）。形状多变的场景下，这部分开销
    # 可以和 kernel 本身的计算时间同量级 —— compile 把它压成"一次调用 + 一张图"。
    def _build_compiled_prefill(self):
        # mode / dynamic 两个开关决定"编译多激进"与"形状变化时要不要重编译"，详见 08-Prefill-Compile。
        # mode 决定编译多激进，dynamic 决定形状变化时要不要重编译。
        fn = torch.compile(
            self._prefill_forward,
            mode=self._resolve_compile_mode(),
            dynamic=self.config.compile_dynamic,
        )
        # 假输入
        # 假输入的规模取 min(256, context_len)：够大到能覆盖典型形状，又不会热身很久。
        # 规模取 min(256, context_len)：够覆盖典型形状，热身又不会太久。
        n = min(256, self.config.context_len)
        ids = torch.arange(1, n + 1, dtype=torch.int64, device=self.device)
        pos = torch.arange(n, dtype=torch.int64, device=self.device)
        cu = torch.tensor([0, n], dtype=torch.int32, device=self.device)
        sm = torch.arange(n, dtype=torch.int32, device=self.device)
        # PP 非首段的热身也要带上中间输入，否则图和真实路径的签名不一致。
        it = self._make_pp_buffer(n) if not self.is_first_pp else None
        # 记录阶段：torch.compile 是"首次调用时才编译"的，所以必须先用一批假输入跑通一次把
        # 编译触发掉。否则第一个真实请求会替我们付这笔编译开销（几百毫秒到几秒的卡顿）。
        # 热身是**真的**在写 KV cache（假输入也是真 token），完了必须 zero_() 清干净，
        # 否则残留的垃圾数据会被后续请求的 attention 读到 → 输出静默错乱。
        # 必须真的跑一次 —— torch.compile 是首次调用才编译的。
        with attention_context(self._meta), torch.inference_mode():
            fn(ids, pos, sm, cu, n, it)
        # 同步：确保热身真正跑完，再去清 KV。
        torch.cuda.synchronize()
        # 这行是必须的：热身真的往 KV cache 里写了东西（假 token 也是 token），
        # 不清零的话这些残留数据会被后续请求的 attention 当成"前文"读进去。
        # 清掉热身写进去的假 KV。
        self.kv_cache.zero_()
        # 不清的话，这些残留会被后续请求的 attention 当成前文读进去 → 输出静默错乱。
        return fn

    # ======================== 资源释放 ========================
    # 释放：CG 图与编译函数都要断引用。
    def destroy(self) -> None:
        if self.cuda_graph_runner is not None:
            self.cuda_graph_runner.destroy()
            self.cuda_graph_runner = None
        self.prefill_fn = None
