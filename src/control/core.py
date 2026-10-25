# ============================================================================
# 【本文件是做什么的】EngineCore = 引擎的核心循环：
#     调度（选谁跑）→ 执行（跑前向）→ 采样（取 token）→ 后处理（推进状态）
# 循环往复，一步一层。它夹在门面 Engine 与执行器 Executor 之间。
#
# 【它的边界在哪】两件事它都不管：
#     ① 请求怎么来（文本怎么变成 token）—— 那是门面 Engine 的事；
#     ② 张量怎么算、在几个进程上算 —— 那是 Executor 的事。
#   EngineCore 只认 ScheduledBatch 与采样结果这两个"接口数据结构"。
#
# 【为什么要 _MAX_IDLE_STEPS 这个计数器】调度器返回空批但请求还没跑完，
# 最常见的根因是 KV 池被占满：所有请求都进不来，于是每步都调度不出东西。
# 如果不主动报错，外层就会一直空转 —— 这是最难查的一类故障（进程活着、
# GPU 却是 0%）。所以宁可在这里抛异常，把"池耗尽"这个根因直接写进报错里。
#
# 【metric 是什么】本步处理的 token 数（prefill 的真实 token 数 + decode 的请求数），
# 上层拿它算吞吐。注意 prefill 要按实际调度的 token 数计：
# 命中前缀缓存之后，真正要算的 token 会少于原始 prompt 长度。
#
# 【读这个文件的顺序】
#     __init__ / _init_kv / _build_scheduler（KV 与调度器的初始化顺序）
#       → add_request / has_unfinished（请求管理，纯转发）
#       → step（唯一的循环体，看这四段就够）
#       → destroy（逆序释放）
# ============================================================================

from __future__ import annotations

import gc
import os
from contextlib import contextmanager

import torch

from ..compute.model_runner import gpu_stats_dump, zos_stats_dump

from ..config import EngineConfig
from ..data import EngineCoreOutput, KVManager, ScheduledBatch, Sequence
from .executor import create_executor
from .scheduler import Scheduler


# ==============================================================================
# EngineCore：推理引擎的核心循环。
#
# 位于 Engine 与 Executor 之间，负责：
#   驱动每一步的 调度 → 执行 → 采样 → 后处理 流程。
#   管理 KV cache 的生命周期（初始化、重建）。
#   监控调度器空闲状态，防止 KV 耗尽导致的静默死循环。
#
# EngineCore 只关心拿到 ScheduledBatch，交给 Executor 跑完，收回采样结果。
# ==============================================================================


class EngineCore:

    # 调度器连续返回空的最大容忍步数。
    # 若仍有未完成请求却持续无调度产出，大概率是 KV 池耗尽，直接报错。
    # 连续空转的容忍上限。1024 步足够区分「暂时没活」与「永远没活」。
    _MAX_IDLE_STEPS = 1024

    # 构造时就把执行器与 KV 都建起来 —— 引擎是个「一次建好、长期复用」的对象。
    def __init__(
        self,
        config: EngineConfig,
        *,
        eos_token_id: int,
        device: torch.device,
        dtype: torch.dtype,
        executor=None,
    ) -> None:
        # 配置与 eos 是循环里要用到的两样东西。
        self.config = config
        self.eos_token_id = eos_token_id

        # executor 可由外部注入（torchrun 场景下，rank0 的 MultiprocExecutor
        # 已在进程启动阶段创建好）。未注入时，由 create_executor 按配置构建。
        self.executor = executor or create_executor(config, device, dtype)

        # 空闲步数计数器：用来发现「一直调度不出东西」这种死锁。
        self._idle_steps = 0

        # ---------------------------------------------------------------- #
        # ZOS（zero-overhead scheduling）总开关。
        #
        # 为什么默认开：它是纯调度侧的改动，正确性由「逐 token 与同步模式一致」保证，
        # 只在性能上做加法（把「等采样」藏进下一次前向里）。
        # 为什么与投机互斥：投机下一步的起点是区间（接受长度 j 由 GPU 验证后才产生），
        # 而 ZOS 要求 CPU 在一步之前就把位置算死；两者叠在一起需要把坐标修正搬上 GPU，
        # 成本大于收益，所以这里直接退回同步路径并打印原因。
        # ---------------------------------------------------------------- #
        want_zos = os.environ.get("KRITACE_ZOS", "1").strip().lower() not in ("0", "false", "no")
        self.zos = want_zos and self.config.num_speculative_tokens == 0
        if want_zos and not self.zos:
            print(
                "[zos] 投机解码开启（num_speculative_tokens>0），ZOS 自动关闭："
                "接受长度 j 在 GPU 上产生，CPU 侧调度无法提前一步拿到。"
            )
        # nsys / Nsight 的打点开关，默认关（关时是零开销的空 contextmanager）。
        self._nvtx_on = os.environ.get("KRITACE_NVTX", "0").strip().lower() not in ("0", "", "false", "no")
        # 建 KV 与调度器。顺序不能反 —— 调度器要拿 KV 的块数。
        self._init_kv()

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #

    # 调度器需要 KVManager 的引用：调度时要查还剩多少块可用。
    def _build_scheduler(self) -> Scheduler:
        """
            构建调度器。
            调度器需要持有 KVManager 引用，用于在调度时查询剩余可用块数。
        """
        # 把准入相关的三个上限（并发数、单步 token 数、是否混排）一起交给调度器。
        return Scheduler(
            self.kv,
            max_num_seqs=self.config.max_num_seqs,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            eos_token_id=self.eos_token_id,
            mix_prefill_decode=self.config.mix_prefill_decode,
            # 【装配点】把两个投机配置从引擎配置下发给调度器：
            # K（每步草稿长度）与 n-gram 的最大后缀长度。
            # 注意草稿在调度器里生成（它需要拿到序列的 token_ids），
            # 而不是在 ModelRunner 里 —— 因为"这一步要预留多少 KV"必须在调度时就决定（K+1 而不是 1）。
            spec_k=self.config.num_speculative_tokens,
            spec_ngram_max_n=self.config.spec_ngram_max_n,
        )

    # KV 与调度器必须一起建：调度器的准入判断依赖块数。
    def _init_kv(self) -> None:
        """
            初始化 KV cache 并重建调度器。
            KV 池由 Executor 分配（涉及显存分配与块表构建），
            调度器依赖 KV 池的块数信息，因此两者需要一起初始化。
        """
        # 池由 Executor 分配（显存 + 块表都在那边），门面只拿块数。
        self.kv: KVManager = self.executor.init_kv()
        self.scheduler = self._build_scheduler()
        # 缓存命中量统计的初值。
        self.last_num_cached_tokens = 0

    # ------------------------------------------------------------------ #
    # 请求管理
    # ------------------------------------------------------------------ #

    # 入队。请求走到这里时已经 tokenize 完了。
    def add_request(self, seq: Sequence) -> None:
        """将一条已编码的请求加入调度队列。"""
        self.scheduler.add(seq)

    # 用调度器的完成状态回答，而不是自己数。
    def has_unfinished(self) -> bool:
        """是否还有未完成的请求（等待中或正在生成）。"""
        return not self.scheduler.is_finished()

    # ------------------------------------------------------------------ #
    # 核心循环
    # ------------------------------------------------------------------ #

    @contextmanager
    def _nvtx(self, name: str):
        """给 nsys / Nsight 打一个区间标记（KRITACE_NVTX=1 时生效，默认零开销）。"""
        if not self._nvtx_on:
            yield
            return
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()

    # 一个 decode step 的完整流程：调度 → 发射 → 取结果 → 回灌。
    def step(self) -> tuple[list[EngineCoreOutput], int]:
        """
            执行一步推理。

            ZOS 开（默认、且未开投机）时本函数只做三件事：
                ① 调度第 t 步 → ② 发射第 t 步的前向 → ③ 回头收第 t-1 步的结果。
            "收上一步"本来卡在 ① 之前（同步模式），现在挪到了 ② 后面 ——
            GPU 在算第 t 步的时候，CPU 才有机会去算第 t 步的调度。

            ZOS 关（KRITACE_ZOS=0、或开了投机）时退化成同步顺序：
                ① 调度 → ② 发射 → ③ 当场收本步 → ④ 后处理。

            Returns:
                (outputs, metric)
                - outputs：本步有增量产出的请求列表（新 token / 完成状态）。
                  ZOS 下它是"第 t-1 步"的产出，晚一步，但**一条都不会少**。
                - metric：本步处理的 token 总数（prefill tokens + decode 请求数）。
                  同理，ZOS 下它配的是"刚收的那一批"，不是"刚发射的那一批"。
        """
        # ---- 调度 ----
        # ZOS 下这里读到的是"上一批回灌之后"的状态：出队/入队都是新账，
        # 但在途那一批的 token 还是未知的 —— 它由调度器自己记着（_inflight_*）。
        with self._nvtx("zos.schedule"):
            sched: ScheduledBatch = self.scheduler.schedule()

        # 空批不等于出错：没请求时它也是空的。
        if sched.is_empty():
            # ZOS：发射不出新批，但可能还有一批在途没收回 —— 尾段必须把它收干净，
            # 否则最后那个 token 会丢。
            prev = self.scheduler.take_pending() if self.zos else None
            if prev is None:
                # 调度器没有产出，但仍有未完成请求：通常是 KV 池被占满。
                # 连续空闲超过阈值后主动报错，避免静默死循环。
                if not self.scheduler.is_finished():
                    self._idle_steps += 1
                    if self._idle_steps >= self._MAX_IDLE_STEPS:
                        raise RuntimeError(
                            "scheduler returned empty for too long while requests remain "
                            f"(idle_steps={self._idle_steps}). Likely KV pool exhaustion "
                            f"(free_blocks={self.kv.num_free})."
                        )
                return [], 0
            with self._nvtx("zos.read"):
                p_ids, d_ids = self.executor.read_samples(prev.n_p, prev.n_d, prev.slot)
            with self._nvtx("zos.finish"):
                return self.scheduler.finish(prev, p_ids, d_ids), prev.metric

        # 有活干就复位计数器。
        self._idle_steps = 0

        # ---- 执行（只发射，不等待）----
        # run 返回 (n_prefill, n_decode, slot)：slot 指明这次采样结果写进了哪一块
        # 双缓冲 —— ZOS 下 CPU 手上同时有两批，必须靠它区分。
        with self._nvtx("zos.launch"):
            n_p, n_d, slot = self.executor.run(sched)
        # 记录本步的缓存命中量，供上层统计。
        self.last_num_cached_tokens = self.kv.last_num_cached_tokens

        # 本步处理的 token 数：prefill 按实际调度数（命中前缀缓存后可能少于原始长度），
        # decode 每个请求只算 1 个 token。
        prefill_toks = sum(s.num_scheduled_tokens for s in sched.prefills)
        metric = prefill_toks + len(sched.decodes)

        # 把刚发射的一批登记为"在途"，换回上一批（可能为 None）。
        prev = self.scheduler.swap_pending(sched, n_p, n_d, slot, metric)
        if not self.zos:
            # 同步模式：发射完当场就把本批收回，等价于同步顺序的第三步。
            prev = self.scheduler.take_pending()

        # 第一步（或刚开完 pool）没有上一批可收，本步到此为止。
        if prev is None:
            return [], 0

        # ---- 采样与后处理（收的是上一批）----
        # 这里会阻塞等 D2H 完成。区别在于：此刻 GPU 上已经有第 t 步在跑了，
        # 气泡被"埋"在 GPU 的时间里，而不是赤裸地摊在两步之间。
        with self._nvtx("zos.read"):
            p_ids, d_ids = self.executor.read_samples(prev.n_p, prev.n_d, prev.slot)

        # 调度器回灌：追加 token、判断结束、回收已完成序列的 KV 块（推迟一步）。
        with self._nvtx("zos.finish"):
            outputs = self.scheduler.finish(prev, p_ids, d_ids)
        return outputs, prev.metric

    # ------------------------------------------------------------------ #
    # 资源释放
    # ------------------------------------------------------------------ #

    # 释放顺序：执行器 → 断引用 → GC → 清 CUDA 缓存。
    def destroy(self) -> None:
        """
            释放引擎核心资源。
            先销毁执行器（释放模型权重与 KV 显存），
            再断开对 KV 和调度器的引用，最后触发 GC 与 CUDA 缓存回收。
        """
        # 先拆执行器：模型权重与 KV 显存都在它手里。
        self.executor.destroy()
        self.kv = None
        self.scheduler = None
        # 观测收尾：把机制统计与 GPU 探针结果打出来（两个 env 都默认关）。
        if os.environ.get("KRITACE_ZOS_STATS", "0").strip().lower() not in ("0", "", "false", "no"):
            print(zos_stats_dump(), flush=True)
        if os.environ.get("KRITACE_ZOS_GPU", "0").strip().lower() not in ("0", "", "false", "no"):
            gpu_stats_dump()
        gc.collect()
        torch.cuda.empty_cache()