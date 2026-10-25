# ==============================================================================
# Engine：推理引擎的对外门面。
#
# 职责：
#   - 加载 tokenizer，处理请求的编码、校验与截断。
#   - 通过 EngineCoreClient（InprocClient 或 MPClient）驱动底层推理。
#   - 维护门面侧的请求镜像 _seqs，用于流式输出和最终文本解码。
#
# Engine 本身不持有 EngineCore，也不感知 TP / PP / DP 的具体展开方式。
# 所有并行策略的分支都下沉到 make_client 和 EngineCore 内部。
# ==============================================================================


# 控制面的门面类。它自己不碰张量运算，所有"算"的部分都甩给 ModelExecutor。

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import replace

import torch
from transformers import AutoTokenizer

from ..config import EngineConfig
from ..data import Sequence, SequenceStatus
from ..sampling_params import SamplingParams
from .client import InprocClient, make_client


# ==============================================================================
# Engine：推理引擎的对外门面。
#
# 职责：
#   - 加载 tokenizer，处理请求的编码、校验与截断。
#   - 通过 EngineCoreClient（InprocClient 或 MPClient）驱动底层推理。
#   - 维护门面侧的请求镜像 _seqs，用于流式输出和最终文本解码。
#
# Engine 本身不持有 EngineCore，也不感知 TP / PP / DP 的具体展开方式。
# 所有并行策略的分支都下沉到 make_client 和 EngineCore 内部。
# ==============================================================================


# 控制面的门面类。它自己不碰张量运算，所有"算"的部分都甩给 ModelExecutor。
class Engine:
    """
        LLM 推理引擎门面。

        对外提供四个核心操作：
            add_request：编码并校验请求，加入调度队列。
            step：执行一步调度 → 前向 → 采样。
            generate_batch：批量同步生成，阻塞直到所有请求完成。
            generate：单条流式生成，逐 token yield。

        底层执行路径由 make_client 决定：
            dp=1：InprocClient，同进程直接调用。
            dp>1：MPClient，经 ZMQ 路由到多个副本子进程。
    """

    def __init__(self, config: EngineConfig, executor=None) -> None:
        self.config = config
        self.device = torch.device("cuda:0")
        self.dtype = self._resolve_dtype()

        # 混合 GPU 主机：按本 Engine 实际用到的卡锁定 FlashInfer arch，
        # 避免默认扫到 Blackwell 后在 A100/3090 上 NoKernelImageForDevice。
        device_ids = config.device_ids or list(range(config.world_size))
        from ..utils.cuda_env import apply_flashinfer_arch_list

        apply_flashinfer_arch_list(device_ids, overwrite=True)

        # 禁用 BF16 matmul 的低精度 reduction，保证累加精度。
        # 禁用 BF16 matmul 的低精度 reduction，保证矩阵乘累加精度。
        # 这行是为了数值正确性，不是为了速度：BF16 矩阵乘默认允许"降低精度做归约"，
        # 能快一点，但累加误差会累积。关掉它换来的是可复现的数值。
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        self.tokenizer = self._load_tokenizer()
        self.eos_token_id = int(self.tokenizer.eos_token_id)
        self.context_len = config.context_len
        self.block_size = config.block_size

        # 构造底层客户端。
        # executor 参数仅在 torchrun 多机路径下使用：
        # 外部已初始化好 rank0 的 MultiprocExecutor，直接注入，避免重复创建。
        self._client = make_client(
            config,
            eos_token_id=self.eos_token_id,
            device=self.device,
            dtype=self.dtype,
            executor=executor,
        )

        # 门面侧的请求镜像。
        # dp=1 时，这里的 Sequence 与调度器内是同一对象（共享引用），
        #   调度器原地更新，门面侧无需额外同步。
        # dp>1 时，调度器在子进程里，门面侧需要根据 step 返回的增量输出
        #   手动推进这些镜像对象。
        self._seqs: dict[int, Sequence] = {}

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #

    # ---------- 初始化 ----------
    # 三个解析/加载函数，都在构造期被调用一次：dtype、tokenizer、KV 池 + 调度器。
    def _resolve_dtype(self) -> torch.dtype:
        """根据配置解析计算精度，默认回退到模型自带的 torch_dtype。"""
        if self.config.dtype != "auto":
            return getattr(torch, self.config.dtype)
        dt = getattr(self.config.hf_config, "torch_dtype", None)
        return dt if isinstance(dt, torch.dtype) else torch.float16

    # tokenizer 从同一个模型目录加载。
    def _load_tokenizer(self):
        """加载 tokenizer，若缺少 pad_token 则用 eos_token 补位。"""
        # trust_remote_code=True 会执行模型仓库里的自定义代码 ——
        # 方便（很多模型需要），但对来源不明的模型仓是个真实风险。
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model, trust_remote_code=True
        )
        # 必须补 pad_token：生成场景其实不用 padding，但 tokenizer 的很多内部
        # 路径会读它，缺了会在别的角落炸。直接用 eos 顶上是最常见的做法。
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @property
    def scheduler(self):
        """
            暴露底层调度器，仅用于调试和测试。
            只在 dp=1 时可用，dp>1 时调度器在子进程内，无法直接访问。
        """
        # dp>1 时，门面侧的 Sequence 是独立镜像，需要根据返回的增量手动推进。
        # dp=1 时，Sequence 与调度器共享同一对象，已由底层原地更新，这里跳过。
        if not isinstance(self._client, InprocClient):
            raise NotImplementedError(
                "scheduler is only reachable for the in-process (dp=1) engine"
            )
        return self._client.engine_core.scheduler

    @property
    def last_num_cached_tokens(self) -> int:
        """上一步命中前缀缓存的 token 数。dp>1 时暂不支持，返回 0。"""
        # dp>1 时，门面侧的 Sequence 是独立镜像，需要根据返回的增量手动推进。
        # dp=1 时，Sequence 与调度器共享同一对象，已由底层原地更新，这里跳过。
        if not isinstance(self._client, InprocClient):
            return 0
        return self._client.engine_core.last_num_cached_tokens

    # 换前缀缓存后端时**必须重建整条链**：新的索引结构意味着页池的语义变了，
    # 旧的调度器还攥着旧引用，继续用会读到已释放的对象。
    def reset_kv(self, prefix_backend: str | None = None) -> None:
        """
            清空 KV cache 并重建调度器。
            可选切换前缀缓存后端（"none" / "hash" / "radix"）。
            仅支持 dp=1。
        """
        if not isinstance(self._client, InprocClient):
            raise NotImplementedError("reset_kv is only supported for dp=1")
        if prefix_backend is not None and prefix_backend not in ("none", "hash", "radix"):
            raise ValueError(f"invalid prefix_backend: {prefix_backend!r}")
        core = self._client.engine_core
        core.kv = core.executor.reset_kv(prefix_backend)
        core.scheduler = core._build_scheduler()
        core.last_num_cached_tokens = 0
        # 把刚丢掉的旧池显存还给 CUDA 分配器 —— 不还的话，反复 reset 会让显存越吃越多。
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # 请求入队
    # ------------------------------------------------------------------ #

    # ---------- 请求入队 ----------
    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
    ) -> Sequence:
        """
            编码 prompt，校验长度，创建 Sequence 并提交到调度队列。

            若 prompt 超过 context_len，截断保留末尾（保留最近的上下文）。
            若 max_tokens + prompt 长度超过 context_len，压缩 max_tokens。

            Returns:
                已入队的 Sequence 对象。
                dp=1 时该对象与调度器内部共享同一引用，可直接读取生成进度。
        """
        # 没传就用默认采样参数（贪心、max_tokens 取默认值）。
        sp = sampling_params or SamplingParams()
        # 将输入统一转换为 token id 列表。
        # 文本和 token id 列表两种输入统一成 token id 列表 —— 之后所有逻辑只认 id。
        token_ids = (
            list(prompt) if isinstance(prompt, list) else self.tokenizer.encode(prompt)
        )
        # 空 prompt 直接拒绝：往下走会得到一条永远算不出东西的序列。
        if not token_ids:
            raise ValueError("empty prompt")

        # prompt 超长：截断，保留末尾部分。
        # prompt 超长时保留末尾 token，以尽量保留最近上下文。
        # 超长 prompt 的处理策略：**截掉开头、保留末尾**，而不是截掉末尾。
        # 这不是随手选的：生成任务里"最近的上下文"信息量最大，
        # 保留末尾比保留开头更有用。但这也意味着最前面的指令可能被丢掉了。
        if len(token_ids) > self.context_len:
            warnings.warn(
                f"prompt length {len(token_ids)} > context_len={self.context_len}; "
                f"truncating to last {self.context_len} tokens",
                stacklevel=2,
            )
            token_ids = token_ids[-self.context_len :]

        # 生成预算校验：确保 prompt + 生成不超过上下文窗口。
        # 限制 max_tokens，确保 prompt + completion 不超过 context_len。
        # 算出还剩多少位置可以生成，用来夹紧 max_tokens。
        max_ok = max(0, self.context_len - len(token_ids))
        if sp.max_tokens > max_ok:
            warnings.warn(
                f"max_tokens={sp.max_tokens} + prompt={len(token_ids)} exceeds "
                f"context_len={self.context_len}; clamping max_tokens to {max_ok}",
                stacklevel=2,
            )
            # 用 replace 而不是就地改：SamplingParams 是外部传进来的对象（还可能是共享的），
            # 就地改会污染调用方手里的那份。这里生成一个新的。
            sp = replace(sp, max_tokens=max_ok)

        # 到这里才真正造出序列对象：把 token、块大小、采样参数打包在一起，
        # 之后调度和 KV 管理都以 Sequence 为单位。
        seq = Sequence.from_prompt(token_ids, block_size=self.block_size, sampling=sp)
        self._seqs[seq.seq_id] = seq
        self._client.add_request(seq)
        return seq

    # ------------------------------------------------------------------ #
    # 执行
    # ------------------------------------------------------------------ #

    # ---------- 执行 ----------
    def step(self) -> tuple[list[tuple[int, list[int]]], int]:
        """
            驱动一步推理：调度 → 前向 → 采样 → 后处理。

            Returns:
                (finished, metric)
                - finished：本步完成的请求列表，每项为 (seq_id, completion_token_ids)。
                - metric：本步处理的 token 数（用于吞吐统计）。
        """
        outputs, metric = self._client.step()

        # dp>1 时，门面侧的 Sequence 是独立镜像，需要根据返回的增量手动推进。
        # dp=1 时，Sequence 与调度器共享同一对象，已由底层原地更新，这里跳过。
        if not isinstance(self._client, InprocClient):
            for out in outputs:
                seq = self._seqs.get(out.seq_id)
                if seq is None:
                    continue
                for tid in out.new_token_ids:
                    seq.append_token(tid)
                if out.is_finished:
                    seq.status = SequenceStatus.FINISHED

        # 收集本步完成的请求，提取生成部分的 token ids。
        finished: list[tuple[int, list[int]]] = []
        for out in outputs:
            if out.is_finished:
                seq = self._seqs.get(out.seq_id)
                if seq is not None:
                    finished.append(
                        (out.seq_id, seq.token_ids[seq.num_prompt_tokens :])
                    )
        return finished, metric

    # ---------- 两个便捷入口 ----------
    # 都是围绕 step() 的循环包装：一个攒完再返回（便于压测），一个边生成边吐（便于交互）。
    def generate_batch(
        self,
        prompts: list[str | list[int]],
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
    ) -> list[str]:
        """
        批量同步生成。

        将所有 prompt 一次性入队，循环 step 直到全部完成，
        返回与输入顺序对应的生成文本列表。
        """
        # 归一化采样参数，并将所有请求加入调度器。
        sps = self._normalize_sampling_params(prompts, sampling_params)
        seqs = [self.add_request(p, sp) for p, sp in zip(prompts, sps)]
        finished_text: dict[int, str] = {}
        while self.has_unfinished():
            outputs, _ = self.step()
            # 完成的请求才进结果表；没完成的继续留在调度器里。
            for seq_id, completion_ids in outputs:
                finished_text[seq_id] = self.tokenizer.decode(completion_ids)
        return [finished_text[seq.seq_id] for seq in seqs]

    # 单条流式生成（生成器）。
    def generate(
        self,
        prompt: str | list[int],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
    ) -> Iterator[str]:
        """
            单条流式生成。

            每步检查是否有新 token 产出，有则立即 yield 解码后的文本片段。
            调用方可用 for 循环逐段消费。
        """
        # 构造采样参数并提交请求。
        sp = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_new_tokens,
        )
        seq = self.add_request(prompt, sp)
        emitted = 0
        # 每次 step 后增量解码已经新生成的 token。
        # 注意这里等的是**这一条**序列完成，不是全调度器空 ——
        # 所以它可以在一个共享引擎上和其他请求共存。
        while not seq.is_finished:
            self.step()
            # 将本步新产出的 token 逐个解码并 yield。
            # 内层循环负责"把这一步新产生的 token 全部吐出去"：
            #   emitted 是已发出的游标，每步可能新增 0 个（还在 prefill）或多个（prefill 一次吐一个）。
            # 游标必须按 num_completion_tokens 判断，不能用 len(token_ids) ——
            # 后者包含 prompt，会导致重复吐 prompt 的 token。
            while emitted < seq.num_completion_tokens:
                tid = seq.token_ids[seq.num_prompt_tokens + emitted]
                emitted += 1
                # 注意是**逐 token 解码**：多字节字符可能被切开，边界处会短暂出现乱码，
                # 这是流式解码的固有现象（真实服务通常靠"保留尾部若干字节"来缓解）。
                yield self.tokenizer.decode([tid])

    def has_unfinished(self) -> bool:
        """是否还有未完成的请求在调度队列中。"""
        return self._client.has_unfinished()

    @staticmethod
    def _normalize_sampling_params(
        prompts: list[str | list[int]],
        sampling_params: SamplingParams | list[SamplingParams] | None,
    ) -> list[SamplingParams]:
        """
            将采样参数统一为与 prompts 等长的列表。
            支持三种输入：None（全部默认）、单个（广播）、列表（逐一对应）。
        """
        if sampling_params is None:
            return [SamplingParams() for _ in prompts]
        if isinstance(sampling_params, SamplingParams):
            return [sampling_params for _ in prompts]
        if len(sampling_params) != len(prompts):
            raise ValueError("sampling_params length must match prompts")
        return list(sampling_params)

    # ---------- 资源释放 ----------
    def destroy(self) -> None:
        """释放引擎资源：关闭底层 client，清空请求镜像。"""
        self._client.destroy()
        self._seqs.clear()