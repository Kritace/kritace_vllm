# ============================================================================
# 【本文件是做什么的】**一个 Worker = 一个进程 = 一张卡**。它是"这张卡上的一切"的
# 持有者：通信组、模型、KV 池、采样器、以及串起它们的 ModelRunner。
#
# 【它在整个启动链里的位置】
#     Engine（driver 进程）
#       → MultiprocExecutor 按 TP×PP 拉起 N 个 worker 进程（torchrun 或 spawn）
#         → 每个进程跑 Worker：init_environment → initialize → 等指令循环
#   所以这个文件是"计算侧进程"的入口。上层看到的是"一个引擎"，
#   实际下面有 world_size 个 Worker 各自持有一部分模型。
#
# 【为什么构造与初始化分两步】
#   构造（__init__）只记身份与配置；真正的装配（initialize）在启动通信之后。
#   序不能反：**先建进程组，再建模型** —— 因为权重切片要靠（TP/PP 的）组信息。
#
# 【⚠️ 三个容易踩的点】
#     ① `_probe_cg_footprint` 会**真的捕获一次 CUDA Graph**，所以它不便宜
#        （几百毫秒到数秒），而且必须在"还没有真 KV 池"的时候做 ——
#        否则量出来的差值里混着页池。
#     ② 探针用完必须把显存完整还回去（destroy → 删引用 → gc → empty_cache），
#        少任何一步都会让后面真正建池时凭空少掉一块预算。
#     ③ `execute_model` 与 `read_samples` 是**故意分开**的：
#        前者只发射（异步），后者才同步取数 —— 分开才能让 H2D/D2H 与计算重叠。
#
# 【读这个文件的顺序】
#     __init__ / init_environment（我是谁、跟谁通信）
#       → initialize / _load_model / _make_kv（装配顺序：模型 → 基线 → KV 池）
#       → _probe_cg_footprint（CG 的显存代价从哪来）
#       → execute_model / read_samples（运行期的两个半程）
#       → destroy_environment（逆序拆除）
# ============================================================================

from __future__ import annotations

import copy
import gc
import io
from contextlib import redirect_stdout

import torch

from ..config import EngineConfig
from ..data import SchedulerOutput
from ..distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_pp_group,
    get_world_group,
    init_distributed_environment,
    initialize_model_parallel,
    set_expert_parallel,
    set_nccl_env,
    set_pp_layer_ranges,
)
from .kv_pool import KVPool
from .model_runner import ModelRunner
from .models.loader import default_dtype_context, load_model, skip_param_init
from .models.qwen3 import Qwen3ForCausalLM
from .sampler import Sampler


class Worker:
    # 一个 Worker 就是一个进程：身份 + 配置 + 运行时组件。
    def __init__(
        self,
        config: EngineConfig,
        rank: int,
        tp_rank: int,
        pp_rank: int,
        local_rank: int | None = None,
    ):
        # 记下自己的身份：全局 rank，以及 TP / PP 两个维度的组内序号。
        self.config = config
        self.rank = rank
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.tp_size = config.tp_size
        self.pp_size = config.pp_size
        self.world_size = config.world_size
        # local_rank 是机内 GPU 下标：单机 spawn 时等于 rank；多机 torchrun 时为 LOCAL_RANK。
        self.local_rank = rank if local_rank is None else local_rank
        # 真正用于 cuda:x 的卡号由 _device_id() 决定（device_ids 可以显式映射）。
        self.device = torch.device(f"cuda:{self._device_id()}")
        # dtype 未显式指定时，从 HF config 的 torch_dtype 推。
        self.dtype = self._resolve_dtype()
        # 下面这些运行时组件都在 initialize() 里装配，构造阶段一律先留空。
        self.sampler: Sampler | None = None
        self.kv: KVPool | None = None
        self.model: torch.nn.Module | None = None
        self.is_last_pp = False
        self.model_runner: ModelRunner | None = None
        # 加载完模型后的显存基线 —— 测 CG 足迹时要拿它当零点。
        self._mem_baseline = 0

    # 建通信环境。**必须在建模型之前调**，因为权重切片要靠组信息。
    def init_environment(self, init_method: str | None = None) -> None:
        from ..utils.cuda_env import apply_flashinfer_arch_list

        # 【必须在 import flashinfer 之前调】把本卡的架构写进 FLASHINFER_CUDA_ARCH_LIST，
        # 否则 FlashInfer 会用默认探测结果编译，异构机器上可能漏编某一档（见 utils/cuda_env.py）。
        # overwrite=True：以代码里的推导为准，不看外部有没有设过。
        apply_flashinfer_arch_list([self._device_id()], overwrite=True)
        # 单机 spawn 默认走 lo；跨机在 config.nccl_ifname 填 ib0/eth0。
        ifname = self.config.nccl_ifname or "lo"
        # 不填网卡时禁掉 IB：本地根本没有 InfiniBand，开着只会让 NCCL 卡在设备探测。
        set_nccl_env(ifname, disable_ib=not bool(self.config.nccl_ifname))
        # 单机（loopback）用文件 rendezvous，不依赖 rank0 先起 TCPStore 的时序。
        if init_method is None and self.config.master_addr in ("", "127.0.0.1", "localhost"):
            # 单机用 file:// rendezvous：各 rank 通过同一个临时文件握手，不需要谁先起 TCPStore。
            init_method = (
                f"file:///tmp/kritace_vllm_rdv_{self.config.nccl_port}"
                f"_{self.config.tp_size}_{self.config.pp_size}"
            )
        else:
            init_method = init_method or (
                f"tcp://{self.config.master_addr}:{self.config.nccl_port}"
            )
        # 初始化 world：这一步在 NCCL 层面真正把进程连起来。
        init_distributed_environment(
            world_size=self.world_size,
            rank=self.rank,
            local_rank=self._device_id(),
            backend="nccl",
            init_method=init_method,
        )
        # 再切 TP / PP 两个子组 —— 用的是纯算术规则，不依赖任何外部布局描述。
        initialize_model_parallel(
            self.tp_size,
            self.pp_size,
            self.tp_rank,
            self.pp_rank,
        )
        # 【EP 开关的下发点】两个条件缺一不可：模型确实是 MoE（稠密模型没有专家可切）、
        # 用户显式开了 enable_ep。注意这里不回传任何东西 —— EP 只影响 MoE 层的切法，
        # 组还是那个 TP 组（见 parallel_state.set_expert_parallel）。
        set_expert_parallel(self.config.enable_ep and self.config.is_moe)
        # 只有最后一个 PP stage 才持有 lm_head 并产出 token。
        self.is_last_pp = get_pp_group().is_last_rank
        # 不均分 PP：按 config.pp_layer_counts 覆盖等分层区间（如 12/28 适配 12G+24G 异卡）。
        if self.config.pp_layer_counts is not None:
            counts = self.config.pp_layer_counts
            # 两道断言：长度与总层数都要对得上 —— 写错了会静默少算/多算层。
            assert len(counts) == self.pp_size, "pp_layer_counts 长度须等于 pp_size"
            assert sum(counts) == self.config.hf_config.num_hidden_layers, (
                "pp_layer_counts 之和须等于模型层数"
            )
            set_pp_layer_ranges(counts)

    # 决定这一进程用哪张**物理**卡。
    def _device_id(self) -> int:
        # device_ids 显式映射（如单卡调试 [0,0]）优先，否则用机内 local_rank。
        if self.config.device_ids is not None:
            return self.config.device_ids[self.rank]
        return self.local_rank

    # 装配顺序有讲究：加载模型 → 同步 → 记显存基线 → 再建 KV 池（池的大小依赖基线）。
    def initialize(self) -> None:
        model = self._load_model()
        # 权重加载是异步的，先排空再量显存，否则基线会偏小。
        torch.cuda.synchronize()
        # 把加载过程的临时块还给驱动，让基线尽量干净。
        torch.cuda.empty_cache()
        self.model = model
        # 采样器无状态，构造即用。
        self.sampler = Sampler()
        # 记下「模型已就位、KV 池还没有」这一刻的显存水位。
        self._mem_baseline = torch.cuda.memory_reserved()
        # KV 池的块数在这里定：要么用配置给的，要么按剩余显存自动算。
        self.kv = self._make_kv()
        # ModelRunner 负责组包 + 前向 + 采样，是 Worker 与计算逻辑之间的那一层。
        self.model_runner = ModelRunner(
            model, self.kv, self.sampler, config=self.config, device=self.device
        )

    # dtype 解析：配置显式给了就用它，否则跟随 HF config。
    def _resolve_dtype(self) -> torch.dtype:
        if self.config.dtype != "auto":
            return getattr(torch, self.config.dtype)
        # HF config 里的 torch_dtype 可能只是个字符串（没被加载成 dtype），所以下面还要再判一次类型。
        dt = getattr(self.config.hf_config, "torch_dtype", None)
        # 兜底 fp16 —— 拿不到就按最常见的半精度走，而不是让 None 一路传下去。
        return dt if isinstance(dt, torch.dtype) else torch.float16

    # 建模型 + 灌权重，全程要压住默认 dtype 与参数初始化。
    def _load_model(self) -> torch.nn.Module:
        # 【MoE 模型的分叉点】MoE 与稠密走两个不同的类，而不是一个类里塞分支：
        # 两者只在 MLP 与权重加载上有差异，用继承表达最清楚（见 models/qwen3_moe.py 文件头）。
        from .models.qwen3_moe import Qwen3MoeForCausalLM
        from ..moe.parallel import MoEParallelConfig, resolve_moe_backend

        # 让建张量阶段就以目标 dtype 进行，避免「先 fp32 再转换」的峰值显存。
        with default_dtype_context(self.dtype):
            # 跳过随机初始化：权重马上会被文件覆盖，随机初始化在这类模型上是几十秒的纯浪费。
            with skip_param_init():
                # MoE 需要两样额外信息才能建模型：
                #     ① fused 后端（cutlass / triton）—— 运行时探测，写死了换机器就得改代码；
                #     ② MoE 的并行布局（tp=1 + ep=旧 tp，或保持普通 TP）—— 由 config 推导。
                if self.config.is_moe:
                    # 后端探测：显式指定时不可用会直接抛错；auto 时 cutlass 优先、退 Triton。
                    backend = resolve_moe_backend(self.config.moe_backend)
                    parallel = MoEParallelConfig.from_engine(
                        tp_size=self.tp_size,
                        tp_rank=self.tp_rank,
                        enable_ep=self.config.enable_ep,
                        fused_backend=backend,
                    )
                    # 把 parallel 传进去：每一层 MoE 都要按它决定"专家权重怎么装、
                    # forward 里要不要跨卡归约"。稠密模型不需要这个参数 —— 这也是两者的分界。
                    model = Qwen3MoeForCausalLM(self.config.hf_config, parallel=parallel)
                else:
                    # 先按结构建空壳。
                    model = Qwen3ForCausalLM(self.config.hf_config)
        # 再把权重灌进去 —— TP/PP 的切片发生在各自的 weight_loader 里。
        load_model(model, self.config.model)
        # 最后统一搬 dtype 与设备，并切到 eval（关掉训练期行为）。
        return model.to(dtype=self.dtype, device=self.device).eval()

    # 探针：量出 CUDA Graph 要额外占多少显存。这笔钱要从 KV 池的预算里**先扣掉**。
    def _probe_cg_footprint(self) -> int:
        # 复制一份配置来探测，避免污染真实配置。
        probe_config = copy.copy(self.config)
        # 只测 CUDA Graph，把 compile 关掉 —— 否则测出来的数里混着编译器的分配。
        probe_config.torch_compile = False
        # 用 8 个块的迷你池把 ModelRunner 实例化起来（探针只需要它跑一次捕获）。
        kv = KVPool(
            self.config.hf_config,
            num_blocks=8,
            block_size=self.config.block_size,
            device=self.device,
            dtype=self.dtype,
        )
        # 探测过程会打一堆日志，屏蔽掉，别污染启动输出。
        with redirect_stdout(io.StringIO()):
            runner = ModelRunner(
                self.model, kv, self.sampler, config=probe_config, device=self.device
            )
        # 同步后再读显存，否则拿到的是还没落地的估计值。
        torch.cuda.synchronize()
        # 差值就是「模型已就位 → CG 捕获完成」多出来的那一部分。
        footprint = torch.cuda.memory_reserved() - self._mem_baseline
        # 必须把探针自己占的显存完整还回去：destroy → 删引用 → gc → empty_cache。
        runner.destroy()
        del runner, kv
        # 这里少一步，后面真正建池时就会凭空少掉一块预算。
        gc.collect()
        # 清掉 CUDA 缓存块（上一步只是把张量对象释放了）。
        torch.cuda.empty_cache()
        # 返回值就是 CG 的额外开销。
        return int(footprint)

    # 建 KV 池。两条路：手动指定块数（可复现），或按剩余显存自动算。
    def _make_kv(self) -> KVPool:
        use_cg = not self.config.enforce_eager
        # 手填了块数 → 完全信任它，不探测也不用探针：实验可复现性优先。
        if self.config.num_kvcache_blocks > 0:
            num_blocks = self.config.num_kvcache_blocks
        else:
            # 自动模式：先量出 CG 要吃多少。
            reserve = self._probe_cg_footprint() if use_cg else 0
            # 再让池自己算能开多少块（它会取各 rank 的最小值，保证大家块号一致）。
            num_blocks = KVPool.auto_num_blocks(
                self.config.hf_config,
                block_size=self.config.block_size,
                dtype=self.dtype,
                device=self.device,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                reserve_bytes=reserve,
            )
            # 把算出来的块数写回配置：后续重建池（reset_kv）时要保持一致。
            self.config.num_kvcache_blocks = num_blocks
            # 把实际测到的 CG 足迹打进日志 —— 这是解释「为什么并发开不到预期」的关键数字。
            if use_cg:
                print(
                    f"CG footprint measured: {reserve / 2**30:.3f} GiB; "
                    f"KV pool sized to remainder ({num_blocks} blocks)"
                )
        # 按最终块数开池。
        return KVPool(
            self.config.hf_config,
            num_blocks=num_blocks,
            block_size=self.config.block_size,
            device=self.device,
            dtype=self.dtype,
        )

    # 给调度器用：它需要知道池子总共有多少块。
    def get_kv_cache_size(self) -> int:
        # 还没初始化就来问 = 调用顺序错了，直接断言。
        assert self.kv is not None, "initialize() must be called first"
        return self.kv.num_blocks_total

    # 重建 KV 池与 ModelRunner：换前缀缓存后端、或重开引擎时用。
    def reset_kv(self, prefix_backend: str | None = None) -> None:
        """重建 GPU KV 池与 ModelRunner。多卡 driver 广播后每 rank 都要走这里。"""
        # 先拆掉旧的 runner —— 它持有指向旧池的视图，不能复用。
        if self.model_runner is not None:
            self.model_runner.destroy()
            self.model_runner = None
        if prefix_backend is not None:
            # 前缀缓存后端可以在这里切换；多卡场景由 driver 广播，每个 rank 都要走到。
            self.config.prefix_backend = prefix_backend
        assert self.model is not None and self.sampler is not None
        # 重开池，再按新池重建 runner。
        self.kv = self._make_kv()
        self.model_runner = ModelRunner(
            self.model, self.kv, self.sampler, config=self.config, device=self.device
        )

    # 只发射前向 + 异步 D2H，**不在这里等结果**（这样才和计算重叠）。
    def execute_model(self, sched: SchedulerOutput) -> tuple[int, int, int] | None:
        """只发射前向 + 异步 D2H，不等待采样结果。"""
        assert self.model_runner is not None
        result = self.model_runner.run_forward(sched)
        # ZOS + PP：在途行的输入 token 要从「上一批的采样结果」里 gather，而这份结果
        # 只在采样 rank 的显存里。非末段 rank 不跑 _stage_samples（run_forward 提前
        # return 了），它的 _sample_dev 永远是初值 —— 不广播就会 gather 到垃圾。
        # 走 NCCL device group，不碰 host、不引入同步点；pp_size<=1 时是空操作。
        self._broadcast_prev_samples()
        # 非最后一段 PP 没有输出，直接返回 None。
        if not self.is_last_pp:
            return None
        return result

    def _broadcast_prev_samples(self) -> None:
        """把上一批采样结果同步到所有 rank（只有 PP>1 才需要）。"""
        if self.pp_size <= 1:
            return
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        group = get_world_group()
        if group.size <= 1:
            return
        runner = self.model_runner
        assert runner is not None
        # 采样 rank 的全局编号，口径与 control/executor.py 的 _sampler_rank 一致：
        # rank 按 pp 优先编号，所以最后一个 pp stage 的第一个 tp rank 就是它。
        src = (self.pp_size - 1) * self.tp_size
        torch.distributed.broadcast(runner.sample_dev, src=src, group=group.device_group)

    # 真正去读采样结果（会同步）。与执行分开是为了让取数和下一步计算重叠。
    def read_samples(
        self, n_p: int, n_d: int, slot: int
    ) -> tuple[list[int], list[int]]:
        assert self.model_runner is not None
        return self.model_runner.read_samples(n_p, n_d, slot)

    # 逆序拆除：同步 → 释放 runner / KV / 模型 → 复位 PP 分层表 → 销毁进程组。
    def destroy_environment(self) -> None:
        # CUDA 上下文可能已经坏了（异常路径进来），同步失败也不该让清理中断。
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                pass
        if self.model_runner is not None:
            self.model_runner.destroy()
        self.model_runner = None
        self.kv = None
        self.model = None
        # 复位全局分层表，否则同一进程里再建引擎会沿用上一次的切法。
        set_pp_layer_ranges(None)
        destroy_model_parallel()
        destroy_distributed_environment()
