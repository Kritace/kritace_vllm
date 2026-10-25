# ============================================================================
# 【本文件是做什么的】引擎配置：一个 dataclass 装下所有可调项，外加
#     __post_init__ 里的**归一化**（补默认值、把 HF 配置里的字段搬进来）
#     与 _validate_parallel 的**并行校验**（组合不合法时启动即报错）
#
# 【为什么校验要放在构造时】并行相关的错误（tp 不整除头数、EP 不整除专家数……）
#     如果等到跑起来才炸，现象往往是"某张卡数值不对"或"通信卡死"，很难回溯到配置。
#     放在这里，启动阶段一句人话就能定位。
#
# 【两种并行对切分的要求不同（这是校验里最容易记混的一处）】
#     不开 EP：MoE 按中间维切 ⇒ 要求 `moe_intermediate_size % tp_size == 0`；
#     开了 EP：MoE 按**专家**切 ⇒ 要求 `num_experts % tp_size == 0`。
#
# 【读配置的顺序】config.yaml 给默认值 → 命令行覆盖 → post_init 归一化 → 校验。
#     所以"配置文件里写的"不一定是最终生效值（例如 dtype=auto 会被解析成 bfloat16）。
# ============================================================================

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from transformers import AutoConfig, PretrainedConfig


# 【这个模型是不是 MoE】不能只看一个字段：不同代 HF 配置的表达方式不一样 ——
#     Qwen1.5-MoE 在 architectures 里带 "Moe"；
#     有些配置只给出 num_experts（架构名不带 Moe）；
#     还有的把 MoE 藏在 text_config 里面（多模态模型）。
# 所以这里是"三个信号或起来"，宁可判得宽一点 —— 判错会走错模型结构，那是致命的。
def _is_moe_config(cfg: PretrainedConfig) -> bool:
    # 信号①：architectures 列表（例如 [\「Qwen3MoeForCausalLM\」]）。
    arch = getattr(cfg, "architectures", None) or []
    name = type(cfg).__name__
    # 信号②：类名里带 Moe —— 有些本地配置的 architectures 为空，只有类名能看。
    if any("Moe" in a or "MoE" in a for a in arch) or "Moe" in name or "MoE" in name:
        return True
    # 信号③：配置里直接给了 num_experts（> 0 就按 MoE 处理）。
    return int(getattr(cfg, "num_experts", 0) or 0) > 0


# 引擎的全部配置。做成 dataclass，默认值就是"开箱能用"的一套。
@dataclass
class EngineConfig:
    """引擎配置：调度 / 内存 / 算子 / 并行。

    并行语义（三者正交，分叉点不同）::

        dp_size  → 独立 Engine 副本数（``MPClient`` / ZMQ）；不进入 NCCL world
        tp_size  → 同 Engine 内张量并行度（NCCL all-reduce / gather）
        pp_size  → 同 Engine 内流水线 stage 数（NCCL P2P hidden+residual）

    ``enable_ep`` 不增加 world：Attention 仍走 TP，MoE 在同一批 TP rank 上按 expert 切分。
    单 Engine 的 NCCL ``world_size = tp_size * pp_size``。
    """

    model: str

    #  调度器
    context_len: int = 4096
    # 单条请求的最大长度（prompt + 生成）。它直接决定 KV 池要开多大。
    max_num_seqs: int = 256
    # 单步最多几条序列并发。
    max_num_batched_tokens: int = 8192
    mix_prefill_decode: bool = True

    kvcache_block_size: int = 256
    # 必须是 256 的倍数 —— 这是 FlashAttention-2 paged 的实现约束，
    # 不满足时中间会有 assert 报错。所以 __post_init__ 里提前拦一道。
    gpu_memory_utilization: float = 0.9
    num_kvcache_blocks: int = 0
    prefix_backend: str = "hash"

    enforce_eager: bool = True
    torch_compile: bool = False
    compile_mode: str = "reduce-overhead"
    compile_dynamic: bool = True
    # True = 符号形状：一次编译覆盖任意长度，代价是数值上略有保守。
    # False = 固定形状：每次遇到新长度都要重编译（~2-12 秒），只适合离线固定长度。
    dtype: str = "auto"

    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    # NCCL rendezvous 端口。多副本时 MPClient 会按副本错开（+r×16），避免撞车。
    nccl_port: int = 29500
    # 单机就是 127.0.0.1；跨机时才需要填别的地址。
    master_addr: str = "127.0.0.1"
    # master_port 是 TCPStore 的端口，与 nccl_port 是两个独立的口子。
    master_port: int = 29500
    device_ids: list[int] | None = None
    # NCCL 通信网卡名：留空 = 单机 loopback；跨机必须填（否则 NCCL 可能挑错网卡）。
    nccl_ifname: str = ""
    pp_layer_counts: list[int] | None = None

    # 【EP 开关】注意它**不增加 world**：Attention 仍走原来的 TP，MoE 在同一批 rank 上按专家切。
    # 所以开了 EP 之后 `world_size` 不变（见文件末尾的 world_size）。
    enable_ep: bool = False
    # fused MoE 的后端。'auto' = 运行时探测（cutlass 优先、退 Triton，见 moe/parallel.py）；
    # 显式写 'cutlass' / 'triton' 则在不可用时**启动即报错**，不会静默降级。
    moe_backend: Literal["auto", "cutlass", "triton"] = "auto"

    # --- 投机解码（v6 新增）---
    # spec_draft 为空 = 关闭；"ngram" = 用序列自身的 n-gram 匹配（零额外模型）；
    # 其它值视为 draft 模型路径，必须与 target 同词表。
    # 三个字段的关系：`spec_draft` 选草稿来源、`num_speculative_tokens` 是每步草稿长度 K、
    # `spec_ngram_max_n` 只对 n-gram 草稿有意义。三者一起决定"这一步要不要投机、用谁草稿"。
    spec_draft: str = ""
    # 每步提交给 target 验证的草稿长度 K。
    # 【为什么默认 0（关闭）】投机不是无条件的优化，它的收益是**期望产出**：
    #     1 + α + α² + … + α^K   （α = 接受率；全接受时还会额外白送一个 token）
    # α 低时每步只多产出很少，却要**多算 K 个位置**（K+1 个 query 的前向 + K+1 个位置的 KV），
    # 净效果可能是变慢。所以默认关，由使用者按输入类型自己开
    # （n-gram 草稿只在含重复片段的输入上有 α）。
    num_speculative_tokens: int = 0
    # ngram 草稿允许的最长匹配后缀。
    spec_ngram_max_n: int = 4

    #  运行时
    # 由 __post_init__ 填进来，用户不需要管。
    hf_config: PretrainedConfig = field(init=False)
    # 【两个"算出来"的字段】init=False 表示它们不在构造参数里，由 __post_init__ 填。
    #     hf_config —— 真正读到的 HF 配置（层数/头数/专家数都从它来）
    #     is_moe    —— 上一步的判断结果，后面校验与建模型都要用
    is_moe: bool = field(init=False, default=False)

    # 构造完立刻做校验与归一化 —— 让"配置不合法"在启动时暴露，而不是跑到一半才炸。
    def __post_init__(self) -> None:
        # 展开 ~ 之类的路径简写。
        self.model = os.path.expanduser(self.model)
        # 早失败原则：路径不存在就现在报，别等到加载权重时才报一个模糊的错。
        if not os.path.isdir(self.model):
            raise FileNotFoundError(
                f"Model path not found: {self.model}. "
                "Use a local directory (e.g. ~/huggingface/Qwen3-0.6B)."
            )
        # 与 FlashAttention-2 的页大小约束对齐（见上面字段注释）。
        if self.kvcache_block_size % 256 != 0:
            raise ValueError("kvcache_block_size must be a multiple of 256 (FA2 paged)")
        # 三个后端名的合法性检查 —— 写错了应该立刻报，而不是在 KVManager 里静默回退。
        if self.prefix_backend not in ("none", "hash", "radix"):
            raise ValueError(
                f"prefix_backend must be none|hash|radix, got {self.prefix_backend!r}"
            )
        if self.tp_size < 1 or self.pp_size < 1 or self.dp_size < 1:
            raise ValueError("tp_size / pp_size / dp_size must be >= 1")
        # 后端名写错必须启动时报错：留到 KVManager 里静默回退，你就不知道自己在跑哪个后端了。
        if self.moe_backend not in ("auto", "cutlass", "triton"):
            raise ValueError(
                f"moe_backend must be auto|cutlass|triton, got {self.moe_backend!r}"
            )
        # 校验：K 不能为负（0 = 关闭是合法值）。
        if self.num_speculative_tokens < 0:
            raise ValueError(
                f"num_speculative_tokens must be >= 0, got "
                f"{self.num_speculative_tokens}"
            )
        # 【配伍校验】开了投机（K>0）就必须指明草稿来源，否则每步都会"想投机但没有草稿"，
        # 白跑一遍分支。这里提前拦住，避免配置错误在运行时表现为"投机没生效"这种静默现象。
        if self.num_speculative_tokens > 0 and not self.spec_draft:
            raise ValueError(
                "spec_draft must be set when num_speculative_tokens > 0 "
                "(use 'ngram' for the model-free drafter)"
            )
        if self.spec_ngram_max_n < 1:
            raise ValueError(
                f"spec_ngram_max_n must be >= 1, got {self.spec_ngram_max_n}"
            )
        # 真正读模型配置（层数、头数、头维都从这里来）。
        # trust_remote_code=True 会执行模型仓库里的自定义代码。
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        # 读 HF 配置之后立刻判定"是不是 MoE"。
        # 这个值决定了后面走哪条校验分支、以及模型侧建稠密层还是 MoE 层。
        self.is_moe = _is_moe_config(self.hf_config)
        text = getattr(self.hf_config, "text_config", self.hf_config)
        # 用模型自身的最大位置编码**夹紧**用户填的 context_len：
        #   开得比模型支持的还大，RoPE 表会越界查，属于静默错误（位置编码全乱）。
        self.context_len = min(
            self.context_len,
            getattr(text, "max_position_embeddings", self.context_len),
        )
        self._validate_parallel(text)
        # MoE 可以进 Decode CUDA Graph：token 数按 captured BS 垫齐后，
        # align / silu / sum / GEMM workspace 都按容量复用，不再逐步 cudaMalloc。
        # 若环境仍要强制 eager：KRITACE_MOE_FORCE_EAGER=1。
        # 【MoE + CUDA Graph 的逃生开关】MoE 的 fused kernel 会把 workspace 按容量池化，
        # 因此**可以**进 CG（见 config 上方的说明）；但某些环境/驱动组合下仍会出问题，
        # 于是留一个环境变量 KRITACE_MOE_FORCE_EAGER=1 强制退回 eager，便于定位问题。
        if self.is_moe and os.environ.get("KRITACE_MOE_FORCE_EAGER", "0") in (
            "1",
            "true",
            "True",
        ):
            # 逃生开关生效：把用户显式传的 enforce_eager=False 覆盖掉。
            self.enforce_eager = True

    def _validate_parallel(self, text: PretrainedConfig) -> None:
        n_heads = int(text.num_attention_heads)
        n_kv = int(text.num_key_value_heads)
        inter = int(text.intermediate_size)
        if n_heads % self.tp_size != 0:
            raise ValueError(
                f"num_attention_heads ({n_heads}) must be divisible by tp_size ({self.tp_size})"
            )
        if n_kv >= self.tp_size:
            if n_kv % self.tp_size != 0:
                raise ValueError(
                    f"num_key_value_heads ({n_kv}) must be divisible by tp_size ({self.tp_size})"
                )
        elif self.tp_size % n_kv != 0:
            raise ValueError(
                f"tp_size ({self.tp_size}) must be divisible by num_key_value_heads ({n_kv})"
            )
        if inter % self.tp_size != 0:
            raise ValueError(
                f"intermediate_size ({inter}) must be divisible by tp_size ({self.tp_size})"
            )
        if self.pp_size > int(text.num_hidden_layers):
            raise ValueError(
                f"pp_size ({self.pp_size}) > num_hidden_layers ({text.num_hidden_layers})"
            )
        # 【非 MoE 模型的分支】到这里并行校验就结束了 ——
        # 但要多拦一笔：对稠密模型开 EP 是没有意义的（MoE 层的切法根本不存在），
        # 静默忽略会让人以为 EP 生效了。
        if not self.is_moe:
            # 【EP / TP 互斥的校验分支 —— 本文件最容易记混的一处】
            # 开 EP：专家是切分单位 ⇒ 要求 num_experts % tp_size == 0；
            # 不开 EP：单个专家的中间维是切分单位 ⇒ 要求 moe_intermediate_size % tp_size == 0。
            # 两条要求的对象完全不同，写反了会在运行时才炸（而且现象是数值错、不是报错）。
            if self.enable_ep:
                raise ValueError("enable_ep requires a MoE model")
            return
        # 以下都只在 MoE 模型上做。先取 MoE 专属字段，缺了就报错（别用默认值硬跑）。
        moe_inter = getattr(text, "moe_intermediate_size", None)
        if moe_inter is None:
            raise ValueError("MoE config missing moe_intermediate_size")
        moe_inter = int(moe_inter)
        # 路由专家的中间维与专家个数 —— 后面两个分支分别对应「切中间维」与「切专家」。
        n_experts = int(getattr(text, "num_experts", 0) or 0)
        if n_experts <= 0:
            raise ValueError("MoE config has num_experts <= 0")
        # 共享专家的中间维：Qwen3-MoE 没有这个字段 / 值为 0（它取消了共享专家），
        # 所以这里必须容忍 0，不能用"必须整除"的通用规则一刀切。
        shared_inter = int(getattr(text, "shared_expert_intermediate_size", 0) or 0)
        if self.enable_ep:
            # 128 个专家、tp=4 ⇒ 每卡 32 个；不整除就没法均分，直接拦。
            if n_experts % self.tp_size != 0:
                raise ValueError(
                    f"enable_ep requires num_experts ({n_experts}) % tp_size "
                    f"({self.tp_size}) == 0"
                )
        else:
            # 不开 EP：MoE 与稠密层一样按中间维切，所以这里卡的是中间维。
            if moe_inter % self.tp_size != 0:
                raise ValueError(
                    f"moe_intermediate_size ({moe_inter}) must be divisible by "
                    f"tp_size ({self.tp_size})"
                )
            # 共享专家也要跟着切（它没走 EP，仍是普通 TP）。shared_inter 为 0 时跳过 ——
            # 那代表模型没有共享专家。
            if shared_inter and shared_inter % self.tp_size != 0:
                raise ValueError(
                    f"shared_expert_intermediate_size ({shared_inter}) must be "
                    f"divisible by tp_size ({self.tp_size})"
                )

    # 别名：代码里到处用的是 block_size，配置字段名多了个 kvcache_ 前缀。
    # 保留 property 而不是改名，是为了配置文件的可读性（明确它是 KV 的块大小）。
    @property
    def spec_enabled(self) -> bool:
        """是否启用投机解码。"""
        return self.num_speculative_tokens > 0 and bool(self.spec_draft)

    @property
    # 单 Engine 的 NCCL world：**不含 dp**。
    # DP 的多个副本是互相独立的引擎，各自有自己的 world（见 control/mp_client.py）。
    def world_size(self) -> int:
        """单 Engine 的 NCCL world（不含 DP 副本）。"""
        # 注意**不含 dp** —— DP 是独立的 Engine，不在同一个 NCCL world 里。
        # tp × pp 才是参加集合通信的进程数；把 dp 乘进来会让 NCCL 初始化等一个永远不来的 rank。
        return self.tp_size * self.pp_size

    @property
    def num_gpus(self) -> int:
        """门面视角下的总 GPU 数：``dp_size * tp_size * pp_size``。"""
        # 门面视角的总卡数（每个 DP 副本各自再展开 TP × PP）。
        return self.dp_size * self.tp_size * self.pp_size

    @property
    def block_size(self) -> int:
        # 代码里统一用 block_size 这个名字，配置字段名多带了 kvcache_ 前缀。
        return self.kvcache_block_size
