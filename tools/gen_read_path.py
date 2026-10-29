# -*- coding: utf-8 -*-
"""生成 docs/09-全量阅读路径.md：71 个产品模块的线性阅读顺序。

顺序由人定（按「读它之前必须先懂什么」），但**合法性由机器检查**：
每个文件的所有 eager 依赖都必须排在它前面，否则报错退出。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dep_graph import scan  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "09-全量阅读路径.md"

# (模块名, 一句话：读这个文件要看什么)
# 顺序 = 依赖顺序；同一站内按「先被谁调用」排。
PATH: list[tuple[str, str]] = [
    # ── 站 0：地基（配置 / 模型来源 / 环境）──
    ("src", "包门面：版本号、对外导出哪些名字。36 行，扫一眼就知道这个项目暴露了什么。"),
    ("src.config", "全部配置项与校验。后面每个 `conf.xxx` 都在这里定义；重点看 `is_moe`、`num_speculative_tokens`、`moe_backend` 怎么互相约束。"),
    ("utils._loader", "转发 shim 的加载器：`src/xxx.py` 只是个壳，真身在 `utils/`。看懂它就知道为什么项目根有 `utils/`。"),
    ("utils.model_hub", "模型下载与路径解析（HF 缓存、`HF_ENDPOINT`、`snapshot_download`）。"),
    ("src.model_hub", "33 行的转发 shim：真身在 `utils/model_hub.py`，这里只做导出。"),
    ("utils.sampling_params", "采样参数的数据结构（temperature/top_p/top_k/min_p/seed）。"),
    ("src.sampling_params", "同上，shim。"),
    ("utils.model_loader", "safetensors 权重加载：怎么按层名映射到模型、怎么处理 tie_word_embeddings。"),
    ("src.utils", "包门面：重导出 cuda_env / profiler / paths 三件。"),
    ("src.utils.paths", "仓库内路径常量（模型目录、bench 输出目录）。"),
    ("utils.run_utils", "跑一次生成的上层封装：warmup、计时、输出 token。**`drain_engine` 的参数传反就是这里的 bug**（已修）。"),
    ("utils", "包门面。"),
    ("src.utils.cuda_env", "环境探测与设置：flashinfer arch list、NCCL 环境变量、可用显存。"),
    ("src.utils.profiler", "分段计时（`cpu_timer`）与统计汇总，`docs/05` 里的数字就是它打出来的。"),
    # ── 站 1：请求与显存（数据结构）──
    ("src.data.block_manager", "分页块管理：`allocate`/`free`/`can_append`/`may_append`。**ZOS 的 `can_append_at` 也在这里**，注意它按「位置」算而 `can_append` 按 `len(seq)` 算。"),
    ("src.data.prefix.hash_util", "块哈希：前缀缓存靠它把 token 块映射成 key。"),
    ("src.data.prefix.radix_tree", "基数树的节点结构（纯数据结构，无锁、无并发）。"),
    ("src.data.prefix.radix_index", "前缀缓存的主体：`match`/`publish`/`pin_match`/`reclaim_pages`。**注意 pin 机制**——被引用的页不能被回收。"),
    ("src.data.prefix.hash_index", "哈希前缀缓存的另一个实现（`--prefix-backend hash` 走这条），比 radix 简单得多。"),
    ("src.data.prefix.noop", "关闭前缀缓存时的空实现，用来对比接口。"),
    ("src.data.sequence", "序列状态机：`token_ids`/`block_table`/`num_cached_tokens`/`status`。**全项目的账本就在这里**，ZOS 只多两个字段 `zos_remap`/`zos_start_pos`。"),
    ("src.data.batch", "跨进程消息定义（msgspec Struct）：`SeqSchedule`/`SchedulerOutput`/`EngineCoreOutput`。**调度与计算之间唯一的契约**。"),
    ("src.data.kv", "KV 管理器：把 block_manager + 前缀缓存 + 序列三者串起来（`allocate`/`append_slot`/`deallocate`/`sync_prefix`）。"),
    ("src.data.prefix", "前缀缓存包门面：按 backend 选 radix / hash / noop 三种实现。"),
    ("src.data", "包门面。"),
    # ── 站 2：并行与通信底座 ──
    ("src.distributed.parallel_state", "TP/PP/DP 的 rank 定义与两种通信域：`_TP`（tp 组）与 `_PP`（pp 组）。**PP 分组是按 tp_size 切片的**，这点影响 `_broadcast_prev_samples`。"),
    ("src.distributed.communication_op", "三个通信原语的封装：`tensor_model_parallel_all_reduce`/`all_gather`/`send`。"),
    ("src.distributed.control_channel", "控制通道（DP 场景下 scheduler 与 worker 之间的命令/结果传递）。"),
    ("src.distributed.utils", "分布式工具函数（`init_distributed_environment` 等）。"),
    ("src.distributed", "包门面。"),
    # ── 站 3：算子层 ──
    ("src.compute.sampler", "采样 kernel：贪心 / top-k / top-p / min-p / gumbel 的融合实现。**`_stage_samples` 之前的结果就是它产生的**。"),
    ("src.compute.layers.ops", "`torch.library.custom_op` 包装的四个融合算子（silu_and_mul / rotary / rmsnorm / fused_add_rmsnorm）。**compile 能 trace 就是靠这里的 fake impl**。"),
    ("src.compute.layers.activation", "激活函数（SiLU 等）的纯 torch 版本，作为 ops 的参考实现。"),
    ("src.compute.layers.rotary", "RoPE 的实现与位置编码接口。"),
    ("src.compute.layers.layernorm", "RMSNorm 的纯 torch 版本。"),
    ("src.compute.kv_ops", "Triton 写的 `store_kvcache` kernel：按 `slot_mapping` 把 k/v 写进分页池。**decode 每层都调它**。"),
    # ── 站 4：注意力与显存池 ──
    ("src.compute.layers._flash_attn_fallback", "没有 flash-attn 时的纯 torch 回退（含 `flash_attn_with_kvcache` 的批量路径）。**本机跑的就是这条**。"),
    ("src.compute.layers.attention", "注意力层：prefill 走 varlen，decode 走 `flash_attn_with_kvcache`。**try/except 选真库还是回退就在这里**。"),
    ("src.compute.layers.linear", "线性层与量化/分片（TP 下按列/行切分）。"),
    ("src.compute.layers.vocab_parallel", "词表并行：最后一层 logits 的 all-gather。"),
    ("src.compute.layers", "包门面。"),
    ("src.compute", "包门面。"),
    ("src.compute.kv_pool", "KV 显存池：`[num_blocks, block_size, num_kv_heads, head_dim]` 的裸张量 + 按 slot 索引。**CUDA Graph 要求它地址固定**。"),
    # ── 站 5：组包与录图 ──
    ("src.compute.batch", "组包缓冲：把调度结果写进 GPU 输入张量（`input_ids`/`positions`/`slot_mapping`/`cache_seqlens`/`block_tables`）。**ZOS 的 gather 在 `build_decode` 末尾**，`_make_pinned_set` 是宿主侧双缓冲。"),
    ("src.compute.cuda_graph", "CUDA Graph 的分档录制与重放：`make_bs_list` 生成 batch size 档位，`_pad_bs` 处理 padding。"),
    # ── 站 6：模型 ──
    ("src.compute.models.loader", "模型加载入口，按配置选 dense 还是 MoE。"),
    ("src.compute.models.qwen3", "Dense Qwen3：embedding → 28 层（attention + MLP）→ 最后 RMSNorm → lm_head。**先读这个，MoE 版只是把 MLP 换掉**。"),
    ("src.compute.models", "包门面。"),
    # ── 站 7：一趟前向 ──
    ("src.compute.model_runner", "**全项目最核心的文件**：`run_forward` 组包 → 前向 → 采样 → `_stage_samples`（D2H）。`build_decode`/`_run_prefill`/`_spec_verify`/`sample_dev` 都在这里。"),
    # ── 站 8：调度 ──
    ("src.control.scheduler", "**第二核心**：`schedule()` 决定这一批算什么（decode 每序列 +1 token、chunked prefill、抢占、前缀命中），`update()` 落账（token、EOS、释放块）。ZOS 版没有 `update()`——落账拆成 `swap_pending`/`finish`/`_apply`，在文件后半段（见 `docs/02-ZOS.md`）。"),
    # ── 站 9：投机（可选支线，但顺序上在 scheduler 之后）──
    ("src.spec.ngram", "n-gram 草稿器：拿最后几个 token 去历史里匹配，边匹配边延长。**纯 CPU、几十微秒**。"),
    ("src.spec.draft", "草稿器的抽象接口（`propose` 的契约）。"),
    ("src.spec.verify", "验证：`greedy_accept_length` 算接受长度、`verify` 组织验证批。**接受长度 j 决定了下一步的位置**。"),
    ("src.spec.engine", "把草稿与验证串成一条链（`SpecEngine`），供 scheduler 调用。"),
    ("src.spec", "包门面。"),
    # ── 站 10：MoE（可选支线）──
    ("src.moe.fused", "MoE 的统一入口 `fused_moe_forward`：按 backend 分发到 cutlass / triton / naive。（它在拓扑上排在后端之后，可以读完两个后端再回来看分发。）"),
    ("src.moe.kernels.cuda", "cutlass 后端的 Python 侧：调 flashinfer 的 `cutlass_fused_moe`，外加自己写的 kernel 绑定（`moe_align_block_size`/`moe_sum`/`act_and_mul`）。"),
    ("src.moe.kernels.triton", "Triton 后端：自己写的 grouped GEMM + 三个辅助 kernel，**和 cuda.py 是同一组接口的两种实现**，对着读最省力。"),
    ("src.moe.experts", "专家层：把 token 按专家排序（align）→ 分组 GEMM → 按权重求和（sum）。"),
    ("src.moe.block", "MoE 的 decoder block：attention + MoE MLP（对应 `qwen3` 里的 dense MLP）。"),
    ("src.moe.parallel", "专家并行的划分（EP）与 all-to-all。"),
    ("src.moe.kernels", "kernel 包门面。"),
    ("src.moe", "包门面。"),
    ("src.compute.models.qwen3_moe", "MoE 版 Qwen：把 `qwen3` 的 MLP 换成 `moe.block`，**对照 `qwen3.py` 读，差异只有几行**。"),
    # ── 站 11：执行与进程 ──
    ("src.compute.worker", "worker：进程初始化（分布式、模型、KV 池）→ `execute_model` 调 `run_forward` → `read_samples`。**PP 下 `_broadcast_prev_samples` 也在这里**。"),
    ("src.control.executor", "执行器：Uniproc（单进程直调）与 Multiproc（ZMQ + 采样 rank 广播）两条路径。**`(n_p, n_d, slot)` 就是从这里透传的**。"),
    ("src.control.core", "**循环本体**：`step()` = 调度 → 发射 → 取结果 → 落账。ZOS 改的就是这四段的顺序与等待点，`_nvtx` 也在这里。"),
    # ── 站 12：对外接口 ──
    ("src.control.client", "客户端门面：`add_request`/`generate`/`has_unfinished`，dp=1 时就地转发。"),
    ("src.control.engine", "引擎门面：把 config、tokenizer、core 组装起来，对外暴露 `generate`。"),
    ("src.control.mp_client", "多进程客户端（dp>1）：ZMQ 与多个 core 通信，结果按 request id 汇总。"),
    ("src.control", "包门面。"),
]

# 20 分钟核心路径：只读这些就能把"一次 decode step"和"ZOS 为什么成立"讲完。
# (模块, 只读哪部分/为什么)
CORE: list[tuple[str, str]] = [
    ("src.config", "只扫 `EngineConfig` 的字段（`max_num_seqs`/`num_speculative_tokens`/`moe_backend`/`block_size`），不用逐行。"),
    ("src.data.sequence", "**账本**：`token_ids`/`num_cached_tokens`/`block_table`/`status` + ZOS 的 `zos_remap`/`zos_start_pos`。"),
    ("src.data.block_manager", "只读 `can_append` / `may_append` 与 **`can_append_at` / `may_append_at`**（按「位置」算块，ZOS 靠它预留）。"),
    ("src.data.kv", "`allocate` / `append_slot` / `sync_prefix` / `deallocate` —— 块、前缀、序列三者的粘合层。"),
    ("src.data.batch", "`_decode_schedule`：注意 `if seq.zos_remap >= 0` 那条分支（`token_ids=[]`、`remap=` 行号）。"),
    ("src.compute.sampler", "只看它输出什么形状、`_gumbel_sample` 怎么工作，不必抠 kernel 细节。"),
    ("src.compute.layers.attention", "prefill 走 varlen、decode 走 `flash_attn_with_kvcache`；`try/except` 选真库还是回退。"),
    ("src.compute.kv_pool", "`alloc_slot` 与池的子张量布局（CUDA Graph 要求地址固定，这是「为什么不能换 buffer」的答案）。"),
    ("src.compute.batch", "`build_decode`（位置/槽位/csl 怎么算）+ **末尾的 gather/remap** + `_make_pinned_set` 双缓冲。"),
    ("src.compute.models.qwen3", "只读 `forward` 的骨架：embedding → N 层 → norm → lm_head，中间每层调 attention + MLP。"),
    ("src.compute.model_runner", "`run_forward` → 采样 → `_stage_samples`（同流 D2H + 双缓冲 slot）；`read_samples` 是唯一阻塞点。"),
    ("src.control.scheduler", "`schedule()` 的 decode 三分支（blocked / 在途 / 普通）+ `_apply()` 落账；`swap_pending`/`finish` 在文件后半段。"),
    ("src.control.core", "**循环本体** `step()`：调度 → 发射 → 取结果 → 落账，四段顺序就是 ZOS 的全部秘密。"),
    ("src.control.executor", "只读 `UniprocExecutor.run` / `read_samples` 与抽象签名 —— 看 `(n_p, n_d, slot)` 怎么从 worker 透传回 core。"),
]


def main() -> None:
    info = scan()
    prod = [m for m in info if m == "src" or m.startswith(("src.", "utils"))]
    known = set(prod)

    order_p = [m for m, _ in PATH]
    dup = {m for m in order_p if order_p.count(m) > 1}
    missing = sorted(known - set(order_p))
    extra = sorted(set(order_p) - known)
    if dup or missing or extra:
        print("PATH 有问题：")
        if dup:
            print("  重复:", sorted(dup))
        if missing:
            print("  漏掉:", missing)
        if extra:
            print("  多出:", extra)
        raise SystemExit(1)

    # 按 PATH 的意图做拓扑排序：每步从「依赖已就绪」的集合里挑 PATH 位置最靠前的那个。
    # 既保证依赖在前，又尽量保留人定的阅读节奏（包 __init__ 会被自动挪到父包之后）。
    prio = {m: i for i, m in enumerate(order_p)}
    done: set[str] = set()
    order: list[str] = []
    while len(order) < len(order_p):
        cand = [m for m in order_p if m not in done and (info[m]["eager"] & known) <= done]
        if not cand:
            print("拓扑排序卡住，剩余:", sorted(known - done))
            raise SystemExit(1)
        cand.sort(key=lambda m: prio[m])
        cur = cand[0]
        order.append(cur)
        done.add(cur)

    # 复核：每个 eager 依赖必须排在前面
    pos = {m: i for i, m in enumerate(order)}
    bad = [(a, d) for a in order for d in info[a]["eager"] & known if pos[d] >= pos[a]]
    if bad:
        print("顺序违反依赖：", bad[:5])
        raise SystemExit(1)
    notes = dict(PATH)

    total = sum(info[m]["lines"] for m in order)
    idx = {m: i for i, m in enumerate(order, 1)}

    SEGMENTS = [
        ("地基", "配置、模型来源、环境、计时", lambda m: m in {
            "src", "src.config", "utils._loader", "utils.model_hub", "src.model_hub",
            "utils.sampling_params", "src.sampling_params", "utils.model_loader",
            "src.utils", "src.utils.paths", "utils.run_utils", "utils",
            "src.utils.cuda_env", "src.utils.profiler"}),
        ("请求与显存", "块、前缀缓存、序列账本、跨进程消息、KV", lambda m: m.startswith("src.data")),
        ("分布式底座", "TP/PP/DP 的 rank 与通信原语", lambda m: m.startswith("src.distributed")),
        ("算子与缓冲", "kernel、融合算子、注意力、组包、录图", lambda m: m.startswith("src.compute") and m not in {
            "src.compute.models.qwen3_moe", "src.compute.models.qwen3", "src.compute.models",
            "src.compute.models.loader", "src.compute.model_runner", "src.compute.worker"}),
        ("模型与前向", "网络结构 + model_runner", lambda m: (m.startswith("src.compute.models") and m != "src.compute.models.qwen3_moe") or m == "src.compute.model_runner"),
        ("投机（可跳）", "草稿 → 验证 → 引擎", lambda m: m.startswith("src.spec")),
        ("调度", "一次 schedule 与落账", lambda m: m == "src.control.scheduler"),
        ("MoE（可跳）", "三后端栈与专家层", lambda m: m.startswith("src.moe") or m == "src.compute.models.qwen3_moe"),
        ("执行与接口", "worker / executor / core / client / engine", lambda m: (m.startswith("src.control") and m != "src.control.scheduler") or m == "src.compute.worker"),
    ]
    seg_rows = []
    used: set[str] = set()
    for title, desc, pred in SEGMENTS:
        hit = [m for m in order if m not in used and pred(m)]
        if not hit:
            continue
        used.update(hit)
        a, b = idx[hit[0]], idx[hit[-1]]
        seg_rows.append(f"| {a}–{b} | {title} | {desc} | {sum(info[m]['lines'] for m in hit)} |")
    rest = [m for m in order if m not in used]
    assert not rest, rest

    skip_lines = sum(info[m]["lines"] for m in order
                     if m.startswith("src.spec") or m.startswith("src.moe")
                     or m == "src.compute.models.qwen3_moe")
    core_set = {m for m, _ in CORE}
    core_lines = sum(info[m]["lines"] for m, _ in CORE)
    core_rows = [f"| {i} | `{info[m]['path'] .relative_to(ROOT).as_posix()}` | {info[m]['lines']} | {note} |" for i, (m, note) in enumerate(CORE, 1)]

    lines = [
        "# 09 阅读路径（核心 20 分钟 / 全量 66 分钟）",
        "",
        f"## 〇、核心路径：14 个文件 / {core_lines:,} 行（约 {core_lines / 250:.0f} 分钟）",
        "",
        "只读这些就能把「一次 decode step 怎么走完」和「ZOS 为什么成立」讲清楚。",
        "其余 57 个文件是工具类、包门面、支线实现，**不用读**（各自需要时再 grep）。",
        "",
        "| # | 文件 | 行数 | 读到哪 |",
        "|---|---|---|---|",
        *core_rows,
        "",
        "### 明确跳过的 57 个",
        "",
        "| 类别 | 文件 | 什么时候才需要看 |",
        "|---|---|---|",
        "| 工具与 shim | `utils/*`（6 个）、`src/utils/*`（4 个）、`src/model_hub.py`、`src/sampling_params.py`、`src/compute/models/loader.py` | 换个模型 / 换镜像源 / 看计时口径时 |",
        "| 包门面 | 全部 `__init__.py`（12 个） | 查某个名字从哪导出时 grep |",
        "| 前缀缓存 | `src/data/prefix/*`（5 个） | 被问到 prefix cache 或 `sync_prefix` 顺序 bug 时（只读 `radix_index`） |",
        "| 投机支线 | `src/spec/*`（5 个） | 被问到投机解码时（`ngram` → `verify` → `engine`） |",
        "| MoE 支线 | `src/moe/*`（8 个）+ `models/qwen3_moe.py` | 被问到 MoE/EP 时（`fused` → `cuda`/`triton` 对照） |",
        "| 并行底座 | `src/distributed/*`（5 个） | 被问到 TP/PP/DP 怎么实现时（`parallel_state` 优先） |",
        "| 算子细节 | `layers/{ops,activation,rotary,layernorm,linear,vocab_parallel}`（6 个） | 被问到融合算子 / compile / TP 切分时 |",
        "| 录图 | `src/compute/cuda_graph.py` | 被问到 CUDA Graph 分档捕获时 |",
        "| 对外接口 | `src/control/{client,engine,mp_client}.py`（3 个） | 被问到 dp>1 / 请求怎么进来时 |",
        "",        "读完自检：能在白板上画出 `schedule → run → read_samples → finish` 四个框、标出哪一步等 GPU、",
        "并说出 ZOS 把「等」挪到哪一步、代价是什么（`input_ids` 留空由 GPU gather）。",
        "",
        "---",
        "",
        "## 一、全量顺序（想通读再看这部分）",
        "",
        "",
        f"产品代码共 **{len(order)} 个模块 / {total:,} 行**。下面是一条严格按依赖排的线性顺序：",
        "**任何一个文件读到时，它依赖的东西都已经读过了**（`tools/gen_read_path.py` 机械检查，违反就报错）。",
        "",
        f"按 200 行/分钟估：全量约 **{total / 200:.0f} 分钟**；跳过投机与 MoE 两条支线约 **{(total - skip_lines) / 200:.0f} 分钟**。",
        "",
        "### 分段速查",
        "",
        "| 行号 | 段 | 这段在讲什么 | 行数 |",
        "|---|---|---|---|",
        *seg_rows,
        "",
        "> 与 `08-依赖图与阅读路径.md` 的分工：08 讲**分层与依赖关系**（包级矩阵、调用链图、六个坑），",
        "> 09 只讲**顺序**——一条包含全部文件的线。先看 08 的 §1/§2 建立地图，再用 09 逐行读。",
        "",
        "| # | 文件 | 行数 | ★ | 读什么 |",
        "|---|---|---|---|---|",
    ]
    for i, m in enumerate(order, 1):
        p = info[m]["path"].relative_to(ROOT).as_posix()
        lines.append(f"| {i} | `{p}` | {info[m]['lines']} | {'★' if m in core_set else ''} | {notes[m]} |")

    lines += [
        "",
        "## 读完主干后可选的两条支线",
        "",
        "上面的顺序里，投机段（`src/spec/*`）与 MoE 段（`src/moe/*` + `qwen3_moe.py`）是**独立支线**：",
        "它们只被 `scheduler` / `qwen3` 各调用一次，跳过不影响理解主干。想省时间就在读完调度后直接跳到「执行与接口」。",
        "",
        "## 测试与压测（不属于产品代码，但读源码时很有用）",
        "",
        "1. `tests/helpers/engine.py`（66 行）：`Engine` 的最小用法，看清「一个请求怎么进来」。",
        "2. `bench/zos_matrix.py`（112 行）：怎么压 n 并发 × 长度，参数怎么传给引擎。",
        "3. `bench/final.sh`：三组正确性回归（sha 对拍）怎么跑。",
        "4. `tests/gpu/*`：需要真卡，本机跑不了。",
    ]

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    print(f"modules {len(order)} / {total} lines; topo OK")


if __name__ == "__main__":
    main()
