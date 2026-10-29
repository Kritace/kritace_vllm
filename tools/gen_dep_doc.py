# -*- coding: utf-8 -*-
"""生成 docs/08-依赖图与阅读路径.md。

依赖数据来自 tools/dep_graph.py（同一套 ast 扫描 + Tarjan 分层）。
手写的部分是两张图、阅读路径、坑；表格是算出来的。

    python3 tools/gen_dep_doc.py
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dep_graph import layers, scan  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "08-依赖图与阅读路径.md"

LABEL = {
    "src": "src 顶层",
    "src.control": "control",
    "src.compute": "compute",
    "src.data": "data",
    "src.distributed": "distributed",
    "src.moe": "moe",
    "src.spec": "spec",
    "src.utils": "src/utils",
    "utils": "utils/",
}


def path_of(mod: str) -> str:
    p = mod.replace(".", "/")
    if (ROOT / (p + ".py")).exists():
        return f"`{p}.py`"
    return f"`{p}/__init__.py`"


HEAD = """# 08 文件依赖图与逐行阅读路径

这份文档解决两个问题：

1. **谁依赖谁** —— §1 职责分层与包级依赖矩阵、§2 一次 decode step 的调用链、§6 全量依赖表（产品代码 {NPROD} 个模块 / {LINES} 行）。
2. **按什么顺序读** —— §4 的十二站路径，每站都写了「读完要能回答什么」，答不上来就别往下走。

> §2 那张调用链是手画的（表达"一 step 里谁调谁"）；§1.1 矩阵、§1.2 包级图、§6 表格与层号都是算出来的
> （`tools/dep_graph.py` 用 ast 扫 import、Tarjan 求环、按最长依赖路径分层）。
> 表格里的层号会被 `__init__.py` 的重导出抬高（一个包只要 `from .x import Y`，它就在 `x` 之上），所以和 §1 的"职责层"不必一一对应。
> 改完代码重跑 `python3 tools/gen_dep_doc.py` 刷新表格，别手改生成物。

## 0. 一分钟版

```
入口             src/control/engine.py               Engine（门面：add_request / generate / abort）
  └─ 客户端      src/control/client.py               dp=1 就地转发
     └─ 多进程   src/control/mp_client.py            dp>1 时 ZMQ 副本 + 后台 core 进程
        └─ 核心循环 src/control/core.py              step(): 调度 → 发射 → 取结果 → 回灌
             ├─ 调度 src/control/scheduler.py        schedule() / finish()
             │   └─ 账本 src/data/{sequence,kv,block_manager,batch,prefix/*}.py
             ├─ 执行 src/control/executor.py         单卡直调 / 多卡 rank0 广播
             │   └─ 单卡 src/compute/worker.py
             │       └─ src/compute/model_runner.py  组包 + 前向 + 采样（+ CUDA Graph / compile）
             │            ├─ src/compute/batch.py    CPU 张量 → pinned → 显存
             │            ├─ src/compute/models/*    28 层怎么组装
             │            └─ src/compute/layers/*    单层算子
             └─ 显存 src/compute/kv_pool.py         真实 KV 显存池
```

一句话记住分层：**control 不碰张量，compute 不碰调度**；`data/` 是两边共用的 CPU 侧结构；`distributed/` 只在建进程时被执行到。

## 1. 分层总览

**职责分层**（自上而下，下层不知道上层存在）：

| 层 | 目录 / 文件 | 职责 | 大概行数 |
|---|---|---|---|
| 入口 | `src/control/engine.py`、`client.py`、`mp_client.py` | 对外 API；dp=1 就地转发、dp>1 起 ZMQ 副本 | 870 |
| 循环 | `src/control/core.py` | `step()`：调度 → 发射 → 取结果 → 回灌 | 280 |
| 执行 | `src/control/executor.py`、`src/compute/worker.py` | 单卡直调 / 多卡 rank0 广播；一进程一卡 | 950 |
| 推理 | `src/compute/model_runner.py`、`models/*` | 组包、前向、采样、CUDA Graph / compile | 1620 |
| 显存 | `src/compute/{batch,kv_pool,cuda_graph}.py` | 组包缓冲、KV 显存池、录图 | 680 |
| 算子 | `src/compute/layers/*`、`kv_ops.py`、`sampler.py` | 单层算子、写 KV、采样 kernel | 1300 |
| 调度 | `src/control/scheduler.py`、`src/data/*` | 准入、账本、块分配、前缀缓存、跨进程消息 | 1900 |
| 基础 | `src/config.py`、`distributed/*`、`utils/*`、`spec/*`、`moe/*` | 配置、并行组、模型加载、投机、MoE | 其余 |

**为什么这么分**：`control` 不碰张量，`compute` 不碰调度 —— 这条线决定了改代码时该动哪边。
`data/` 是两边共用的 CPU 侧结构（`Sequence`、`ScheduledBatch`），`distributed/` 只在建进程时被执行到。

## 1.1 包级依赖矩阵

行 = 谁 import，列 = 被 import；数字是**模块对**的数量（`src.control.scheduler → src.data.kv` 算 1）。
只统计模块顶层的 eager import（函数体里的延迟 import 不算，见 §5 坑 2）。

{MATRIX}

读法：`src.control` 依赖 `src.data`（{C_D} 处）、`src.compute`（{C_C} 处），反向一处都没有 —— 这就是"上层依赖下层"的量化形式。
被引用最多的三个包：{INBOUND}。"src 顶层"是 `src/config.py`、`src/model_hub.py`、`src/sampling_params.py` 这几个散装模块的合集 —— 所有层都要读配置，所以它排第一，这不是设计缺陷。

## 1.2 包级依赖图

```mermaid
flowchart LR
{PMERMAID}
```

（同一个矩阵的画法。粗线=依赖多，细线=只一两处引用。`utils/` 是 vendored 的模型加载工具，只被 `src` 的 shim 引用。）

机器算出的细粒度分层（层号 = 最长依赖路径，会被 `__init__` 的重导出抬高，和上面表格的"职责层"不必一一对应）：

{LAYER_TABLE}

## 2. 一次 decode step 里谁调谁

```mermaid
flowchart TB
  A["client.generate()<br/>循环调 step()"]
  B["core.step()"]
  C["scheduler.schedule()"]
  C1["_schedule_decode()<br/>每序列 +1 token"]
  C2["_pull_prefill_chunk()<br/>chunked prefill"]
  D["data.kv.can_append / may_append"]
  D1["block_manager<br/>按块边界分配"]
  E["data.batch._decode_schedule()<br/>msgspec 消息"]
  F["executor.run(sched)"]
  G["worker.execute_model()"]
  H["model_runner.run_forward()"]
  H1["compute.batch.build_decode()<br/>pos / slot / csl / block_table"]
  H2["_h2d(): pinned → 显存<br/>同一条流，不等"]
  H3["models.qwen3.forward()<br/>28 × (attn + mlp)"]
  H4["layers.attention<br/>decode: flash_attn_with_kvcache"]
  H5["kv_ops.store_kvcache()<br/>写 slot_mapping"]
  H6["sampler.argmax / top-p"]
  H7["_stage_samples()<br/>采样结果 D2H"]
  I["_copy_event.synchronize()<br/>全循环唯一阻塞点"]
  J["_apply(): 落 token / 判 EOS / 还块"]
  A --> B
  B -->|"① schedule()"| C
  C --> C1
  C --> C2
  C1 --> D
  D --> D1
  C --> E
  B -->|"② executor.run()"| F
  F --> G
  G --> H
  H --> H1
  H1 --> H2
  H --> H3
  H3 --> H4
  H4 --> H5
  H --> H6
  H6 --> H7
  B -->|"③ read_samples()"| I
  B -->|"④ scheduler.finish()"| J
  J --> A
```

> ZOS 打开时 ③④ 的时间点会变（把"等"挪到下一轮），但这条链上的函数不变 —— 差异见 `docs/02-ZOS.md`。

## 3. 反向索引：想改某个东西，看哪几个文件

| 想改什么 | 先看这里 | 通常还要动 |
|---|---|---|
| 调度策略（先来先服务、抢占） | `src/control/scheduler.py` | `src/data/kv.py`、`src/data/block_manager.py` |
| 请求的生命周期 / 状态 | `src/data/sequence.py` | `src/control/scheduler.py` |
| KV 块分配、前缀命中 | `src/data/block_manager.py`、`src/data/prefix/` | `src/data/kv.py`、`src/compute/kv_pool.py` |
| 组包（哪些数上显存、怎么同步） | `src/compute/batch.py` | `src/data/batch.py`（跨进程消息） |
| 单层算子 / 注意力 | `src/compute/layers/` | `src/compute/kv_ops.py` |
| 模型结构（层数、hidden） | `src/compute/models/` | `src/config.py` |
| 采样 | `src/compute/sampler.py` | `src/compute/model_runner.py` 的 `_sample_logits` |
| CUDA Graph / compile | `src/compute/cuda_graph.py`、`src/compute/model_runner.py` | `src/compute/layers/ops.py`（custom_op） |
| 并行（TP/PP/DP） | `src/distributed/`、`src/control/executor.py`、`src/control/mp_client.py` | `src/compute/layers/linear.py`（切分）、`src/compute/worker.py`（PP 广播） |
| MoE | `src/moe/` | `src/compute/models/qwen3_moe.py` |
| 投机解码 | `src/spec/`、`src/control/scheduler.py` 的 `_update_spec_batch` | `src/compute/model_runner.py`（verify 分支） |
| ZOS | `src/control/core.py` 的 `step()`、`src/control/scheduler.py` 的 `swap_pending`/`finish` | `src/data/batch.py:79`、`src/compute/batch.py:236` |
| 对外 API | `src/control/engine.py`、`src/control/client.py` | `utils/sampling_params.py` |

## 4. 十二站阅读路径

原则：**每站只读它依赖的层**（§1 保证依赖都在下层），每站读完用一个问题自检，答不上来就回读。
全部约 {PROD_LINES} 行，按 60–90 行/分钟算，净读 5–7 小时；建议分三次。

### 第 1 站 `src/config.py`（294 行）
- 读什么：`EngineConfig` 的字段与默认值、`__post_init__` 的校验、`from_cli()`。
- 为什么先读：后面每个文件都在读这些字段，先建立"有哪些旋钮"的索引。
- 自检：`block_size` / `max_num_batched_tokens` / `num_speculative_tokens` 三者之间有强约束吗？

### 第 2 站 `src/data/sequence.py`（210 行）
- 读什么：`Sequence` 字段、`SequenceStatus`、`append_token()`、`num_blocks`、`slots_for_range()`。
- 关键：`token_ids`（CPU 侧历史）与 `num_cached_tokens`（KV 里已落地的长度）是**两个**数；decode 时前者比后者多 1。
- 自检：`num_cached_tokens` 在什么时候才 += ？（后处理里，不是组包时）

### 第 3 站 `src/data/block_manager.py`（275 行）+ `src/data/kv.py`（237 行）
- 读什么：块怎么分配/回收；`can_append` 为什么只在"跨块边界"时要新块；`KVManager` 怎么把 `Sequence` 和块表绑在一起。
- 自检：一条序列从 1 个 token 长到 `block_size + 1` 个 token，`block_table` 在第几个 token 时变长？

### 第 4 站 `src/data/prefix/`（hash_util 52 → radix_tree 271 → radix_index 201 / hash_index 109 → noop 51）
- 读什么：前缀复用的匹配与发布；`pin` 与 `reclaim` 的区别（正在被用的页不能回收）。
- 自检：两次相同 prompt，第二次的 `num_cached_tokens` 起始值是多少？谁写的？

### 第 5 站 `src/control/scheduler.py`（670 行）
- 读什么：`schedule()` → `_schedule_decode()` / `_pull_prefill_chunk()` → `_schedule_mixed()`；`preempt()`；`finish()` → `_apply()`。
- 关键不变式：decode 每步每序列只排 1 个 token（投机时 K+1）；prefill 可以分块。
- 自检：一条正在 decode 的序列为什么会被抢占？抢占后它的 KV 还在吗？

### 第 6 站 `src/data/batch.py`（234 行）
- 读什么：`ScheduledBatch` / `SeqSchedule`（msgspec.Struct）与 `to_wire()`；`_decode_schedule()` 里那条断言。
- 关键：这是**进程边界**上的消息定义，字段名就是协议。
- 自检：为什么 decode 的 `SeqSchedule.token_ids` 只有一个元素，prefill 却是一个切片？

### 第 7 站 `src/compute/batch.py`（427 行）
- 读什么：`build_decode()` 的四个标量（`pos` / `slot` / `csl` / `block_table`）从哪来；`_h2d` 为什么是同一条流的 non_blocking；`_make_pinned_set` 的双缓冲；CG 的 padding 安全值。
- 自检：`slot = block_table[pos // 256] * 256 + pos % 256` 里三个 256 分别是什么？

### 第 8 站 `src/compute/layers/`（layernorm 51 → rotary 100 → ops 109 → attention 225 → linear 517 / vocab_parallel 296）+ `src/compute/kv_ops.py`（107）
- 读什么：算子怎么包（custom_op）、注意力 prefill 与 decode 两条路、TP 切分。
- 自检：`attention.py` 里 decode 那条路要哪几个参数？为什么 prefill 不需要按页寻址？

### 第 9 站 `src/compute/models/qwen3.py`（425 行）
- 读什么：embedding → N 层 → 末层 norm → lm_head 的组装；PP 时哪些层在本 rank。
- 自检：PP=2 时不是最后一个 stage 的 rank 还做采样吗？答案在 `model_runner.py` 里。

### 第 10 站 `src/compute/model_runner.py`（738 行）
- 读什么：`run_forward()` 的分派、`_run_decode()` / `_run_prefill()`、`_sample_logits()`、`_stage_samples()` / `read_samples()`、CUDA Graph 与 compile 的挂载点。
- 自检：`read_samples()` 里那个 `synchronize()` 为什么是整条循环里唯一必须等 GPU 的地方？

### 第 11 站 `src/compute/worker.py`（364 行）+ `src/control/executor.py`（588 行）
- 读什么：`Worker` 怎么装配（分布式初始化、模型、KV 池、CG）、`execute_model()`；然后 `UniprocExecutor` 直调与 `MultiprocExecutor` 的 `collective_rpc`。
- 自检：TP 下采样结果从哪个 rank 回 driver？PP 下为什么每步要广播一次采样结果？

### 第 12 站 `src/control/core.py`（278 行）→ `client.py`（130）→ `engine.py`（366）→ `mp_client.py`（375）
- 读什么：`step()` 的四段（ZOS 时五段）、空批收尾、`destroy()`；然后门面怎么进来、`dp>1` 怎么展开。
- 自检：`dp>1` 时谁做负载均衡？`Engine.generate` 的进度条式输出从哪来？

### 读完主干之后（按需）
- MoE：`src/moe/fused.py` → `kernels/cuda.py` → `kernels/moe_kernels.cu` → `experts.py` → `block.py` → `models/qwen3_moe.py`。
- 投机：`src/spec/ngram.py` → `src/control/scheduler.py`（`_update_spec_batch`、草稿批走 prefill 通道）→ `src/spec/verify.py`。
- ZOS：`docs/02-ZOS.md` → `src/control/core.py` 的 `step()` → `src/control/scheduler.py`（`swap_pending`/`finish`）→ `src/data/batch.py:79` → `src/compute/batch.py:236` → `src/compute/model_runner.py`（`_stage_samples`/`read_samples`）。

### 4.5 十二站没有覆盖到的文件（{UNCOV} 个 / {UNCOV_LINES} 行）

十二站是**主干**：它覆盖了 {COV}/{NPROD} 个模块、{COV_LINES} 行（{COV_PCT}%），足以讲清"一次请求怎么变成一次前向"。
剩下的是三条支线（MoE / 投机 / 分布式）和基础设施（显存池、录图、采样 kernel、工具层、包的 `__init__`），按需补读。
每组第一行是**建议顺序**，括号里是行数。

{MISSING}

要不要读 `tests/` 和 `bench/`？建议顺序：先 `tests/helpers/engine.py`（66 行，看清 `Engine` 的最小用法），
再 `bench/zos_matrix.py`（112 行，看清怎么压测），最后 `tests/gpu/*`（需要真卡）。

## 5. 读代码时容易踩的六个坑

1. **`src/__init__.py` 用 `__getattr__` 懒加载**：`import src` 不会拉起 torch，`from src import Engine` 才会。读包入口时别以为它什么都没干。
2. **模块级没有环，但有一条靠延迟 import 打破的环**：`client.py` 在函数里 `from .mp_client import MPClient`（建 dp>1 客户端时才执行），而 `mp_client.py` 顶层就 `from .client import EngineCoreClient`。所以 A 依赖 B、B 又依赖 A，只是其中一条在函数体里 —— 这也解释了为什么 §6 的"层"把它们排在同一层。
3. **各层 `__init__.py` 会重导出**：`from .model_runner import ModelRunner` 让你在 `worker.py` 看到的 `ModelRunner` 其实来自第 10 站。跳转用 IDE 的 go to definition，别看 `__init__`。
4. **`utils/` 是 vendored 的**（原在仓库外）：`src/sampling_params.py`、`src/model_hub.py`、`src/compute/models/loader.py` 只是转发 shim，靠 `utils/_loader.export_module()` 把实现挂回来。改这几处要知道改的是同一份实现，不是三份。
5. **`compute/layers/attention.py` 顶部是 `try: import flash_attn / except: 纯 torch 回退`**：两条路参数不同（回退版多一个 `max_seqlen_k`），读 decode 分支注意 `_USING_FALLBACK`。
6. **`moe/kernels/cuda.py` 在 import 时 JIT 编译 `.cu`**：第一次 import 会等几十秒，是正常的；`msgspec.Struct` 的字段名就是跨进程协议，改字段名会让 driver 与 worker 对不上。

## 6. 全量依赖表

列说明：**依赖**= 本文件顶层（eager）import 的本仓库模块；**延迟**= 写在函数体里的 import；**被依赖**= 谁 import 了它。
按层从高到低排列 —— 也就是"自顶向下读"的倒序；想从底层读，从表尾往上读。

{TABLE}
"""


def main() -> None:
    info = scan()
    mod_lvl, comps = layers(info)
    known = set(info)
    rev: dict[str, set[str]] = defaultdict(set)
    for m in info:
        for d in info[m]["eager"] & known:
            rev[d].add(m)

    prod = [m for m in info if m == "src" or m.startswith(("src.", "utils"))]
    by_lvl: dict[int, list[str]] = defaultdict(list)
    for m in prod:
        by_lvl[mod_lvl[m]].append(m)

    # ---- 包级依赖矩阵 ----
    PKGS = ["src", "src.control", "src.compute", "src.data", "src.distributed",
            "src.moe", "src.spec", "src.utils", "utils"]

    def pkg(mod: str) -> str:
        for p in sorted(PKGS, key=len, reverse=True):
            if mod == p or mod.startswith(p + "."):
                return p
        return "?"

    mat: dict[tuple[str, str], int] = defaultdict(int)
    for m in prod:
        for d in info[m]["eager"] & known:
            a, b = pkg(m), pkg(d)
            if a != b and a in PKGS and b in PKGS:
                mat[(a, b)] += 1

    head = "| 谁 ↓ / 被谁 → | " + " | ".join(LABEL[p] for p in PKGS) + " |"
    sep = "|---|" + "---|" * len(PKGS)
    lines = [head, sep]
    for a in PKGS:
        cells = []
        for b in PKGS:
            v = mat.get((a, b), 0)
            cells.append("—" if v == 0 else str(v))
        lines.append(f"| **{LABEL[a]}** | " + " | ".join(cells) + " |")
    matrix = "\n".join(lines)

    inbound = {p: sum(mat[(a, p)] for a in PKGS) for p in PKGS}
    top3 = sorted(((v, p) for p, v in inbound.items()), reverse=True)[:3]
    inbound_txt = "、".join(f"`{LABEL[p]}`（{v} 处）" for v, p in top3)

    nodes = []
    for i, p in enumerate(PKGS):
        mods = [m for m in prod if pkg(m) == p]
        nodes.append(f'  P{i}["{p.replace("src.", "src/")}<br/>{len(mods)} 模块 / {sum(info[m]["lines"] for m in mods)} 行"]')
    edges = []
    for i, a in enumerate(PKGS):
        for j, b in enumerate(PKGS):
            n = mat.get((a, b), 0)
            if n:
                edges.append(f'  P{i} -->|"{n}"| P{j}')
    pmermaid = "\n".join(nodes + edges)

    # ---- 第四节的覆盖率与"未覆盖清单" ----
    sec = HEAD.split("## 4. 十二站阅读路径")[1].split("## 5.")[0]
    files = set(re.findall(r"`(src/[\w/]+\.py)`", sec))
    dirs = set(re.findall(r"`(src/[\w/]+/)`", sec))
    abbrev = {"src/control/client.py", "src/control/engine.py", "src/control/mp_client.py"}
    covered = set()
    for m in prod:
        p = info[m]["path"].relative_to(ROOT).as_posix()
        if p in files or p in abbrev or any(p.startswith(d) for d in dirs):
            covered.add(m)
    missing = [m for m in prod if m not in covered]
    miss_set = set(missing)

    ORDER = {
        "MoE 支线": ["src.moe.fused", "src.moe.kernels.cuda", "src.moe.kernels.triton",
                     "src.moe.experts", "src.moe.block", "src.compute.models.qwen3_moe",
                     "src.moe.parallel"],
        "投机支线": ["src.spec.ngram", "src.spec.draft", "src.spec.engine", "src.spec.verify"],
        "分布式底座（TP/PP/DP）": ["src.distributed.parallel_state", "src.distributed.communication_op",
                                  "src.distributed.control_channel", "src.distributed.utils"],
        "显存 / 录图 / 采样 kernel": ["src.compute.kv_pool", "src.compute.cuda_graph", "src.compute.sampler"],
        "工具层（模型下载、环境、计时）": ["src.utils.paths", "src.utils.cuda_env", "src.utils.profiler",
                                  "utils.model_hub", "utils.model_loader", "utils._loader",
                                  "utils.run_utils", "utils.sampling_params", "src.model_hub",
                                  "src.sampling_params", "src.compute.models.loader"],
        "包的对外契约（扫一眼即可）": ["src", "src.control", "src.data", "src.compute",
                              "src.compute.models", "src.moe", "src.moe.kernels", "src.spec",
                              "src.utils", "src.distributed", "utils"],
    }

    used: set[str] = set()
    bullets: list[str] = []
    for title, order in ORDER.items():
        hits = [m for m in order if m in miss_set]
        if not hits:
            continue
        used.update(hits)
        chain = " → ".join(f"`{info[m]['path'].relative_to(ROOT).as_posix()}`（{info[m]['lines']}）" for m in hits)
        bullets.append(f"- **{title}**：{chain}")

    rest = sorted((m for m in missing if m not in used), key=lambda m: -info[m]["lines"])
    if rest:
        bullets.append(
            "- **其他**："
            + "、".join(f"`{info[m]['path'].relative_to(ROOT).as_posix()}`（{info[m]['lines']}）" for m in rest)
        )
    missing_md = "\n".join(bullets)

    cov_lines = sum(info[m]["lines"] for m in covered)
    prod_lines = sum(info[m]["lines"] for m in prod)

    rows = [
        "| 层 | 模块 | 行数 | 依赖（→） | 延迟 import | 被依赖（←） |",
        "|---|---|---|---|---|---|",
    ]
    for l in sorted(by_lvl, reverse=True):
        for m in sorted(by_lvl[l]):
            deps = ", ".join(sorted(info[m]["eager"] & known)) or "—"
            lz = ", ".join(sorted(info[m]["lazy"] & known)) or "—"
            users = ", ".join(sorted(rev[m])) or "—"
            rows.append(f"| L{l} | {path_of(m)} | {info[m]['lines']} | {deps} | {lz} | {users} |")

    desc = {
        0: "叶子：配置、块管理、采样/写 KV 的 kernel、分布式基础、MoE 与投机的底层核、`utils/`",
        1: "单层算子（attention/layernorm）、包级 `__init__`、loader/shim、通信与 Triton MoE",
        2: "调度器、请求状态机、投机引擎、各子包的 `__init__`",
        3: "KV 池、CUDA Graph、linear/vocab_parallel、块表、前缀索引、MoE 融合层",
        4: "组包缓冲、dense 模型、MoE 专家",
        5: "一步推理主体：model_runner、MoE 模型与 block",
        6: "单卡进程：worker",
        7: "执行器：单卡/多卡分叉",
        8: "核心循环：core",
        9: "客户端：dp=1 就地转发 / dp>1 ZMQ",
        10: "门面与多进程核心：engine、mp_client",
    }
    lt = ["| 层 | 模块数 | 行数 | 内容 |", "|---|---|---|---|"]
    for l in sorted(by_lvl):
        mods = by_lvl[l]
        lt.append(f"| L{l} | {len(mods)} | {sum(info[m]['lines'] for m in mods)} | {desc.get(l, '')} |")

    text = (
        HEAD.replace("{TABLE}", "\n".join(rows))
        .replace("{MATRIX}", matrix)
        .replace("{PMERMAID}", pmermaid)
        .replace("{INBOUND}", inbound_txt)
        .replace("{C_D}", str(mat.get(("src.control", "src.data"), 0)))
        .replace("{C_C}", str(mat.get(("src.control", "src.compute"), 0)))
        .replace("{LAYER_TABLE}", "\n".join(lt))
        .replace("{NPROD}", str(len(prod)))
        .replace("{LINES}", str(sum(info[m]["lines"] for m in prod)))
        .replace("{PROD_LINES}", f"{sum(info[m]['lines'] for m in prod):,}")
        .replace("{UNCOV}", str(len(missing)))
        .replace("{UNCOV_LINES}", f"{sum(info[m]['lines'] for m in missing):,}")
        .replace("{COV}", str(len(covered)))
        .replace("{COV_LINES}", f"{cov_lines:,}")
        .replace("{COV_PCT}", f"{cov_lines / prod_lines * 100:.0f}")
        .replace("{MISSING}", missing_md)
    )
    if "{LAYER_TABLE}" in text:
        text = text.replace("{LAYER_TABLE}", "\n".join(lt))
    OUT.write_text(text, encoding="utf-8", newline="\n")
    cyc = [c for c in comps if len(c) > 1]
    print(f"wrote {OUT} ({len(text.encode('utf-8'))} bytes, {text.count(chr(10)) + 1} lines)")
    print(f"product modules {len(prod)} / {sum(info[m]['lines'] for m in prod)} lines; eager cycles: {cyc or 'none'}")


if __name__ == "__main__":
    main()
