# -*- coding: utf-8 -*-
"""把流程图（merged_super_flow.mmd / 推理全流程图-完整版.svg）与项目全景手册接到一起。

产出四样东西：
  1) 后处理过的 SVG：每个流程节点多出
         class="... fm fm-C4"        —— 按手册章节上色
         id="flow-E2"                —— 供手册各章反向链接过来
         data-cid / data-title / data-refs  —— 供悬停面板与搜索用
  2) NODES：{节点 id: {cid, title, refs, kw, label, cites, vllm}}
         cites —— 这一步**精确对应**的代码：文件 + 符号 + 行号 + 真实代码片段
         vllm  —— 同一个环节在 vLLM 官方实现里的对应位置（有才给）
  3) CHAP2NODES：{C4: [E2, D1, ...]}，写进每章开头那行"本章在图上的位置"
  4) 面板用的 CSS/JS（PANEL_CSS / PANEL_JS），手册与独立查看器共用同一套。

代码片段从快照 `_v6src/src/**` 里现读（快照 = 真项目 `~/dzyy-vllm/vllm-v6` 的拷贝；
用 `check_sync` 可核对）。"在 VS Code 打开"用的是 Remote-WSL 深链，
落到**真项目**文件名与行号上。
"""
from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path

# ---------------------------------------------------------------- 章节色板
CID_COLOR = {
    'C1': '#9aa4b2', 'C2': '#5aa9ff', 'C3': '#89ddff', 'C4': '#7ee787',
    'C5': '#56d4dd', 'C6': '#4dd0a5', 'C7': '#ffb86b', 'C8': '#c792ea',
    'C9': '#f78c6c', 'C10': '#82aaff', 'C11': '#ffcb6b', 'C12': '#b0bec5',
}
CID_NAME = {
    'C1': '配置与入口', 'C2': '门面与主循环', 'C3': '进程与并行拓扑', 'C4': '调度器',
    'C5': '数据与 KV 账本', 'C6': '前缀缓存', 'C7': '计算路径', 'C8': '算子与层',
    'C9': '模型实现', 'C10': 'MoE 与专家并行', 'C11': '投机解码', 'C12': '观测与基准',
}

# ---------------------------------------------------------------- 真机（WSL）路径
WSL_DISTRO = 'Ubuntu'
REAL_ROOT = '/home/Kritace/dzyy-vllm/vllm-v6'          # 真项目（VS Code 深链指向这里）
VLLM_ROOT = ('/home/Kritace/miniconda3/envs/vllm-exp/lib/python3.12/'
             'site-packages/vllm')                       # vLLM 0.29.0（官方实现，对比用）
VSCODE_REMOTE = 'vscode://vscode-remote/wsl+%s%s' % (WSL_DISTRO, REAL_ROOT)
VSCODE_VLLM = 'vscode://vscode-remote/wsl+%s%s' % (WSL_DISTRO, VLLM_ROOT)


def vscode_uri(rel: str, line: int | None = None, root: str = 'real') -> str:
    base = VSCODE_REMOTE if root == 'real' else VSCODE_VLLM
    return '%s/%s%s' % (base, rel, (':%d' % line) if line else '')


# ---------------------------------------------------------------- 分类规则
# (正则, 章节, [候选文件（存在才会挂链接）], 关键词)
ID_RULES = [
    (r'^L\d+$', 'C2', ['src/control/engine.py', 'src/control/core.py'], '引擎主循环 连续批处理 step'),
    (r'^RQ$', 'C4', ['src/control/scheduler.py'], 'running 队列 队头 抢占队尾'),
    (r'^W1$', 'C4', ['src/control/scheduler.py'], 'waiting 队列 FIFO 插队'),
    (r'^E2$', 'C4', ['src/control/scheduler.py'], 'schedule ScheduledBatch'),
    (r'^E3$', 'C4', ['src/control/scheduler.py'], 'mix_prefill_decode 混批策略'),
    (r'^M\d$', 'C4', ['src/control/scheduler.py'], '混批 策略 prefill 优先'),
    (r'^D\d+$', 'C4', ['src/control/scheduler.py'], 'decode 通道 出草稿 预算 抢占 记账'),
    (r'^P[A-Z0-9]+$', 'C4', ['src/control/scheduler.py', 'src/data/kv.py'], 'chunked prefill 切块 前缀命中 need_sample'),
    (r'^N\d$', 'C4', ['src/control/scheduler.py'], '归一 定宽 投机归一 ScheduledBatch'),
    (r'^S\d$', 'C11', ['src/spec/ngram.py', 'src/spec/verify.py', 'src/spec/draft.py'], '投机 草稿 K+1 定宽 判接受 j+1'),
    (r'^H\d$', 'C7', ['src/compute/model_runner.py', 'src/compute/batch.py'], 'CPU GPU 交接 H2D D2H synchronize 组表'),
    (r'^X\d$', 'C3', ['src/distributed/communication_op.py'], 'allreduce 集合通信 词表并行 部分和'),
    (r'^Q\d+$', 'C7', ['src/compute/model_runner.py'], 'read_samples 采样回传 切片 stage_samples'),
    (r'^R\d$', 'C4', ['src/control/scheduler.py', 'src/data/kv.py'], '回写 账本 update sync_prefix 判结束 摘除'),
]

# 卡片链上的节点按标签关键词分（同一批代码，落在不同章节）
KW_RULES = [
    ('查表', 'C9', ['src/compute/layers/vocab_parallel.py', 'src/compute/models/qwen3.py'], 'embedding 查表 词表并行'),
    ('lm_head', 'C8', ['src/compute/layers/vocab_parallel.py'], 'lm_head logits 只算需要的位置'),
    ('router', 'C10', ['src/moe/block.py', 'src/moe/experts.py'], 'router top-k 专家 部分和'),
    ('专家', 'C10', ['src/moe/block.py', 'src/moe/experts.py', 'src/moe/parallel.py'], '专家 切片 EP 中间维'),
    ('采样', 'C7', ['src/compute/sampler.py'], '采样 top-k top-p min-p argmax 参数打包'),
    ('qkv_proj', 'C8', ['src/compute/layers/linear.py'], 'qkv_proj 列并行 QKVParallelLinear'),
    ('o_proj', 'C8', ['src/compute/layers/linear.py'], 'o_proj 行并行 RowParallelLinear 部分和'),
    ('attention', 'C8', ['src/compute/layers/attention.py'], 'attention varlen paged cache_seqlens block_tables'),
    ('RoPE', 'C8', ['src/compute/layers/rotary.py'], 'RoPE q/k 头内 norm 位置'),
    ('norm', 'C8', ['src/compute/layers/layernorm.py'], 'RMSNorm norm① norm② 末层 norm'),
    ('写本卡 KV', 'C7', ['src/compute/kv_ops.py', 'src/compute/kv_pool.py'], 'store_kvcache slot_mapping KV 池'),
    ('KV', 'C5', ['src/data/kv.py', 'src/data/block_manager.py'], 'KV 池 块 slot_mapping block_table'),
    ('残差', 'C9', ['src/compute/models/qwen3.py', 'src/compute/models/qwen3_moe.py'], '残差 residual 下一层'),
    ('入参', 'C7', ['src/compute/model_runner.py', 'src/compute/batch.py'], '入参 batch token 形状'),
    ('recv', 'C7', ['src/compute/model_runner.py'], 'PP recv h residual'),
    ('send', 'C7', ['src/compute/model_runner.py'], 'PP send h residual'),
]

# ---------------------------------------------------------------- 精确代码锚点
# 节点 id → [(相对路径, 符号, 一句话)]；符号写法：
#   'Scheduler.schedule'  类.方法（先在文件里找 class，再找它的方法）
#   'store_kvcache'       模块级函数 / 类
#   'L97'                 直接给行号（用于没有名字的代码段）
CITE: dict[str, list[tuple[str, str, str]]] = {
    # ---- 门面与主循环（C2）
    'L1': [('src/control/engine.py', 'Engine.add_request', '门面入口：文本 → Sequence，空 prompt 抛错、超长截头留尾')],
    'L2': [('src/control/engine.py', 'Engine.generate_batch', '主循环：反复调 EngineCore.step()，一步 = 一个批')],
    'L3': [('src/control/core.py', 'EngineCore.has_unfinished', '还有没有没跑完的请求（决定要不要再 step）')],
    'L4': [('src/control/engine.py', 'L276', '空转保护：连续 1024 步没活干就抛 RuntimeError')],
    'L5': [('src/control/engine.py', 'Engine.generate', '循环退出后把 token_id 解码成文本')],
    # ---- 调度器（C4）
    'E2': [('src/control/scheduler.py', 'Scheduler.schedule', '整个调度入口：产出 ScheduledBatch(prefills, decodes)')],
    'E3': [('src/control/scheduler.py', 'Scheduler.schedule', 'mix_prefill_decode 分支：混批还是 prefill 优先'),
           ('src/config.py', 'EngineConfig', 'mix_prefill_decode / max_num_batched_tokens / max_num_seqs 都定义在这')],
    'M1': [('src/control/scheduler.py', 'Scheduler._schedule_mixed', '混合模式：先跑 decode，再把剩余 token 预算给 prefill')],
    'M2': [('src/control/scheduler.py', 'Scheduler._schedule_prefill_first', 'prefill 优先：有 prefill 就一个 decode 都不跑')],
    'W1': [('src/control/scheduler.py', 'Scheduler.__init__', 'waiting / running 两个 deque 的初始化与语义'),
           ('src/control/scheduler.py', 'Scheduler.preempt', '被抢占的序列 appendleft 回 waiting 队头')],
    'RQ': [('src/control/scheduler.py', 'Scheduler._schedule_decode', 'decode 从 running 队头取，抢占牺牲队尾')],
    'D1': [('src/control/scheduler.py', 'Scheduler._schedule_decode', 'decode 通道主干：popleft → 出草稿 → 预算 → 抢占 → 记账'),
           ('src/control/scheduler.py', 'Scheduler._room', '序列数预算：len(prefills) + len(decodes) < max_num_seqs')],
    'D2': [('src/spec/ngram.py', 'NgramDrafter.propose', 'n-gram 查表出草稿：纯 CPU，不碰显存')],
    'D3': [('src/control/scheduler.py', 'Scheduler._schedule_decode', 'token 预算判断：tokens_used + n_new > max_num_batched_tokens 就停')],
    'D4': [('src/control/scheduler.py', 'Scheduler._schedule_decode', '预算用完：appendleft 放回 running 后 break')],
    'D5': [('src/data/kv.py', 'KVManager.can_append_n', '够不够 n_new 个空位（只算，不占）'),
           ('src/data/block_manager.py', 'BlockManager.can_append_n', '块级判断：free + cached 够不够')],
    'D6': [('src/control/scheduler.py', 'Scheduler.preempt', '抢占队尾：丢块（publish=False）+ 丢草稿 + 回 waiting 队头')],
    'D7': [('src/control/scheduler.py', 'Scheduler.preempt', '腾出空间后重试；腾不出来就轮到抢占自己')],
    'D8': [('src/control/scheduler.py', 'Scheduler._schedule_decode', '抢占自己：本序列回 waiting 队头，本步不带它')],
    'D9': [('src/control/scheduler.py', 'Scheduler._schedule_decode', '记账 spec_draft_ids / num_scheduled_tokens 后进 decodes'),
           ('src/data/kv.py', 'KVManager.may_append_n', '真正占页：may_append_n')],
    'D10': [('src/control/scheduler.py', 'Scheduler._schedule_decode', 'extendleft(reversed(decodes))：放回 running 队头，保住 FIFO')],
    'P1': [('src/control/scheduler.py', 'Scheduler._pull_prefill_chunk', 'prefill 通道主干：遍历 list(waiting) 快照')],
    'PA': [('src/control/scheduler.py', 'Scheduler._room', '序列数预算不够就不再拉新的'),
           ('src/control/scheduler.py', 'Scheduler._try_alloc', 'block_table 非空直接复用，否则问 kv.can_allocate')],
    'P3': [('src/control/scheduler.py', 'Scheduler._try_alloc', '按块复用：已经有 block_table 的序列不重新分配')],
    'P4': [('src/data/kv.py', 'KVManager.allocate', '先查前缀能复用就复用，剩下的新分配'),
           ('src/data/kv.py', 'KVManager._fit_prefix', '前缀命中 + 可回收 cached 一起算空间')],
    'P5': [('src/data/kv.py', 'KVManager.can_allocate', '页池不够就进不来（FIFO，后面的只会更差）')],
    'P6': [('src/control/scheduler.py', 'Scheduler._pull_prefill_chunk', 'need = num_tokens − num_cached_tokens 的计算'),
           ('src/data/sequence.py', 'Sequence.last_num_cached_tokens', '前缀缓存命中多少 token 的记账')],
    'P7': [('src/control/scheduler.py', 'Scheduler._pull_prefill_chunk', 'need ≤ 0：整条 prompt 都在前缀缓存里')],
    'P8': [('src/control/scheduler.py', 'Scheduler._pull_prefill_chunk', '一次 prefill 都不做：直接 RUNNING，去参加 decode')],
    'P9': [('src/control/scheduler.py', 'Scheduler._pull_prefill_chunk', 'allow_trailing_partial=False 时不许切半截')],
    'P10': [('src/control/scheduler.py', 'Scheduler._schedule_prefill_first', 'prefill 优先模式：延迟可预测优先'),
            ('src/control/scheduler.py', 'Scheduler._pull_prefill_chunk', '半截序列在 prefill 优先模式下的处理')],
    'P11': [('src/control/scheduler.py', 'Scheduler._pull_prefill_chunk', 'chunk = min(need, remaining)：切在哪由 token 预算决定')],
    'P12': [('src/control/scheduler.py', 'Scheduler._pull_prefill_chunk', 'num_cached_tokens + chunk == num_tokens 恰好走完')],
    'P13': [('src/control/scheduler.py', 'Scheduler._pull_prefill_chunk', '走完 ⇒ need_sample=True、进 running，本步吐第一个 token')],
    'P14': [('src/control/scheduler.py', 'Scheduler._pull_prefill_chunk', '没走完 ⇒ n_partial += 1 后 break（每轮最多一条半截）')],
    'P15': [('src/control/scheduler.py', 'Scheduler._schedule_mixed', '拉不到 prefill ⇒ 退化成纯 decode')],
    'P16': [('src/control/scheduler.py', 'Scheduler._pull_prefill_chunk', '循环退出条件：waiting 空 / 预算用尽 / 序列数满 / 页池不够')],
    'N1': [('src/control/scheduler.py', 'Scheduler._spec_width', '定宽：width = max(K)+1，有草稿的序列从 decodes 移到 prefills')],
    'N2': [('src/data/batch.py', 'ScheduledBatch', '交给执行器的批结构：prefills / decodes 的顺序契约')],
    # ---- 投机解码（C11）
    'S1': [('src/spec/ngram.py', 'NgramDrafter.propose', '① 出草稿：不跑模型 ⇒ 没有草稿 KV、也没有概率 q'),
           ('src/spec/ngram.py', 'NgramDrafter._match_once', 'n-gram 匹配的最长后缀查找')],
    'S2': [('src/spec/engine.py', 'SpecEngine.build_verify_input', '② 验证输入 = [最后一个 token, t₁ … t_K]，共 K+1 个位置')],
    'S3': [('src/spec/engine.py', 'SpecEngine.step', '③ 定宽补齐与一轮投机的位置编排')],
    'S5': [('src/spec/verify.py', 'greedy_accept_length', '④ 判接受：从头起连续匹配几个草稿（贪心）'),
           ('src/spec/engine.py', 'SpecEngine.acceptance', '接受长度 → 本轮实际产出 j+1 个 token')],
    'S6': [('src/control/scheduler.py', 'Scheduler._update_spec_batch', '⑤ 账本只推进 j+1；已写进 KV 的 K−j 个无效槽不回收')],
    'S7': [('src/data/kv.py', 'KVManager.sync_prefix', '⚠ 发布前缀索引必须传 num_new_tokens=j+1（默认 K+1 会发布错前缀）')],
    # ---- CPU/GPU 交接与回写（C7 / C4）
    'H1': [('src/compute/batch.py', 'Batch.__init__', '交接① H2D：pinned 6 张表拷进显存（non_blocking）'),
           ('src/compute/batch.py', '_h2d', '拷表的具体实现')],
    'H2': [('src/compute/model_runner.py', 'ModelRunner.read_samples', '交接② 异步 D2H：结果拷回 pinned 后立刻返回')],
    'H3': [('src/compute/model_runner.py', 'ModelRunner.read_samples', '交接③ 同步点：_copy_event.synchronize() —— CPU 唯一真正阻塞处')],
    'H4': [('src/control/engine.py', 'Engine.generate_batch', '交接④⑤⑥ 流出/回流：这一步的增量 token 交给上层')],
    'Q1': [('src/compute/model_runner.py', 'ModelRunner.read_samples', '按元素数切片：p_ids = s[n_d:n_d+n_p]、d_ids = s[:n_d]'),
           ('src/compute/model_runner.py', 'ModelRunner._stage_samples', '采样结果写回 pinned 暂存区')],
    'Q2': [('src/control/scheduler.py', 'Scheduler.update', '回写入口：先 decodes、后 prefills；再判结束、摘 running')],
    'Q3': [('src/control/scheduler.py', 'Scheduler.update', 'chunked prefill 的中间块：不采样')],
    'Q4': [('src/control/scheduler.py', 'Scheduler.update', '中间块 continue：也不判结束')],
    'Q5': [('src/control/scheduler.py', 'Scheduler.schedule', '批的组成每步都在变（退出若干、进来若干）')],
    'R1': [('src/data/kv.py', 'KVManager.sync_prefix', 'sync_prefix 只把写满的整块登记进前缀索引')],
    'R2': [('src/control/scheduler.py', 'Scheduler._update_spec_batch', '投机批有效长度 j+1，不是 num_scheduled_tokens')],
    'R3': [('src/control/scheduler.py', 'Scheduler.update', '判结束：EOS（ignore_eos 时无视）或 num_completion_tokens ≥ max_tokens'),
           ('src/data/kv.py', 'KVManager.deallocate', '结束即 free：publish=True 保留可复用前缀块')],
    'R4': [('src/control/scheduler.py', 'Scheduler.update', '★ 已完成的序列必须从 running 摘掉（重建 deque）')],
    'R5': [('src/data/batch.py', 'EngineCoreOutput', '增量 token / is_finished 的结构')],
    # ---- 四张卡（C8 算子 / C9 模型 / C10 MoE / C7 计算路径）
    'C0a': [('src/compute/batch.py', 'Batch.build_decode', 'decode 批的组表（四张卡同一份 schedule）'),
            ('src/compute/batch.py', 'Batch.build_prefill', 'prefill 批的组表')],
    'C1a': [('src/compute/model_runner.py', 'ModelRunner.run_forward', '每步进模型前的入口（四卡同批）')],
    'C0b': [('src/compute/layers/vocab_parallel.py', 'VocabParallelEmbedding', '词表并行查表：本卡只查自己那半张（末段补零）'),
            ('src/compute/layers/vocab_parallel.py', '_pad_vocab_size', '151936 → 补到可分片')],
    'C1b': [('src/compute/layers/vocab_parallel.py', 'VocabParallelEmbedding._vocab_range', '本卡负责的词表区间（后半张）')],
    'C0c': [('src/compute/layers/layernorm.py', 'RMSNorm.forward', 'norm①：作用在完整 hidden 上 ⇒ 零通信')],
    'C1c': [('src/compute/layers/layernorm.py', 'RMSNorm.forward', 'norm①：与卡 0 同一份完整 hidden')],
    'C2b': [('src/compute/layers/layernorm.py', 'RMSNorm.forward', 'norm①：PP 收到的 h 已完整，零通信')],
    'C3b': [('src/compute/layers/layernorm.py', 'RMSNorm.forward', 'norm①：同上')],
    'C0d': [('src/compute/layers/linear.py', 'QKVParallelLinear', 'qkv_proj 列并行：切输出维，权重加载期只留本卡行区间'),
            ('src/compute/layers/linear.py', 'QKVParallelLinear._shard_qkv', '按头数分片（8 Q 头 / 4 KV 头）')],
    'C1d': [('src/compute/layers/linear.py', 'QKVParallelLinear', '另外 8 个 Q 头、4 个 KV 头就在这一层里')],
    'C2c': [('src/compute/layers/linear.py', 'QKVParallelLinear', 'PP 第 2 段的 qkv_proj')],
    'C3c': [('src/compute/layers/linear.py', 'QKVParallelLinear', 'PP 第 2 段、另一半头')],
    'C0e': [('src/compute/models/qwen3.py', 'Qwen3Attention._split_norm_rope', 'q/k 头内 norm + RoPE，v 不转'),
            ('src/compute/layers/rotary.py', 'RotaryEmbedding.forward', 'RoPE 本体')],
    'C1e': [('src/compute/layers/rotary.py', 'RotaryEmbedding.forward', '位置来自 positions，与卡无关')],
    'C0f': [('src/compute/kv_ops.py', 'store_kvcache', '写本卡 KV：slot_mapping 由 block_table 换算（唯一换算入口）'),
            ('src/compute/kv_pool.py', 'KVPool', 'KV 池：本卡 paged cache 的物理页')],
    'C1f': [('src/compute/kv_ops.py', 'store_kvcache', '另外 4 个 KV 头写进本卡的页')],
    'C0g': [('src/compute/layers/attention.py', 'attention', 'prefill 走 varlen（cu_seqlens）、decode 走 paged（cache_seqlens + block_tables）'),
            ('src/compute/layers/attention.py', 'Attention.forward', '按是否 prefill 选 kernel')],
    'C1g': [('src/compute/layers/attention.py', 'attention', '另外 8 个 Q 头，同一份 KV 分片')],
    'C2e': [('src/compute/layers/attention.py', 'attention', 'PP 第 2 段的 attention')],
    'C0h': [('src/compute/layers/linear.py', 'RowParallelLinear.forward', 'o_proj 行并行：只有部分和，所以要 allreduce'),
            ('src/compute/layers/linear.py', 'RowParallelLinear', '切输入维 / 求和维')],
    'C1h': [('src/compute/layers/linear.py', 'RowParallelLinear.forward', '另一份部分和，由 allreduce 相加')],
    'C0i': [('src/compute/models/qwen3.py', 'Qwen3DecoderLayer.forward', '残差 + norm② 的编排')],
    'C0j': [('src/moe/block.py', 'SparseMoeBlock.forward', 'router → top-k → 本卡专家切片 ⇒ 部分和'),
            ('src/moe/block.py', 'route_experts', '路由本体（复制权重 ⇒ 各卡路由一致）')],
    'C1j': [('src/moe/block.py', 'SparseMoeBlock.forward', '同权重 ⇒ 路由结果与卡 0 完全一致')],
    'C2h': [('src/moe/block.py', 'SparseMoeBlock.forward', 'PP 第 2 段的 MoE')],
    'C3h': [('src/moe/experts.py', 'Experts.forward', '本卡专家切片的前向（EP 时按专家轴切）')],
    'C0k': [('src/compute/models/qwen3_moe.py', 'Qwen3MoeDecoderLayer.forward', '残差 → 下一层（14 层循环）')],
    'C0l': [('src/compute/model_runner.py', 'ModelRunner._send_intermediate', 'send h 与 residual → 卡 2'),
            ('src/distributed/communication_op.py', 'send_tensors', 'PP 点对点发送')],
    'C1l': [('src/compute/model_runner.py', 'ModelRunner._send_intermediate', 'send h 与 residual → 卡 3')],
    'C2a': [('src/compute/model_runner.py', 'ModelRunner._recv_intermediate', 'recv h 与 residual ← 卡 0'),
            ('src/distributed/communication_op.py', 'recv_tensors', 'PP 点对点接收')],
    'C3a': [('src/compute/model_runner.py', 'ModelRunner._recv_intermediate', 'recv h 与 residual ← 卡 1')],
    'C2j': [('src/compute/models/qwen3.py', 'Qwen3Model.forward', '末层 norm：只有最后一段有')],
    'C2k': [('src/compute/layers/vocab_parallel.py', 'ParallelLMHead.forward', 'lm_head：只算需要的位置（prefill 每序列末位 / decode 1 个）'),
            ('src/compute/models/qwen3.py', 'Qwen3ForCausalLM.compute_logits', '缩到需要的位置再投影')],
    'C3k': [('src/compute/layers/vocab_parallel.py', 'ParallelLMHead', '复制所有权重 ⇒ logits 与卡 2 逐位相同')],
    'C2l': [('src/compute/sampler.py', 'Sampler.forward', '采样：T → top-k → top-p → min-p；all_greedy 时走 argmax'),
            ('src/compute/sampler.py', 'Sampler._sample_ordered', '按顺序施加各个过滤')],
    'C3l': [('src/control/executor.py', '_sampler_rank', '采样只由固定一张卡做：sampler_rank=(pp_size-1)*tp_size'),
            ('src/control/executor.py', 'L259', '其它 rank 执行完就等在 channel 上，只有 sampler_rank 读回样本')],
    # ---- 集合通信与并行（C3）
    'X0': [('src/compute/layers/vocab_parallel.py', 'VocabParallelEmbedding.forward', '词表并行：本卡部分结果 → allreduce 拼完整 h'),
           ('src/distributed/communication_op.py', 'tensor_model_parallel_all_reduce', 'TP allreduce 入口')],
    'X1': [('src/distributed/communication_op.py', 'tensor_model_parallel_all_reduce', '层内第 1 次 allreduce：o_proj 的部分和相加')],
    'X2': [('src/distributed/communication_op.py', 'tensor_model_parallel_all_reduce', '层内第 2 次 allreduce：专家输出的部分和相加')],
    'X5': [('src/distributed/communication_op.py', 'tensor_model_parallel_all_reduce', 'PP 第 2 段：o_proj 部分和')],
    'X6': [('src/distributed/communication_op.py', 'tensor_model_parallel_all_reduce', 'PP 第 2 段：专家部分和')],
    'X3': [('src/compute/model_runner.py', 'ModelRunner._pp_dst', '_pp_dst / _pp_src 决定 PP 上下游'),
           ('src/distributed/communication_op.py', '_p2p', '点对点通信本体（只有这一个方向，不经 CPU）')],
}

# vLLM 官方实现对照：vLLM 0.29.0（conda env vllm-exp）。
# 行号是实测值，直接写死 —— vLLM 源码不在快照里，不去读、只给「位置 + 差别」。
VLLM_SECTIONS: dict[str, dict] = {
    'engine_loop': {
        'refs': [
            ('vllm/v1/engine/core.py', 'EngineCore.step', 597,
             '单步：schedule → execute_model(异步 future) → sample_tokens → update_from_output'),
            ('vllm/v1/engine/core.py', 'EngineCoreProc.run_busy_loop', 1411,
             '常驻忙循环：先收请求，再反复 _process_engine_step 推进'),
            ('vllm/v1/engine/core.py', 'EngineCore.step_with_batch_queue', 638,
             '批队列/异步模式：上一步 future 与本步调度重叠的 step 变体'),
            ('vllm/v1/core/sched/scheduler.py', 'Scheduler.schedule', 501,
             '每步产出一个 token 级的 SchedulerOutput（num_scheduled_tokens）'),
        ],
        'same': '都是连续批处理：每步重新决策，顺序是调度 → 执行 → 抽样 → 按输出更新请求状态。',
        'diff': '我们在 Engine.generate_batch 里对着 EngineCore.step 自旋；空调度计数到 _MAX_IDLE_STEPS=1024 就抛 '
                'RuntimeError。vLLM 把「单步」(EngineCore.step) 和「进程忙循环」(EngineCoreProc.run_busy_loop) 拆开，'
                '没有模型执行时 sleep(0.001) 而不是抛错，还多了批队列重叠(step_with_batch_queue)与 _should_throttle_prefills 限流。',
    },
    'scheduler_switch': {
        'refs': [
            ('vllm/v1/core/sched/scheduler.py', 'Scheduler.schedule', 501,
             '统一调度入口：没有 prefill/decode 阶段之分，按 num_computed_tokens 追平'),
            ('vllm/v1/core/sched/scheduler.py', 'Scheduler.schedule', 550,
             '“First, schedule the RUNNING requests”：先给 running 列表分 token 预算'),
            ('vllm/v1/core/sched/scheduler.py', 'Scheduler.schedule', 670,
             'PRIORITY 策略下抢占 (priority, arrival_time) 最差者；FCFS 则 pop 队尾'),
            ('vllm/v1/core/sched/request_queue.py', 'SchedulingPolicy', 13,
             '策略枚举 FCFS/PRIORITY，决定 waiting 队列用哪种实现'),
        ],
        'same': '默认 FCFS 下都先跑已在跑的解码请求，再用剩下的 token 预算从 waiting 拉 prefill。',
        'diff': '我们用一个布尔开关 mix_prefill_decode 二选一：_schedule_mixed 或 _schedule_prefill_first'
                '（后者只要拉到 prefill 就 return，decode 会被饿死）。vLLM 没有这个开关，永远只有一个 schedule()，'
                '先 running 后 waiting，running 是 list 而不是队列，另外可以用 policy 换成 PRIORITY 堆，'
                '还有 skipped_waiting 队列和 async_scheduling。',
    },
    'chunked_prefill': {
        'refs': [
            ('vllm/config/scheduler.py', 'SchedulerConfig.enable_chunked_prefill', 105,
             '默认 True：允许一条 prefill 跨步切分、吃满剩余 max_num_batched_tokens'),
            ('vllm/config/scheduler.py', 'SchedulerConfig.max_num_batched_tokens', 49,
             '输入预算；小于 max_model_len 时必须开 chunked prefill，否则拒绝长序列'),
            ('vllm/config/scheduler.py', 'SchedulerConfig.max_num_scheduled_tokens', 56,
             '本步总预算（含 decode），默认等于 max_num_batched_tokens'),
            ('vllm/config/scheduler.py', 'SchedulerConfig.long_prefill_token_threshold', 70,
             '非 0 时强制把单个 prefill chunk 截到这个阈值'),
            ('vllm/v1/core/sched/scheduler.py', 'Scheduler.schedule', 954,
             'request_token_budget = min(token_budget, input_budget − draft_slots)'),
            ('vllm/v1/core/sched/scheduler.py', 'Scheduler.schedule', 990,
             '没开 chunked prefill 且新 token 超预算时直接 break，不截断'),
            ('vllm/v1/core/sched/scheduler.py', 'Scheduler.schedule', 998,
             'num_new_tokens = min(num_new_tokens, request_token_budget)：chunk 的切分点'),
        ],
        'same': '都用每步的 token 预算把长 prefill 切块，剩下的预算继续接纳后面的请求。',
        'diff': '我们的切法是 chunk = min(need, remaining)，并用 n_partial（每轮最多一条「半截 prefill」）就 break；'
                '预算只有一个 max_num_batched_tokens，在调度器里自减 tokens_used。vLLM 用两个独立预算 '
                'token_budget 与 input_budget（后者要扣掉草稿槽）、没有 n_partial 计数，chunked prefill 可以整个关掉'
                '（关掉时直接 break 而不是截断），另外还有 long_prefill_token_threshold、mamba 对齐、编码器输入等二次切分。',
    },
    'preempt': {
        'refs': [
            ('vllm/v1/core/sched/scheduler.py', 'Scheduler.schedule', 705,
             'FCFS 下 preempted_req = self.running.pop()：牺牲队尾的解码请求'),
            ('vllm/v1/core/sched/scheduler.py', 'Scheduler._preempt_request', 1405,
             '释放块与编码器缓存，status=PREEMPTED，清零 spec_token_ids'),
            ('vllm/v1/core/sched/scheduler.py', 'Scheduler._preempt_request', 1425,
             '进度归零：num_computed_tokens = 0，也就是完整重算（recompute）'),
            ('vllm/v1/core/sched/scheduler.py', 'Scheduler._preempt_request', 1445,
             'self.waiting.prepend_request(request)：被抢占者插回 waiting 队头'),
            ('vllm/v1/core/sched/scheduler.py', 'Scheduler._free_request_blocks', 2522,
             '把该请求占用的 KV 块交还块池（可能留在前缀缓存里）'),
        ],
        'same': '都从 running 尾部挑牺牲者、把请求放回 waiting 队头，并丢弃已有进度从头重算。',
        'diff': '我们的 preempt(seq) 用 kv.deallocate(publish=False) 明确不把块发布进前缀索引，同时清 spec_draft_ids；'
                'vLLM 由 _free_request_blocks 归还块、num_preemptions += 1、清零 spec_token_ids，并处理异步调度下的 '
                'drop_stale_output / num_stale_output_tokens（旧输出作废）。vLLM v1 只有 recompute 一种抢占，'
                '没有 swap / PreemptionMode 枚举，块一律丢弃而不是换出到 CPU。',
    },
    'kv_blocks': {
        'refs': [
            ('vllm/v1/core/kv_cache_manager.py', 'KVCacheManager', 118,
             'request 级 KV 门面：持有 coordinator，暴露 allocate / get_computed / cache / free'),
            ('vllm/v1/core/kv_cache_manager.py', 'KVCacheManager.allocate_slots', 343,
             '一次完成水位检查 + 前缀命中块复用 + 新块分配，失败返回 None'),
            ('vllm/v1/core/kv_cache_manager.py', 'KVCacheManager.get_computed_blocks', 228,
             '查前缀缓存命中的块 → (块, num_new_computed_tokens, num_external)'),
            ('vllm/v1/core/kv_cache_manager.py', 'KVCacheManager.free', 566,
             '请求结束/被抢占时释放它的块（满块可能留作缓存）'),
            ('vllm/v1/core/kv_cache_coordinator.py', 'KVCacheCoordinator', 65,
             '跨 KV cache 组（混合/滑动窗/Mamba）协调分配与命中的抽象层'),
            ('vllm/v1/core/kv_cache_coordinator.py', 'KVCacheCoordinator.allocate_new_blocks', 271,
             '按组向块池申请新块'),
            ('vllm/v1/core/block_pool.py', 'BlockPool', 143,
             '全局物理块池：空闲块双向链表 + hash→block 映射表'),
            ('vllm/v1/core/block_pool.py', 'BlockPool.get_new_blocks', 647,
             '取 n 个空闲块，不够时按 LRU 逐出已缓存块，取不到就报错'),
            ('vllm/v1/core/block_pool.py', 'BlockPool.free_blocks', 723,
             '把块还回空闲链表（LRU 顺序由还回顺序决定）'),
        ],
        'same': '结构同源：空闲块池 + 每请求块表追加 + 满块进前缀缓存；分配失败就返回 None/抛错，让调度器放弃这一轮。',
        'diff': '我们只有两层：KVManager（can_allocate/allocate/can_append_n/may_append_n/sync_prefix）+ BlockManager'
                '（free 用 deque 当栈、cached_ids 用 OrderedDict 做 LRU、ref_count、_ensure_free_page 先回收再 LRU）。'
                'vLLM 是三层 KVCacheManager → KVCacheCoordinator → BlockPool；allocate_slots 一次完成回收+分配，'
                '还带 watermark_blocks、reserved_blocks、full_sequence_must_fit、num_lookahead_tokens 这些门控；'
                '块可以被多个 attention 组共享，支持非均匀块（Mamba/SWA），命中由 coordinator 统一裁决。',
    },
    'prefix_hash': {
        'refs': [
            ('vllm/v1/core/kv_cache_utils.py', 'hash_block_tokens', 621,
             '块 hash = hash_fn((parent_block_hash, 本块 token 元组, extra_keys))：父 hash 链式传递'),
            ('vllm/v1/core/kv_cache_utils.py', 'get_request_block_hasher', 792,
             '只对写满 hash_block_size 的块增量算 hash，不足一块立刻返回空'),
            ('vllm/v1/core/kv_cache_utils.py', 'BlockHash', 48,
             '块 hash 用 NewType(bytes)，另有 make_block_hash_with_group_id 拼组号'),
            ('vllm/v1/core/kv_cache_utils.py', 'generate_block_hash_extra_keys', 583,
             '把多模态 / LoRA / cache_salt 等特征并进 hash，避免跨请求误命中'),
            ('vllm/v1/core/block_pool.py', 'BlockPool.cache_full_blocks', 225,
             '把满块 promote 并把 hash 写进映射表（跳过 null/被 mask 的块）'),
            ('vllm/v1/core/block_pool.py', 'BlockPool.get_cached_block', 198,
             '按 BlockHashWithGroupId 直接取缓存块并 touch（刷新 LRU）'),
            ('vllm/v1/core/kv_cache_coordinator.py', 'KVCacheCoordinator.find_longest_cache_hit', 407,
             '取最长可复用前缀块序列（子类按单组/混合组分别实现）'),
        ],
        'same': '都是链式块哈希（父 hash + 本块 token ids），只登记写满的整块，按最长前缀命中复用。',
        'diff': '我们用 xxhash.xxh64：把父 hash 当种子写进 hasher 再喂 token ids，并在 match 里用 block.token_ids != toks '
                '逐块比内容兜底；索引有 HashPrefixIndex 与 RadixPrefixIndex 两套，靠 publish/invalidate/release_seq 手工维护。'
                'vLLM 的 hash 输入是 (parent_hash, token_tuple, extra_keys)，走可配 hash_function，键是 BlockHash + group_id，'
                '块→hash 用 BlockHashToBlockMap（允许一块对多 hash），粒度上还有 BlockHashListWithBlockSize / '
                'resolve_block_hashes 来兼容粗块与部分块缓存。',
    },
    'waiting_running': {
        'refs': [
            ('vllm/v1/core/sched/scheduler.py', 'Scheduler.__init__', 198,
             'self.waiting = create_request_queue(self.policy)：按策略建 waiting 队列'),
            ('vllm/v1/core/sched/scheduler.py', 'Scheduler.__init__', 200,
             'self.skipped_waiting：因 max_num_running_reqs 等被跳过的请求单独排队'),
            ('vllm/v1/core/sched/scheduler.py', 'Scheduler.__init__', 201,
             'self.running: list[Request] = []：running 是普通列表而不是队列'),
            ('vllm/v1/core/sched/request_queue.py', 'FCFSRequestQueue', 75,
             'deque 实现的 FCFS 队列，支持 peek 不出队'),
            ('vllm/v1/core/sched/request_queue.py', 'FCFSRequestQueue.prepend_request', 92,
             'appendleft：被抢占/需优先的请求插到队头'),
            ('vllm/v1/core/sched/request_queue.py', 'PriorityRequestQueue', 131,
             'heapq 按 (priority, arrival_time) 排序的队列，prepend 退化成普通插入'),
            ('vllm/v1/core/sched/request_queue.py', 'create_request_queue', 201,
             '按 SchedulingPolicy 选择队列实现的工厂'),
        ],
        'same': 'waiting 都是 FCFS：新请求入队尾、被抢占者回队头（prepend_request / appendleft）。',
        'diff': '我们的 waiting 和 running 都是 deque，decode 用 running.popleft()，抢占用 running.pop() 取队尾。'
                'vLLM 的 running 是 list（按下标顺序遍历），被抢占者靠 waiting.prepend_request 决定回队头；'
                'waiting 是可插拔 RequestQueue —— PRIORITY 下换成 heapq 队列、prepend 语义退化为按优先级插入，'
                '还多了 skipped_waiting 与 peek_request（只看不取）。',
    },
    'sampling_sync': {
        'refs': [
            ('vllm/v1/worker/gpu_model_runner.py', 'GPUModelRunner._sample', 3720,
             '取 sampling_metadata 调 sampler；有投机元数据时改走 rejection_sampler'),
            ('vllm/v1/sample/sampler.py', 'Sampler.forward', 73,
             'logits→float32→logits processors→采样→gather logprobs，返回 SamplerOutput'),
            ('vllm/v1/sample/metadata.py', 'SamplingMetadata', 15,
             '批量采样参数：temperature/top_p/top_k/min_p 张量 + all_greedy/all_random 快路径 + generators'),
            ('vllm/v1/outputs.py', 'SamplerOutput', 279,
             '采样输出容器：sampled_token_ids(GPU 张量) + logprobs_tensors'),
            ('vllm/v1/worker/gpu_model_runner.py', 'GPUModelRunner.__init__', 954,
             '预分配 (max_num_reqs,1) int64 pinned CPU 缓冲与 transfer_event'),
            ('vllm/v1/worker/gpu_model_runner.py', 'GPUModelRunner._to_list', 7659,
             'copy_(non_blocking)→transfer_event.record/synchronize→pinned.tolist()，避免整条流同步'),
            ('vllm/v1/worker/gpu_model_runner.py', 'GPUModelRunner.sample_tokens', 4628,
             '执行采样、记账、写回 input_batch，并可发起 draft token 提议'),
            ('vllm/v1/outputs.py', 'AsyncModelRunnerOutput', 428,
             '异步输出抽象：把 CPU 拷贝交给异步输出流，让调度与拷贝重叠'),
        ],
        'same': '都用 pinned CPU 缓冲 + CUDA event 显式同步来取 token（避免 tolist 触发整条流同步）；采样参数打包成张量批量下发，并带全贪心快路径。',
        'diff': '我们的 read_samples() 在 _copy_event.synchronize() 之后按元素区间切缓冲：p_ids=[n_d:n_d+n_p]、d_ids=[:n_d]，'
                '同一 pinned 缓冲、同一 copy stream，顺序拷两段（这是投机解码需要的切法）。vLLM 固定 (max_num_reqs,1) 缓冲，'
                '_to_list 一次拷全部再 tolist，不区分 prefill/decode 切片；采样在 Worker 内由 Sampler.forward 直接返回 '
                'SamplerOutput(sampled_token_ids, logprobs_tensors)，并把 CPU 拷贝抽成 AsyncModelRunnerOutput'
                '（独立 copy stream + event）以便与调度重叠。',
    },
    'attention': {
        'refs': [
            ('vllm/v1/attention/backend.py', 'AttentionBackend', 59,
             '后端抽象：声明 get_impl_cls/get_builder_cls、支持的 KV cache 与头布局'),
            ('vllm/v1/attention/backends/registry.py', 'AttentionBackendEnum', 34,
             '枚举 FLASH_ATTN/FLASHINFER/TRITON 等全部后端，供按名解析成类'),
            ('vllm/v1/attention/selector.py', 'get_attn_backend', 102,
             '按 AttentionSelectorConfig + 平台挑后端类，支持后端黑名单'),
            ('vllm/v1/attention/selector.py', '_cached_get_attn_backend', 195,
             'lru 缓存选出的后端类，避免每层重复选'),
            ('vllm/model_executor/layers/attention/attention.py', 'Attention', 218,
             '模型侧注意力层：持有 backend，统一管 KV 写入与 forward'),
            ('vllm/model_executor/layers/attention/attention.py', 'use_direct_call', 430,
             '= not opaque_attention_op()：决定 Python 直调还是走 torch.ops 自定义算子'),
            ('vllm/v1/attention/backends/flash_attn.py', 'FlashAttentionImpl', 833,
             'FlashAttention 后端：forward 与 do_kv_cache_update 分离（写 KV / 算注意力两段）'),
        ],
        'same': '都做 varlen prefill（cu_seqlens）+ paged decode（block_table/cache_seqlens），写 KV 都按 slot_mapping。',
        'diff': 'vLLM 把注意力拆成「后端枚举 + 选择器 + 层内 use_direct_call 切换」的三层结构，且有 FA/FlashInfer/Triton 多后端；'
                '我们只有 attention.py:78 一个手写 attention()，写死 flash_attn_varlen_func / flash_attn_with_kvcache，'
                '靠 block_tables 是不是 None 来分流。',
    },
    'linear_tp': {
        'refs': [
            ('vllm/model_executor/layers/linear.py', 'LinearMethodBase', 125,
             '线性层方法插拔基类，量化/非量化实现统一 apply 入口'),
            ('vllm/model_executor/layers/linear.py', 'LinearBase', 239,
             '所有线性层公共基类：管 bias、输出 size 分区、量化方法'),
            ('vllm/model_executor/layers/linear.py', 'ReplicatedLinear', 320,
             '每卡完整权重、无通信（router/gate 用）'),
            ('vllm/model_executor/layers/linear.py', 'ColumnParallelLinear', 425,
             '切输出维；forward:593 按 gather_output 决定要不要 all_gather'),
            ('vllm/model_executor/layers/linear.py', 'RowParallelLinear', 1617,
             '切输入维；forward:1748 按 reduce_results/input_is_parallel 决定 all_reduce'),
            ('vllm/model_executor/layers/linear.py', 'QKVParallelLinear', 989,
             'Q/K/V 打包切分；KV 头不够 tp 时在 rank 间复制 KV 头'),
            ('vllm/model_executor/layers/linear.py', 'MergedColumnParallelLinear', 663,
             '多个输出段合并成一列并行（gate/up），按 shard_id 偏移装载'),
        ],
        'same': '同一套判据：可切的输出维走列并行，需要跨卡求和的走行并行；门/路由用复制层保证各 rank 结果一致。',
        'diff': 'vLLM 每个层都要过 LinearMethodBase（量化方法可插拔），forward 里用 self.quant_method.apply 统一执行；'
                '我们 linear.py:101 的 _shard 在 __init__ 里直接 narrow 权重，forward 手写 F.linear + all_gather/all_reduce，'
                '没有方法插拔这一层。',
    },
    'norm_rope_act': {
        'refs': [
            ('vllm/model_executor/layers/layernorm.py', 'RMSNorm', 37,
             'CustomOp 版 RMSNorm：forward_native:74 / forward_cuda:96 各有实现'),
            ('vllm/model_executor/layers/activation.py', 'SiluAndMul', 112,
             'CustomOp 激活，含 forward_cuda；同文件 MulAndSilu:261 是相反布局'),
            ('vllm/model_executor/layers/rotary_embedding/base.py', 'RotaryEmbeddingBase', 15,
             'RoPE 基类：init_cache:58、forward_cuda:221；按平台分 14 个变体'),
            ('vllm/model_executor/layers/rotary_embedding/common.py', 'ApplyRotaryEmb', 125,
             '独立的「应用旋转」CustomOp，支持 is_neox_style 与是否包 cos/sin'),
            ('vllm/model_executor/layers/fused_qk_norm_rope.py', 'fused_qk_rmsnorm_rope_gate', 138,
             '把 q/k 的 RMSNorm 与 RoPE 融成一个 kernel，省一次访存'),
        ],
        'same': '都只做 RMS 缩放不做均值中心化、都用 fp32 的 cos/sin 表且要求 NeoX 半切分、都把残差加与 norm 融合。',
        'diff': 'vLLM 用 CustomOp 体系（forward_native/forward_cuda/forward_hip 多实现），并额外提供 qk-norm + rope 融合算子'
                '和前缀/交错式 RoPE；我们把 FlashInfer 原语包成 torch.library.custom_op（ops.py:22/37/81/96）自己声明 mutates_args，'
                '没有多平台实现。',
    },
    'model_qwen3': {
        'refs': [
            ('vllm/model_executor/models/qwen3.py', 'Qwen3Attention', 65,
             'q/k/v 拆分 → q_norm/k_norm:150/151 → RoPE → attn，与我们的顺序完全一致'),
            ('vllm/model_executor/models/qwen3.py', 'Qwen3DecoderLayer', 173,
             'pre-LN 层：input_layernorm / self_attn / post_attention_layernorm / mlp'),
            ('vllm/model_executor/models/qwen3.py', 'Qwen3Model', 264,
             '继承 Qwen2Model，复用 PP 切层、embedding 与中间张量逻辑'),
            ('vllm/model_executor/models/qwen3.py', 'Qwen3ForCausalLM', 271,
             '挂 lm_head 并调 logits_processor，只对需要的位置算 logits'),
            ('vllm/model_executor/layers/logits_processor.py', 'LogitsProcessor', 58,
             '只 gather 选中位置的 hidden 再过 lm_head，避免全量算词表投影'),
        ],
        'same': '结构、层循环、q/k 头内 RMSNorm、只对末尾 logits 位置做 lm_head 的思路都是一样的。',
        'diff': 'vLLM 的 Qwen3Model 复用 Qwen2Model 的通用 PP/embedding 骨架，位置选取交给可插拔的 LogitsProcessor 层，'
                'lm_head 是 ParallelLMHead + LogitsProcessor；我们是 qwen3.py:271 自己写 forward，'
                '在 runner 里用 cu_seqlens_q[1:bs+1]-1 选位置。',
    },
    'moe': {
        'refs': [
            ('vllm/model_executor/layers/fused_moe/layer.py', 'FusedMoEFactory', 88,
             '0.29.0 改名后的入口，内部组装 router/kernel/runner 并返回 MoERunner'),
            ('vllm/model_executor/layers/fused_moe/fused_moe.py', 'fused_experts', 1593,
             'Triton 融合专家前向：对齐 → 分组 GEMM → 加权归约一条链'),
            ('vllm/model_executor/layers/fused_moe/fused_moe.py', 'dispatch_fused_moe_kernel', 913,
             '按 M/专家数/量化类型查配置并派发具体 kernel'),
            ('vllm/model_executor/layers/fused_moe/moe_align_block_size.py', 'moe_align_block_size', 11,
             '把不等长的 token→专家 映射补零对齐到 block，供分组 GEMM 用'),
            ('vllm/model_executor/layers/fused_moe/router/fused_topk_router.py', 'fused_topk', 80,
             'router logits→softmax→topk→归一化，融合成一次调用'),
            ('vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py', 'grouped_topk', 80,
             '分组 top-k：先按组打分选出 top 组再在组内选专家（DeepSeek 系）'),
            ('vllm/model_executor/layers/fused_moe/experts/triton_moe.py', 'TritonExperts', 65,
             'Triton 专家实现，继承 FusedMoEExpertsModular'),
            ('vllm/distributed/eplb/eplb_state.py', 'EplbState', 230,
             '专家负载均衡全局状态：统计负载、热重排专家所在卡'),
            ('vllm/distributed/eplb/policy/default.py', 'rebalance_experts', 275,
             '默认 EPLB 策略：复制热点专家并按分层打包重分布'),
        ],
        'same': '都按 token→专家 映射做分组 GEMM、都用 moe_align_block_size 式对齐、都有 Cutlass 与 Triton 两条实现可选。',
        'diff': 'vLLM 是 FusedMoEPrepareAndFinalize（DeepEP/NIXL/Mori 多后端）+ FusedMoEKernel(1600) 组合，'
                '专家并行真跨卡搬 token，还带 EPLB 在线重平衡；我们 fused.py:68 调 cutlass_fused_moe 时 ep_size 恒为 1'
                '（不搬 token，只按 expert_map 本地算），跨卡靠 block.py 里额外一次 all_reduce，没有 EPLB。',
    },
    'sampler': {
        'refs': [
            ('vllm/v1/sample/sampler.py', 'Sampler', 21,
             '采样总入口，forward 里先 apply_logits_processors 再 sample'),
            ('vllm/v1/sample/ops/topk_topp_sampler.py', 'TopKTopPSampler', 77,
             '把 top-k/top-p 采样做成可替换算子，内部含 apply_top_k_top_p:349'),
            ('vllm/v1/sample/logits_processor/builtin.py', 'MinPLogitsProcessor', 23,
             'min-p 实现：阈值 = max_prob × min_p，逐请求生效'),
            ('vllm/v1/sample/metadata.py', 'SamplingMetadata', 15,
             '批量采样元数据，含 all_greedy / all_random 布尔开关'),
        ],
        'same': '都按 温度 → top-k → top-p → min-p 处理，并且都用 all_greedy 这类批级开关走 argmax 快路。',
        'diff': 'vLLM 用可插拔的 logits processors 链（逐请求独立生效）+ 独立的 TopKTopPSampler 算子；'
                '我们 sampler.py:33 用 Gumbel-max 代替 multinomial 以适配 CUDA Graph，而且批里只要有一个请求用 top-k，'
                '整批就走 top-k 路径再用 mask 屏蔽，没有 per-request 的 kernel 派发。',
    },
    'spec_decode': {
        'refs': [
            ('vllm/v1/spec_decode/ngram_proposer.py', 'NgramProposer', 12,
             'CPU 版 n-gram 草稿：维护历史 token 字典批量 propose'),
            ('vllm/v1/spec_decode/ngram_proposer_gpu.py', 'NgramProposerGPU', 217,
             'GPU 版 n-gram 匹配，把匹配搬进 kernel 减少 host 往返'),
            ('vllm/v1/sample/rejection_sampler.py', 'RejectionSampler', 38,
             '拒绝采样验证器；forward:92 张量化一次处理全部草稿位置'),
            ('vllm/v1/spec_decode/metadata.py', 'SpecDecodeMetadata', 10,
             '草稿/目标 logits 的批量索引元数据，把变长草稿拍平成稠密张量'),
        ],
        'same': '都用 n-gram 从历史里找续写当草稿，K 个草稿一次前向验证，接受长度 j 决定只推进 j+1 个 token。',
        'diff': 'vLLM 把接受判定写成张量化的 RejectionSampler（一次算全部位置、支持 top-k/top-p 修正分布），'
                '并配 CPU/GPU 两版 proposer；我们 spec/ngram.py:17 的 propose 返回 list，'
                'spec/verify.py:91 greedy_accept_length 在 host 上逐 token 数连续匹配数，只有贪心接受，'
                'sample_accept 是 Python 逐位 Bernoulli。',
    },
    'pp_p2p': {
        'refs': [
            ('vllm/v1/worker/gpu_worker.py', 'execute_model', 1109,
             'worker 侧一步执行：中间张量收/发 + 采样结果回传'),
            ('vllm/distributed/parallel_state.py', 'GroupCoordinator', 411,
             '通信组抽象；send_tensor_dict:1048 / recv_tensor_dict:1213 传变长张量字典'),
            ('vllm/distributed/parallel_state.py', 'broadcast_tensor_dict', 952,
             '把张量字典广播给全组（含采样结果回传非末段）'),
            ('vllm/v1/worker/gpu_model_runner.py', 'execute_model', 4249,
             'runner 侧 PP 编排：收中间张量 → 跑模型 → 发中间张量/广播采样'),
        ],
        'same': '都是 lockstep：各 stage 每步各跑自己那段层，PP 边界只传 hidden（IntermediateTensors），末段才出 token。',
        'diff': 'vLLM 传的是变长 tensor dict（除 residual 外还有元数据），并有 irecv/isend 异步版与 broadcast 把采样 id 广播回所有 stage；'
                '我们 model_runner.py:150 的 _make_pp_buffer 固定只传 hidden_states 与 residual 两个张量、同步 send/recv，'
                '非末段没有 lm_head，也不是 1F1B。',
    },
    'tp_allreduce': {
        'refs': [
            ('vllm/distributed/parallel_state.py', 'GroupCoordinator', 411,
             'TP 组通信入口；all_reduce:702 先看能不能走 custom allreduce'),
            ('vllm/distributed/device_communicators/custom_all_reduce.py', 'CustomAllreduce', 107,
             '自研 all-reduce（IPC/NVLink），含 capture 与 register_graph_buffers'),
            ('vllm/distributed/parallel_state.py', 'get_tp_group', 1526,
             '取全局 TP GroupCoordinator，供模型层调用'),
            ('vllm/distributed/device_communicators/custom_all_reduce.py', 'custom_all_reduce', 441,
             '模块级快路函数：满足条件时替换 NCCL all_reduce'),
        ],
        'same': '都在 TP 组内对行并行输出做 all_reduce；都为 CUDA Graph 预留固定缓冲（我们在 fused/采样侧预分配，vLLM 直接注册通信 buffer）。',
        'diff': 'vLLM 先尝试 custom allreduce（单节点/NVLink/世界大小受限时走自研 kernel），图捕获时还要 register_graph_buffers 注册 IPC 缓冲；'
                '我们 communication_op.py:85 的 all_reduce 直接 dist.all_reduce(input, op, group)，'
                ':128 的 tensor_model_parallel_all_reduce 只是 get_tp_group() 薄封装，没有 custom allreduce、也没有 pynccl。',
    },
}

# 图上哪些节点对应哪一节对照
VLLM_SECTION: dict[str, str] = {
    'L2': 'engine_loop', 'L3': 'engine_loop', 'L4': 'engine_loop', 'L5': 'engine_loop',
    'E2': 'scheduler_switch', 'E3': 'scheduler_switch', 'M1': 'scheduler_switch', 'M2': 'scheduler_switch',
    'P9': 'chunked_prefill', 'P10': 'chunked_prefill', 'P11': 'chunked_prefill', 'P13': 'chunked_prefill',
    'P14': 'chunked_prefill', 'P16': 'chunked_prefill', 'D3': 'chunked_prefill', 'D4': 'chunked_prefill',
    'D6': 'preempt', 'D7': 'preempt', 'D8': 'preempt', 'D9': 'preempt', 'W1': 'preempt', 'R3': 'preempt',
    'D5': 'kv_blocks', 'P3': 'kv_blocks', 'P4': 'kv_blocks', 'P5': 'kv_blocks', 'PA': 'kv_blocks',
    'S7': 'prefix_hash', 'P6': 'prefix_hash', 'R1': 'prefix_hash',
    'RQ': 'waiting_running', 'D1': 'waiting_running', 'D10': 'waiting_running', 'P1': 'waiting_running',
    'R4': 'waiting_running',
    'Q1': 'sampling_sync', 'Q2': 'sampling_sync', 'R5': 'sampling_sync',
    'H2': 'sampling_sync', 'H3': 'sampling_sync',
    # 四张卡上的算子/模型
    'C0g': 'attention', 'C1g': 'attention', 'C2e': 'attention',
    'C0d': 'linear_tp', 'C1d': 'linear_tp', 'C2c': 'linear_tp', 'C3c': 'linear_tp',
    'C0h': 'linear_tp', 'C1h': 'linear_tp',
    'C0c': 'norm_rope_act', 'C1c': 'norm_rope_act', 'C2b': 'norm_rope_act', 'C3b': 'norm_rope_act',
    'C0e': 'norm_rope_act', 'C1e': 'norm_rope_act',
    'C0i': 'model_qwen3', 'C0k': 'model_qwen3', 'C2j': 'model_qwen3', 'C2k': 'model_qwen3', 'C3k': 'model_qwen3',
    'C0j': 'moe', 'C1j': 'moe', 'C2h': 'moe', 'C3h': 'moe',
    'C2l': 'sampler', 'C3l': 'sampler',
    'S1': 'spec_decode', 'S2': 'spec_decode', 'S3': 'spec_decode', 'S5': 'spec_decode', 'S6': 'spec_decode',
    'C0l': 'pp_p2p', 'C1l': 'pp_p2p', 'C2a': 'pp_p2p', 'C3a': 'pp_p2p', 'X3': 'pp_p2p',
    'X0': 'tp_allreduce', 'X1': 'tp_allreduce', 'X2': 'tp_allreduce', 'X5': 'tp_allreduce', 'X6': 'tp_allreduce',
}

# 节点 → [(vllm 相对路径, 符号, 行号, 一句话)]；对照文字单独给
VLLM: dict[str, list[tuple[str, str, int, str]]] = {
    nid: list(VLLM_SECTIONS[key]['refs']) for nid, key in VLLM_SECTION.items()}
VLLM_CMP: dict[str, list[str]] = {
    nid: [VLLM_SECTIONS[key]['same'], VLLM_SECTIONS[key]['diff']] for nid, key in VLLM_SECTION.items()}

NODE_TAG = re.compile(r'<g class="node([^"]*)" id="([^"]+)"')
# 兼容 ["..."]、("...")、{"..."}、(["..."]) 等所有 mermaid 节点形状
NODE_DEF = re.compile(
    r'^[ \t]*(?!style|classDef|class|linkStyle|subgraph|direction|end\b)'
    r'([A-Za-z][A-Za-z0-9_]*)[ \t]*(?:\(\()?[\[\{]{1,2}"(.*?)"[\]\)\}]{1,2}(?:\)\))?', re.M | re.S)
SVG_STYLE_ADD = """
<style>
g.fm{cursor:pointer}
g.fm>rect{stroke-width:2px !important}
%s
g.fm-hi>rect{stroke:#ff7b72 !important;stroke-width:3.5px !important}
g.fm-dim{opacity:.22}
</style>
"""


def _attr(s: str) -> str:
    return html.escape(s, quote=True)


def classify(nid: str, label: str):
    for pat, cid, files, kw in ID_RULES:
        if re.match(pat, nid):
            return cid, files, kw
    norm = _unescape_tex(label).replace('\\_', '_')
    for token, cid, files, kw in KW_RULES:
        if token in norm or token in label:
            return cid, files, kw
    return None, [], ''


_GATHER_RE = re.compile(r'\\begin\{gathered\}(.*?)\\end\{gathered\}', re.S)
_TEX_ESC = [('\\textbackslash{}', '\\'), ('\\textasciicircum{}', '^'),
            ('\\textasciitilde{}', '~'), ('\\_', '_'), ('\\{', '{'), ('\\}', '}'),
            ('\\$', '$'), ('\\&', '&'), ('\\#', '#'), ('\\%', '%')]


def _unescape_tex(s: str) -> str:
    for a, b in _TEX_ESC:
        s = s.replace(a, b)
    return s


def _gathered_rows(label: str):
    r"""`$$\begin{gathered}\text{…} \\ \text{…} \\ 公式\end{gathered}$$` → 行列表。"""
    m = _GATHER_RE.search(label)
    if not m:
        return None
    out = []
    for row in re.split(r'\\\\', m.group(1)):
        row = row.strip()
        if not row:
            continue
        t = re.match(r'^\\text\{(.*)\}$', row, re.S)
        if t:
            out.append(re.sub(r'\s+', ' ', _unescape_tex(t.group(1))).strip())
        else:
            out.append(row)
    return out or None


def _deemph(s: str) -> str:
    """`<b>x</b>`/`<strong>x</strong>` → `【x】`；其余 HTML 标签去掉（面板是纯文本 + esc）。"""
    s = re.sub(r'<(b|strong)>(.*?)</\1>', r'【\2】', s, flags=re.S)
    s = re.sub(r'<[^>]+>', ' ', s)
    return re.sub(r'[ \t]{2,}', ' ', s).strip()


def short_title(label: str) -> str:
    """标签标题：多行块取开头几行拼起来，清掉 markdown 强调，最长 40 字。"""
    rows = _gathered_rows(label)
    if rows:
        first = ''
        for r in rows:
            if first and (len(first) >= 18 or len(r) < 3):
                break
            if not first and len(rows) > 1 and len(r) <= 6:
                continue                      # 跳过「卡 0 ｜」这种只有卡号的短行
            first += r
    else:
        first = label.split('<br/>')[0].split('<br>')[0].split('\\n')[0]
    first = re.sub(r'\s+', ' ', _deemph(first).replace('**', '').replace('`', '')).strip()
    first = re.sub(r'^[★⚠\s]+', '', first)
    first = re.sub(r'[：:，,；;、|｜]+$', '', first)
    return first[:40]


def plain_label(label: str) -> list[str]:
    """把图上标签拆成面板里显示的行（去掉 markdown 强调）。"""
    rows = _gathered_rows(label)
    if rows:
        return [r for r in rows if r]
    raw = label.replace('<br/>', '\n').replace('<br>', '\n')
    out = []
    for ln in raw.split('\n'):
        s = re.sub(r'[ \t]+', ' ', _deemph(ln.replace('**', '').replace('`', ''))).strip()
        if s:
            out.append(s)
    return out


# ---------------------------------------------------------------- 源码定位与摘录
def _find_line(lines: list[str], sym: str) -> int | None:
    """把符号解析成 1-based 行号；解析不到返回 None。"""
    if sym.startswith('L') and sym[1:].isdigit():
        n = int(sym[1:])
        return n if 1 <= n <= len(lines) else None
    if '.' in sym:
        cls, meth = sym.split('.', 1)
        ci = None
        for i, s in enumerate(lines):
            if re.match(r'^class\s+%s\b' % re.escape(cls), s):
                ci = i
                break
        if ci is None:
            return None
        for i in range(ci + 1, len(lines)):
            s = lines[i]
            if re.match(r'^class\s+\w', s):          # 出了这个类
                break
            if re.match(r'^\s+(?:async\s+)?def\s+%s\b' % re.escape(meth), s):
                return i + 1
        return ci + 1
    # 模块级 def / class
    for i, s in enumerate(lines):
        if re.match(r'^(?:class|(?:async\s+)?def)\s+%s\b' % re.escape(sym), s):
            return i + 1
    return None


def _body(lines: list[str], line: int, cap: int = 40) -> list[str]:
    """取 def/class 起始行的整段代码（按缩进判断结束），最多 cap 行。"""
    i = line - 1
    if i < 0 or i >= len(lines):
        return []
    indent = len(lines[i]) - len(lines[i].lstrip())
    out = [lines[i].rstrip()]
    j = i + 1
    truncated = False
    while j < len(lines):
        s = lines[j]
        if s.strip():
            cur = len(s) - len(s.lstrip())
            if cur <= indent:
                break
        if len(out) >= cap:
            truncated = True
            break
        out.append(s.rstrip())
        j += 1
    while out and not out[-1].strip():
        out.pop()
    if truncated:
        out.append('    # … 其余略；点「在 VS Code 打开」看全文')
    return out


class SourceReader:
    """从快照 src 目录定位符号并摘出真实代码；结果缓存，去重。"""

    def __init__(self, src_root: Path | None = None):
        if src_root is None:
            # rel 形如 'src/control/scheduler.py'（相对 _v6src），所以根就是 _v6src
            src_root = Path(__file__).resolve().parent.parent / '_v6src'
        self.src_root = Path(src_root)
        self._files: dict[str, list[str] | None] = {}
        self._loc: dict[tuple[str, str], tuple[int | None, str, list[str]]] = {}

    def lines(self, rel: str) -> list[str] | None:
        if rel not in self._files:
            p = self.src_root / rel
            try:
                self._files[rel] = p.read_text(encoding='utf-8').splitlines()
            except Exception:
                self._files[rel] = None
        return self._files[rel]

    def locate(self, rel: str, sym: str):
        key = (rel, sym)
        if key in self._loc:
            return self._loc[key]
        lines = self.lines(rel)
        line = _find_line(lines, sym) if lines else None
        code = _body(lines, line) if (lines and line) else []
        self._loc[key] = (line, sym, code)
        return self._loc[key]


def build(mmd_text: str, svg_text: str, known_files: set[str], src_root: Path | None = None):
    """→ (svg_html, NODES, CHAP2NODES, stats)

    NODES[nid] = {cid, title, refs, kw, label, cites:[{f, sym, line, note, code, vc}], vllm:[…]}
    """
    labels = {m.group(1): m.group(2) for m in NODE_DEF.finditer(mmd_text)}
    reader = SourceReader(src_root)
    nodes, chap2nodes, order = {}, {}, []
    for nid, label in labels.items():
        cid, files, kw = classify(nid, label)
        if cid is None:
            continue
        refs = [f for f in files if f in known_files]
        cites, vllm = [], []
        for rel, sym, note in CITE.get(nid, []):
            if rel not in known_files:
                continue
            line, _, code = reader.locate(rel, sym)
            cites.append({'f': rel, 'sym': sym, 'note': note, 'line': line or 0,
                          'code': '\n'.join(code),
                          'vc': vscode_uri(rel, line)})
        for rel, sym, vline, note in VLLM.get(nid, []):
            vllm.append({'f': rel, 'sym': sym, 'note': note, 'line': vline or 0,
                         'code': '', 'vc': vscode_uri(rel, vline, 'vllm')})
        nodes[nid] = {'cid': cid, 'title': short_title(label), 'refs': refs,
                      'kw': (kw + ' ' + label.replace('<br/>', ' ')[:80]).lower(),
                      'label': plain_label(label), 'cites': cites, 'vllm': vllm,
                      'vcmp': VLLM_CMP.get(nid, [])}
        chap2nodes.setdefault(cid, []).append(nid)
        order.append(nid)

    n_cite = sum(1 for n in nodes.values() if n['cites'])
    n_noline = sum(1 for n in nodes.values() for c in n['cites'] if not c['line'])
    stats = {'nodes': len(nodes), 'hit': 0, 'miss': 0, 'unclassified': len(labels) - len(nodes),
             'svg_no_meta': [], 'mmd_no_meta': sorted(set(labels) - set(nodes)),
             'with_cite': n_cite, 'cite_total': sum(len(n['cites']) for n in nodes.values()),
             'cite_no_line': n_noline, 'vllm_nodes': sum(1 for n in nodes.values() if n['vllm'])}

    def repl(m):
        cls, gid = m.group(1), m.group(2)
        mm = re.search(r'-([A-Za-z][A-Za-z0-9_]*)-(\d+)$', gid)
        nid = mm.group(1) if mm else ''
        meta = nodes.get(nid)
        if not meta:
            stats['miss'] += 1
            stats['svg_no_meta'].append(nid or gid)
            return m.group(0)
        stats['hit'] += 1
        return ('<g class="node%s fm fm-%s" id="flow-%s" data-cid="%s" data-title="%s" data-refs="%s"'
                % (cls, meta['cid'], nid, meta['cid'], _attr(meta['title']), _attr(','.join(meta['refs']))))

    svg = NODE_TAG.sub(repl, svg_text)
    css = '\n'.join('g.fm-%s>rect{stroke:%s !important}' % (c, col) for c, col in CID_COLOR.items())
    svg = svg.replace('</svg>', SVG_STYLE_ADD % css + '</svg>') if '</svg>' in svg else svg + (SVG_STYLE_ADD % css)
    return svg, nodes, chap2nodes, stats


# ---------------------------------------------------------------- 图节点 → 手册正文段落
_WORD_RE = re.compile(r'[A-Za-z_][A-Za-z0-9_]{3,}')
_HAN_RE = re.compile(r'[\u4e00-\u9fff]{2,}')


def _tok(text: str) -> set:
    """切词：英文标识符 + 中文整段/二字组（中文没有空格，用二字组当词）。"""
    t = set(w.lower() for w in _WORD_RE.findall(text))
    for h in _HAN_RE.findall(text):
        if len(h) <= 6:
            t.add(h)
        for i in range(len(h) - 1):
            t.add(h[i:i + 2])
    return t


def _block_toks(text: str):
    """正文段落切词，代码块（``` 之间）另算一份（权重减半）。"""
    prose, code = [], []
    for idx, p in enumerate(text.split('```')):
        (prose if idx % 2 == 0 else code).append(p)
    return _tok('\n'.join(prose)), _tok('\n'.join(code))


def strip_md(text: str, cap: int = 260) -> str:
    """把一段 markdown 原文压成可以塞进面板的一行中文。"""
    parts = text.split('```')
    text = ''.join(parts[idx] for idx in range(0, len(parts), 2))
    text = re.sub(r'^\s*([-*+]|\d+[.)])\s+', '', text, flags=re.M)
    text = re.sub(r'`([^`]*)`', r'\1', text)
    text = re.sub(r'\*\*([^*]*)\*\*', r'\1', text)
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:cap] + ('…' if len(text) > cap else '')


def _sentences(text: str) -> list:
    """把一块 markdown 切成句（列表项 / 中文句末标点），供“挑出讲这一步的那一句”。"""
    out = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^\s*([-*+]|\d+[.)])\s+', '', line)
        for s in re.split(r'(?<=[。！？；!?;])', line):
            s = s.strip()
            if s:
                out.append(s)
    return out


def _clean(s: str) -> str:
    s = re.sub(r'`([^`]*)`', r'\1', s)
    s = re.sub(r'\*\*([^*]*)\*\*', r'\1', s)
    s = re.sub(r'^\s*[*_>\s]+', '', s)
    s = re.sub(r'^\s*(Q\d*[：:]|A[：:])\s*', '', s)
    return re.sub(r'\s+', ' ', s).strip()


def _block_penalty(b: dict) -> float:
    """表格/清单不是“讲这一步”的正文，压一压。"""
    t = b.get('text') or ''
    pen = 1.0
    if t.count('|') >= 6:
        pen *= 0.55
    if t.count(':') >= 12 or t.count('、') >= 24:
        pen *= 0.75
    if re.search(r'(清单|目录|总表|速查|术语表|文件列表)', b.get('headtitle') or ''):
        pen *= 0.7
    # 问答 / 自查 / 章节导语不是“讲这一步”的正文
    if re.search(r'(面试问答|自查清单)', b.get('headtitle') or ''):
        pen *= 0.4
    if re.search(r'(这一章解决什么问题)', b.get('headtitle') or ''):
        pen *= 0.5
    if len(t) < 12:
        pen *= 0.8
    # 标题块：正文讲解优先，标题只在没有别的候选时才用
    if len(t) < 70 and not re.search(r'[。；！？]', t):
        pen *= 0.7
    return pen


def _overlap(toks: set, stoks: set) -> set:
    """交集 + 英文标识符的部分包含（drafter ⊂ NgramDrafter，能显著提高命中率）。"""
    hit = set(toks & stoks)
    eng = [t for t in toks if len(t) >= 5 and t.isascii()]
    if eng:
        for x in stoks:
            if len(x) < 4 or not x.isascii():
                continue
            for t in eng:
                if t in x or x in t:
                    hit.add(t)
    return hit


# 人工兜底：某个节点的段落匹配不理想时，钉到指定小节（小节标题里的关键字）。
# 只影响“面板里显示/跳转哪一段”，匹配不到就退回自动结果。
DOC_PIN = {
    # C4 调度器
    'E2': '4.3.1', 'E3': '4.3.1', 'N1': '4.3.1', 'N2': '4.3.1',
    'M1': '4.3.2', 'M2': '4.3.4',
    'D1': '4.3.3', 'D2': '4.3.3', 'D3': '4.3.3', 'D4': '4.3.3', 'D5': '4.3.3',
    'D6': '4.3.5', 'D7': '4.3.3', 'D8': '4.3.5', 'D9': '4.3.3', 'D10': '4.3.3',
    'P1': '4.3.4', 'PA': '4.3.4', 'P3': '4.3.4', 'P4': '6.3.4', 'P5': '4.3.4',
    'P6': '4.3.4', 'P7': '4.3.4', 'P8': '4.3.4', 'P9': '4.3.4', 'P10': '4.3.4',
    'P11': '4.3.4', 'P12': '4.3.4', 'P13': '4.3.4', 'P14': '4.3.4', 'P15': '4.3.4',
    'P16': '4.3.4',
    'R2': '4.3.7', 'R3': '4.3.7', 'R4': '2.3.6', 'R5': '2.3.6',
    'W1': '4.4', 'RQ': '4.4',
    # C2 门面主循环
    'L1': '2.3.2', 'L2': '2.3.3', 'L3': '2.3.3', 'L4': '2.3.3', 'L5': '2.3.8',
    # C3 并行拓扑
    'X0': '3.3.8', 'X1': '3.3.8', 'X2': '3.3.8', 'X5': '3.3.8', 'X6': '3.3.8',
    'X3': '3.3.7',
    # C11 投机解码
    'S1': '11.3.1', 'S2': '11.3.5', 'S3': '11.3.6', 'S4': '11.3.7', 'S5': '11.3.7',
    'S6': '11.3.9', 'S7': '11.3.9',
    # C10 MoE（四张卡的 router 这一步）
    'C0j': '10.3.5', 'C1j': '10.3.5', 'C2h': '10.3.5', 'C3h': '10.3.5',
    # C8 算子与层：四张卡的同一批步骤
    'C0c': '8.3.2', 'C1c': '8.3.2', 'C2b': '8.3.2', 'C3b': '8.3.2',
    'C0i': '8.3.2', 'C1i': '8.3.2', 'C2g': '8.3.2', 'C3g': '8.3.2',
    'C2j': '8.3.2', 'C3j': '8.3.2',
    'C0d': '8.3.3', 'C1d': '8.3.3', 'C2c': '8.3.3', 'C3c': '8.3.3',
    'C0e': '8.3.4', 'C1e': '8.3.4', 'C2d': '8.3.4', 'C3d': '8.3.4',
    'C0f': '8.3.5', 'C0g': '8.3.5', 'C1g': '8.3.5', 'C2e': '8.3.5', 'C3e': '8.3.5',
    'C0h': '8.3.6', 'C1h': '8.3.6', 'C2f': '8.3.6', 'C3f': '8.3.6',
    'C0b': '8.3.8', 'C1b': '8.3.8', 'C2k': '8.3.8', 'C3k': '8.3.8',
    # C9 模型实现
    'C0k': '9.3.5', 'C1k': '9.3.5', 'C2i': '9.3.5', 'C3i': '9.3.5',
    # C7 计算路径：装配 / 收发 / 回传
    'C0a': '7.3.2', 'C1a': '7.3.2', 'C1f': '7.3.4',
    'C0l': '3.3.7', 'C1l': '3.3.7', 'C2a': '3.3.7', 'C3a': '3.3.7',
    'C2l': '7.3.7', 'C3l': '7.3.7',
    'H2': '7.3.7', 'H3': '7.3.7', 'H4': '5.3.7',
    'Q1': '7.3.7', 'Q2': '5.3.6', 'Q3': '5.3.7',
}


def _sect_ok(b: dict, pin: str) -> bool:
    """小节号必须整段匹配：'2.3.3' 不能命中 '12.3.3'，'11.3.1' 不能命中 '11.3.11'。"""
    return bool(re.search(r'(?<![\d.])' + re.escape(pin) + r'(?![\d])', b.get('headtitle') or ''))


def match_blocks(nodes: dict, blocks: list, min_score: float = 0.9, win: int = 150) -> dict:
    """给每个流程节点找出手册里讲它的那一段 + 那一段里最针对它的一两句。

    blocks: [{'cid','k','head','headtitle','text'}]，k 是该块在章节里的序号（render_md 的 collect）。
    打分 = 共有词（英文标识符/中文二字组）的 idf 和，符号名加权，同章加权，长块降权。
    """
    bt = [_block_toks(b.get('text') or '') for b in blocks]
    df = {}
    for p, c in bt:
        for t in p:
            df[t] = df.get(t, 0) + 1
        for t in c:
            df.setdefault(t, 0)
    N = max(1, len(blocks))

    def wt(t: str) -> float:
        return 1.0 + math.log(N / (1.0 + df.get(t, 0)))

    out = {}
    for nid, m in nodes.items():
        toks = _tok('%s %s %s' % (m.get('label', ''), m.get('title', ''), m.get('kw', '')))
        syms = set()
        for c in m.get('cites', []):
            syms |= set(x.lower() for x in _WORD_RE.findall(c.get('sym') or ''))
        toks |= syms
        pin = DOC_PIN.get(nid) or ''
        pinned = [i for i, b in enumerate(blocks) if _sect_ok(b, pin)] if pin else []
        best, bs = None, 0.0
        for pool in ([pinned, range(len(blocks))] if pinned else [range(len(blocks))]):
            for i in pool:
                p, c = bt[i]
                s = 0.0
                for t in _overlap(toks, p):
                    s += wt(t) * (2.5 if t in syms else 1.0)
                for t in _overlap(toks, c):
                    s += 0.5 * wt(t) * (2.5 if t in syms else 1.0)
                if not s:
                    continue
                b = blocks[i]
                if b.get('cid') == m.get('cid'):
                    s *= 1.5
                s *= _block_penalty(b)
                s /= 1.0 + math.log1p(len(b.get('text') or '') / 600.0)
                if s > bs:
                    best, bs = i, s
            if best is not None and bs >= min_score:
                break
        if best is None or bs < min_score:
            continue
        b = blocks[best]
        # 在这块里挑最针对这个节点的一两句（面板里显示的就是它，供用户对照）
        sents = _sentences(b.get('text') or '')
        bsc, bi = -1.0, -1
        for j, s in enumerate(sents):
            st = _tok(s)
            inter = _overlap(toks, st)
            if not inter:
                continue
            sc = sum(wt(t) * (6.0 if t in syms else 1.0) for t in inter)
            if len(s) < 10:
                sc *= 0.6
            if sc > bsc:
                bsc, bi = sc, j
        text = ''
        if bi >= 0:
            text = _clean(sents[bi])
            if len(text) < 70 and bi > 0:
                text = _clean(sents[bi - 1]) + ' ' + text
            text = text[:win] + ('…' if len(text) > win else '')
        hl = []
        low = text.lower()
        for t in sorted(toks, key=lambda x: -wt(x)):
            if t in syms and re.search(r'[A-Za-z_]', t) and t in low:
                hl.append(t)
        for w in _HAN_RE.findall('%s %s' % (m.get('label', ''), m.get('title', ''))):
            if 2 <= len(w) <= 8 and w in text:
                hl.append(w)
        seen, hls = set(), []
        for t in hl:
            if t.lower() not in seen:
                seen.add(t.lower()); hls.append(t)
        out[nid] = {'cid': b.get('cid'), 'k': b.get('k'), 'head': b.get('head', ''),
                    'headtitle': b.get('headtitle', ''), 'text': text, 'hl': hls[:5],
                    'block': strip_md(b.get('text') or ''), 'score': round(bs, 2)}
    return out


# ---------------------------------------------------------------- 面板（手册/查看器共用）
def detail_json(nodes: dict, docs: dict | None = None) -> str:
    """面板要用的数据：每步的完整说明 + 精确代码 + vLLM 对照 + 手册里讲它的那一段。"""
    docs = docs or {}
    data = {}
    for nid, m in nodes.items():
        d = docs.get(nid)
        if not m['cites'] and not m['vllm'] and not d:
            continue
        data[nid] = {
            't': m['title'], 'c': m['cid'], 'n': CID_NAME.get(m['cid'], ''),
            'lab': m['label'],
            'cite': [[c['f'], c['line'], c['sym'], c['note'], c['code'], c['vc']] for c in m['cites']],
            'vllm': [[c['f'], c['line'], c['sym'], c['note'], c['vc']] for c in m['vllm']],
            'vcmp': m.get('vcmp', []),
            'doc': {'a': 'flowdoc-' + nid, 'sec': d['head'], 'sect': d['headtitle'],
                    'cid': d['cid'], 'text': d['text'], 'hl': d.get('hl', []), 'k': d['k']} if d else None,
        }
    return json.dumps(data, ensure_ascii=False, separators=(',', ':'))


PANEL_CSS = """
  #flowpanel { position: fixed; left: 14px; bottom: 14px; width: min(720px, 62vw); max-height: 66vh;
               overflow: auto; z-index: 30; background: #fffffff7; border: 1px solid #b9b1a2;
               border-radius: 10px; padding: 0; box-shadow: 0 10px 30px #0000002e; }
  #flowpanel[hidden] { display: none; }
  #flowpanel .fh { position: sticky; top: 0; background: #fbf9f4; border-bottom: 1px solid #e5ded0;
                   padding: 9px 12px; display: flex; align-items: flex-start; gap: 8px; }
  #flowpanel .fh b { flex: 1; color: #1d2733; font-size: 13.5px; line-height: 1.45;
                     font-family: "Cascadia Mono", Consolas, monospace; }
  #flowpanel .fclose { cursor: pointer; color: #8a8577; border: 0; background: none; font-size: 15px; }
  #flowpanel .fb { padding: 10px 12px 12px; }
  #flowpanel h5 { margin: 10px 0 5px; font-size: 12px; color: #7a715f; font-weight: 600;
                  letter-spacing: .04em; }
  #flowpanel ul.fl { margin: 0; padding-left: 16px; }
  #flowpanel ul.fl li { font-size: 12.5px; color: #2b3644; line-height: 1.6; }
  #flowpanel .fc { border-left: 3px solid #d8cfbd; padding: 0 0 0 8px; margin: 8px 0 0; }
  #flowpanel .fcs { font-family: "Cascadia Mono", Consolas, monospace; font-size: 12px; color: #16613f; }
  #flowpanel .fcn { font-size: 12px; color: #6b6455; margin: 2px 0 5px; }
  #flowpanel pre { margin: 0; background: #f7f5ef; border: 1px solid #e8e2d5; border-radius: 7px;
                   padding: 7px 9px; overflow: auto; max-height: 24vh; font-size: 11.5px;
                   line-height: 1.5; font-family: "Cascadia Mono", Consolas, monospace; color: #22303c; }
  #flowpanel .fa { margin: 5px 0 0; display: flex; flex-wrap: wrap; gap: 6px; }
  #flowpanel .fa a, #flowpanel .fa button { font-size: 11.5px; border-radius: 6px; padding: 2px 8px;
                   text-decoration: none; border: 1px solid #b9d4f5; background: #eef5ff; color: #1c5fa8;
                   cursor: pointer; font-family: inherit; }
  #flowpanel .fa button.fcp { border-color: #7fb37a; background: #eefaf0; color: #1a6b33; font-weight: 600; }
  #flowpanel .fa button.fcp2 { border-color: #d8d2c4; background: #fbfaf6; color: #4a463c; font-weight: 600; }
  #flowpanel .fa a.ok { border-color: #7fb37a; background: #eefaf0; color: #1a6b33; font-weight: 600; }
  #flowpanel .ctag { font-size: 10px; color: #6b6455; background: #f2efe7; border: 1px solid #e3ddd0;
                     border-radius: 4px; padding: 0 3px; margin-left: 4px; }
  #flowpanel .fa a.vc2 { border-color: #ded7c6; background: #f7f5ef; color: #8a8577; }
  #flowpanel .fnote { margin: 10px 0 0; font-size: 11.5px; line-height: 1.6; color: #7a715f;
                      border-top: 1px dashed #e5ded0; padding-top: 8px; }
  #flowpanel .fnote code { background: #f2efe7; padding: 0 4px; border-radius: 4px; }
  #flowpanel .fa a.vc { border-color: #c9b8e8; background: #f4efff; color: #6b3fb5; }
  #flowpanel .fa a:hover, #flowpanel .fa button:hover { filter: brightness(.96); }
  #flowpanel .fch { display: inline-block; background: #eef5ff; border: 1px solid #b9d4f5; color: #1c5fa8;
                    border-radius: 6px; padding: 1px 7px; font-size: 11.5px; text-decoration: none; }
  #flowpanel .fv { border-left: 3px solid #d9c9f0; padding-left: 8px; margin: 8px 0 0; }
  #flowpanel .fv .fcs { color: #6b3fb5; }
  #flowpanel .fdb { background: #fffdf6; border: 1px solid #e8dfc8; border-radius: 8px;
                    padding: 8px 10px 9px; }
  #flowpanel .fdb h5 { margin: 0 0 4px; color: #8a6d1f; }
  #flowpanel .fdh { font-size: 11.5px; color: #7a715f; }
  #flowpanel .fdt { margin: 5px 0 0; font-size: 12.5px; line-height: 1.7; color: #2b3644;
                    border-left: 3px solid #e2c98a; padding-left: 9px; }
  #flowpanel .fa a.fdj { border-color: #e2c98a; background: #fdf6e3; color: #8a6d1f; font-weight: 600; }
  a.fdoc { display: block; height: 0; overflow: hidden; scroll-margin-top: 86px; }
  /* 跳过去只定位，不给正文加任何底色（会看不清字）；这两个类只为清掉历史版本残留 */
  mark.fdmark { background: transparent; color: inherit; }
  .fdhi { background: transparent; box-shadow: none; }
"""

# 宿主页面需要提供两个函数：
#   FLOWCHAP(cid)        → 该章的链接（手册：'#sec-c4'；查看器：'../项目全景手册.html#sec-c4'）
#   FLOWSRC(rel, line)   → 手册里那一行的锚点（没有就给 null）
PANEL_JS = """
function flowPanelInit() {
  var box = document.getElementById('flowpanel');
  if (!box) return;
  // VS Code 内置预览（Simple Browser / webview）会丢掉 vscode:// 这类自定义协议，
  // 但它认识 command: URI —— 于是同一行给两条路：command: 走 VS Code 内部，vscode:// 留给系统浏览器。
  var IN_VSCODE = /vscode-webview|Electron/i.test(navigator.userAgent || '') || !!document.querySelector('meta[name="vscode-webview"]');
  function vscodeCmd(uri, line) {
    if (!uri) return uri;
    var args = [uri, { selection: { startLineNumber: line || 1, startColumn: 1 } }];
    return 'command:vscode.open?' + encodeURIComponent(JSON.stringify(args));
  }
  // 本扩展（dzyy.flowpreview）自己当宿主时会插入 meta[name=flowpreview][content=1] 并打开 enableCommandUris，
  // 于是可以走我们自己的命令：按 项目根 / vLLM site-packages 解析相对路径，比 vscode.open 更可靠。
  var FP = !!document.querySelector('meta[name="flowpreview"][content="1"]');
  function fpCmd(kind, file, line) {
    return 'command:flowpreview.openAt?' + encodeURIComponent(JSON.stringify([kind, file, line || 1]));
  }
  function esc(s) { return (s || '').replace(/[&<>]/g, function (c) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]; }); }
  function copy(txt, btn) {
    function done(ok) {
      var o = btn.textContent;
      btn.textContent = ok ? '✓ 已复制' : '× 复制被拦，请手动选中';
      setTimeout(function () { btn.textContent = o; }, 1200);
    }
    function legacy() {                       // webview 里 clipboard API 可能被拒；execCommand 兜底
      try {
        var ta = document.createElement('textarea');
        ta.value = txt; ta.setAttribute('readonly', '');
        ta.style.position = 'fixed'; ta.style.left = '-9999px'; ta.style.top = '0';
        document.body.appendChild(ta); ta.focus(); ta.select();
        var ok = document.execCommand('copy');
        document.body.removeChild(ta);
        return ok;
      } catch (e) { return false; }
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(txt).then(function () { done(true); }, function () { done(legacy()); });
    } else { done(legacy()); }
  }
  // 跳到手册里讲这一步的那一段：只滚过去定位，不给正文加任何底色/底色高亮（会看不清字）
  function fdUnmark() {
    var ms = document.querySelectorAll('mark.fdmark');
    for (var i = 0; i < ms.length; i++) {
      var p = ms[i].parentNode;
      p.replaceChild(document.createTextNode(ms[i].textContent), ms[i]);
      p.normalize();
    }
    var hs = document.querySelectorAll('.fdhi');
    for (var j = 0; j < hs.length; j++) hs[j].classList.remove('fdhi');
  }
  // 段落锚点是成串的（同一段可能被多个节点钉住）：往后找第一个不是 a.fdoc 的元素，才是正文
  function fdTarget(el) {
    var n = el.nextSibling;
    while (n && (n.nodeType !== 1 || (n.classList && n.classList.contains('fdoc')))) n = n.nextSibling;
    return n || el.parentNode || el;
  }
  function fdGo(a, toks) {
    var el = document.getElementById(a);
    if (!el) return false;
    fdUnmark();                                  // 清掉历史版本留下的 mark/.fdhi
    try { el.scrollIntoView({ block: 'start', behavior: 'smooth' }); } catch (e) { el.scrollIntoView(); }
    if (history.replaceState) { try { history.replaceState(null, '', '#' + a); } catch (e) { } }
    return true;
  }
  window.flowDocGo = fdGo;
  window.flowDetail = function (nid) {
    var d = (window.FLOWD || {})[nid];
    if (!d) { box.hidden = true; return; }
    var h = ['<div class="fh"><b>' + esc(nid + ' · ' + d.t) + '</b>',
             '<button class="fclose" title="关闭">×</button></div><div class="fb">'];
    var ch = (typeof FLOWCHAP === 'function') ? FLOWCHAP(d.c) : null;
    if (d.doc) {
      var da = (typeof FLOWDOC === 'function') ? FLOWDOC(d.doc.a) : ('#' + d.doc.a);
      h.push('<div class="fdb"><h5>手册里讲这一步</h5>');
      if (d.doc.sect) h.push('<div class="fdh">在《' + esc(d.doc.sect) + '》这一节</div>');
      if (d.doc.text) h.push('<p class="fdt">' + esc(d.doc.text) + '</p>');
      h.push('<div class="fa"><a class="fdj" href="' + esc(da) + '" data-fd="' + esc(d.doc.a) +
             '" data-hl="' + esc((d.doc.hl || []).join('|')) + '">↳ 跳到手册里讲这一步的那一段</a>');
      if (ch) h.push('<a class="fch" href="' + ch + '">整章：' + esc(d.n || d.c) + '</a>');
      h.push('</div></div>');
    } else if (ch) {
      h.push('<a class="fch" href="' + ch + '">看手册这一章：' + esc(d.n || d.c) + '</a>');
    }
    if (d.lab && d.lab.length) {
      h.push('<h5>这一步在做什么</h5><ul class="fl">');
      d.lab.forEach(function (s) { h.push('<li>' + esc(s) + '</li>'); });
      h.push('</ul>');
    }
    if (d.cite && d.cite.length) {
      h.push('<h5>精确代码</h5>');
      d.cite.forEach(function (c) {
        var f = c[0], line = c[1], sym = c[2], note = c[3], code = c[4], vc = c[5];
        h.push('<div class="fc"><div class="fcs">' + esc(f) + (line ? ':' + line : '') + '</div>');
        h.push('<div class="fcn">' + esc(sym + (note ? ' — ' + note : '')) + '</div>');
        if (code) h.push('<pre>' + esc(code) + '</pre>');
        h.push('<div class="fa">');
        // 可点预览里首选这一条：直接打开真项目源码的那一行
        if (FP) h.push('<a class="ok" href="' + fpCmd('src', f, line) + '" title="flowpreview.openAt：直接打开 ' + esc(f) + (line ? ':' + line : '') + '">↗ 打开源码这一行</a>');
        // 主按钮：复制 路径:行 —— 任何宿主（含 VS Code 内置预览）都有效，粘到 Ctrl+P 就能跳
        h.push('<button class="fcp" data-copy="' + esc(f + (line ? ':' + line : '')) + '">⧉ 复制 路径:行</button>');
        // 次按钮：复制 code --goto 命令 —— 任何宿主都有效，粘到 WSL 终端就能跳
        h.push('<button class="fcp2" data-copy="' + esc('code --goto ' + f + (line ? ':' + line : '')) + '" title="粘到 WSL 终端执行即可打开这一行">⌘ 复制 code --goto</button>');
        var src = (typeof FLOWSRC === 'function') ? FLOWSRC(f, line) : null;
        if (src) h.push('<a href="' + src + '">看手册里的这一行</a>');
        h.push('<a class="vc" href="' + vscodeCmd(vc, line) + '" title="command:vscode.open?… —— 只有声明了 enableCommandUris 的 webview 才响应，VS Code 内置预览不响应">↗ VS Code 打开<span class="ctag">command:</span></a>');
        h.push('<a class="vc2" href="' + esc(vc) + '" title="自定义协议：只在系统浏览器里打开本页时可用">↗ 浏览器（vscode://）</a>');
        h.push('</div></div>');
      });
    }
    if (d.vcmp && d.vcmp.length) {
      h.push('<h5>和 vLLM 官方实现对照</h5>');
      h.push('<div class="fv"><div class="fcn"><b>一样的地方：</b>' + esc(d.vcmp[0]) + '</div>');
      h.push('<div class="fcn"><b>不一样的地方：</b>' + esc(d.vcmp[1]) + '</div></div>');
    }
    if (d.vllm && d.vllm.length) {
      h.push('<h5>vLLM 里的对应位置</h5>');
      d.vllm.forEach(function (c) {
        h.push('<div class="fv"><div class="fcs">' + esc(c[0] + (c[1] ? ':' + c[1] : '')) + '</div>');
        h.push('<div class="fcn">' + esc(c[2] + (c[3] ? ' — ' + c[3] : '')) + '</div>');
        h.push('<div class="fa">');
        if (FP) h.push('<a class="ok" href="' + fpCmd('vllm', c[0], c[1]) + '" title="flowpreview.openAt：在 vLLM 官方源码里打开这一行">↗ 打开 vLLM 这一行</a>');
        h.push('<a class="vc" href="' + vscodeCmd(c[4], c[1]) + '">↗ VS Code 打开 vLLM</a>'
               + '<button class="fcp2" data-copy="' + esc('code --goto ' + c[0] + (c[1] ? ':' + c[1] : '')) + '">⌘ 复制 code --goto</button></div></div>');
      });
    }
    if (FP) {
      h.push('<p class="fnote"><b>可点预览已就绪：</b>绿色按钮 <b>↗ 打开源码这一行</b> / <b>↗ 打开 vLLM 这一行</b> '
             + '会直接在编辑器里打开对应文件并选中该行，<b>默认开在右侧</b>（预览页不会被盖住；'
             + '要改成当前列就把 <code>flowpreview.openBeside</code> 设为 <code>false</code>）。'
             + '复制按钮仍然保留，方便把位置贴给别人。</p>');
    } else if (IN_VSCODE) {
      h.push('<p class="fnote"><b>为什么点不动：</b>VS Code 的内置预览（Simple Browser / HTML 预览）本质是个 webview，'
             + '<code>command:</code> URI 与 <code>vscode://</code> 自定义协议都会被它拦掉 —— webview 的 '
             + '<code>enableCommandUris</code> 默认 <b>false</b>，只有写这个预览扩展的作者能开，用户侧没有开关。'
             + '想点一下就跳代码，请用命令 <b>dzyy: 打开项目全景手册（可点预览）</b> 或 '
             + '<b>dzyy: 打开流程图查看器（可点预览）</b>（扩展 <code>dzyy.flowpreview</code>）；'
             + '或者用复制按钮：<b>⧉ 复制 路径:行</b> → 粘到 <b>Ctrl+P</b>；'
             + '<b>⌘ 复制 code --goto</b> → 粘到 WSL 终端（Enter 即开）。</p>');
    }
    h.push('</div>');
    box.innerHTML = h.join('');
    box.hidden = false;
    box.querySelector('.fclose').onclick = function () { box.hidden = true; };
    Array.prototype.forEach.call(box.querySelectorAll('button[data-copy]'), function (b) {
      b.onclick = function () { copy(b.dataset.copy, b); };
    });
    var j = box.querySelector('a.fdj');
    if (j) {
      j.onclick = function (ev) {
        var href = j.getAttribute('href') || '';
        if (href.charAt(0) !== '#') return;            // 查看器里是跨文档链接，交给浏览器
        ev.preventDefault();
        fdGo(j.getAttribute('data-fd'),
             (j.getAttribute('data-hl') || '').split('|').filter(function (x) { return x; }));
      };
    }
  };
  Array.prototype.forEach.call(document.querySelectorAll('g.fm'), function (g) {
    g.addEventListener('click', function (ev) { ev.stopPropagation(); flowDetail(g.dataset.nid || (g.id || '').replace('flow-', '')); });
  });
}
"""


if __name__ == '__main__':
    import io
    import sys

    sys.stdout.reconfigure(encoding='utf-8')
    d = Path(r'D:\deepseek-harness\dzyy_vllm\_v6src\docs\diagrams')
    mmd = io.open(d / 'merged_super_flow.mmd', encoding='utf-8').read()
    svg = io.open(d / '推理全流程图-完整版.svg', encoding='utf-8').read()
    root = Path(r'D:\deepseek-harness\dzyy_vllm\_v6src')
    known = {p.relative_to(root).as_posix() for p in root.rglob('*.py')}
    s, nodes, c2n, st = build(mmd, svg, known)
    print(st)
    for c in sorted(c2n):
        print(' ', c, CID_NAME.get(c, ''), len(c2n[c]))
    for nid in ('E2', 'D2', 'P11', 'C0d', 'X1', 'Q1', 'L4', 'C0f'):
        m = nodes.get(nid) or {}
        print('--- %s %s' % (nid, m.get('title')))
        for c in m.get('cites', []):
            print('    %s:%s  %s  [%s]' % (c['f'], c['line'], c['sym'], c['note']))
            if c['code']:
                print('        ' + c['code'].splitlines()[0].strip())
