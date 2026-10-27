# 零开销调度（Zero-Overhead Scheduling, ZOS）

## 1. 气泡从哪来

一步 decode 的 CPU 循环是这四段：

```
schedule() → run() 发射 → read_samples() 等 D2H → update() 回灌
```

`read_samples()` 是全流程唯一的阻塞点：它等 GPU 把这一步算完。等完之后 CPU 才做 `update()` 和下一步的 `schedule()`，而这两段是纯 CPU 的活 —— 期间 GPU 已经算完、队列里没有下一个 kernel 可跑，于是空转。这段 GPU 空转就是气泡 B：

```
T = G + B          T: 步墙钟   G: GPU 真正在算   B: GPU 等 CPU
加速比 = 1 + B/G
```

## 2. 改法：把「等」推迟一拍

```
schedule(t) → run(t) → swap_pending → read_samples(t-1) → finish(t-1)
```

CPU 发射完第 t 步立刻往下走，去收第 t-1 步的结果。此刻 GPU 已经在跑第 t 步，CPU 的记账与下一步调度被埋进这段时间里。

代码在 `src/control/core.py:173 step()`：

```python
with self._nvtx("zos.schedule"):
    sched = self.scheduler.schedule()
...
with self._nvtx("zos.launch"):
    n_p, n_d, slot = self.executor.run(sched)      # 只发射
...
prev = self.scheduler.swap_pending(sched, n_p, n_d, slot, metric)
if not self.zos:
    prev = self.scheduler.take_pending()
if prev is None:
    return [], 0
p_ids, d_ids = self.executor.read_samples(prev.n_p, prev.n_d, prev.slot)
outputs = self.scheduler.finish(prev, p_ids, d_ids)
```

代价是 CPU 账本比 GPU 落后一步：调度第 t 步时，CPU 只知道第 t-1 步及以前的结果。

## 3. 关键问题：第 t+1 步的输入 token 此刻还不知道

调度第 t+1 步时，第 t 步的采样结果还没回到 CPU。而组包需要：

| 需要什么 | 从哪来 | ZOS 下会不会滞后 |
|---|---|---|
| 位置 `pos` | `num_cached_tokens`（CPU 记账） | 滞后一步 → **乐观 +1** |
| KV 槽位 `block_table[pos // 256]` | 块表（CPU 记账） | 同上，块要按位置预分配 |
| 注意力长度 `csl` | `pos + 1` | 同上 |
| **输入 token id** | 上一步采样结果（GPU） | **确实拿不到** |

前三项都是 CPU 账本，乐观推进即可；只有最后一项必须等 GPU。两种解法：

1. 把 `schedule()` 推迟到采样结果回来之后 —— 等于回到同步模式，气泡仍在。
2. **让 GPU 自己补这一格**：CPU 组包时把 `input_ids` 那一格留空、记下"这一行要取上一步采样结果的第几个"，上一步的结果留在显存里不释放，GPU 端做一次 gather。

实现选了 2：

- `src/data/batch.py:91` 在途行 `token_ids=[]`、`num_cached_tokens=zos_start_pos`、`remap=zos_remap`；
- `src/compute/batch.py:236` gather 段：
  ```python
  idx = self.remap[:n_row]
  valid = idx >= 0
  picked = prev_sampled[idx.clamp(min=0)]
  self.input_ids[:n_row].copy_(torch.where(valid, picked, self.input_ids[:n_row]))
  ```
  `prev_sampled` 就是 `model_runner._sample_dev`：每步采样结果先拷到这块**显存常驻**缓冲（`src/compute/model_runner.py:598`），再往 pinned 内存拷。

## 4. 三条不变式

| 不变式 | 为什么 | 代码 |
|---|---|---|
| 在途序列不可抢占 | 第 N 批还在 GPU 上跑，抢走它的块会写坏别人的 KV | `scheduler.py:184` 在途分支抢不到就 `break`，不往 running 队尾找 |
| 在途序列的块落地前不归还 | 下一批可能还要写它最后一格 | `src/control/scheduler.py:594 _release_later()` 把释放推迟一轮，`finish()` 开头统一 flush |
| 采样结果双缓冲 + 同流拷贝 | 第 N 批的 pinned 缓冲不能被第 N+1 批覆盖；拷贝必须与计算同流 | `model_runner.py:579 _stage_samples()`：两块 event / 两块 pinned，`slot` 随 `run()`/`read_samples()` 传递 |

另外三处配套改动：

- **半截 prefill 挡在门外**：块中间被切开的序列这一轮没有采样结果可回灌，记进 `_blocked`，调度时直接 `break`（`scheduler.py:179`、`:275`）。
- **输入暂存也做双缓冲**：`_stage_samples` 给**采样结果**做双缓冲（那份数据要跨轮存活）；输入的锁页暂存则每步换一份，`Batch.begin_step()` 翻 `_buf_slot`（`src/compute/batch.py:194`，由 `run_forward` 开头调用）。
  只换 CPU 侧缓冲 —— GPU 输入张量仍单份、地址不变，所以 **CUDA Graph 不需要重捕**。两份就够的理由是结构性的：第 t 步的 H2D 排在前向之前入队，而下一轮 `read_samples(t)` 必须等第 t 步的 D2H 兑现，那时流必然已越过 H2D(t)；两轮之后才复用同一槽位，因此不可能覆盖尚有 DMA 在途的缓冲。
  （早期版本用 `cudaEvent.query()` 兜底，那只是「通常不会等」，不构成保证，已被这个方案取代。）
- **按位置算块**：账本落后一步时旧判据 `len(seq) % block_size == 1` 会少分配一块，改用 `can_append_at(seq, pos)` / `may_append_at(seq, pos)`（`src/data/block_manager.py:266`、`src/data/kv.py:179`）。

## 5. 代码地图

| 层 | 位置 | 做了什么 |
|---|---|---|
| 开关 | `src/control/core.py:93` | `KRITACE_ZOS` + 与投机互斥 |
| 循环 | `src/control/core.py:173` | 五段顺序、空批收尾 |
| 在途账本 | `src/control/scheduler.py:521-620` | `swap_pending` / `take_pending` / `_install` / `finish` / `_flush_deferred` / `_release_later` / `_apply` |
| 调度放行 | `src/control/scheduler.py:164` | 三分支：`_blocked` / 在途 / 普通 |
| 组包 | `src/data/batch.py:79`、`src/compute/batch.py:188` | 在途行留空 + GPU gather |
| 显存安全 | `src/data/block_manager.py:266`、`src/data/kv.py:179` | 按逻辑位置算块、推迟释放 |
| 采样分期 | `src/compute/model_runner.py:579`、`:618` | 双缓冲、同流拷贝、显存常驻副本 |
| 跨进程 | `src/control/executor.py:169`、`:321`、`:489`、`:521` | `(n_p, n_d, slot)` 透传 |
| 广播 | `src/compute/worker.py` `_broadcast_prev_samples` | PP 下采样 rank 把 `sample_dev` 广播给全组 |

## 6. 与投机的冲突（摘要）

ZOS 把未知量压到"一个 token id"，投机把它放大成"一个长度 `j`"：下一步的 `pos` / `csl` / `slot`
全由 `num_cached_tokens` 推出来，而它只推进 `j+1` —— 位置是标量，不能像 KV 块那样"最坏预留"。

所以两者互斥：`src/control/core.py:93` 的 `want_zos and config.num_speculative_tokens == 0`，
否则打印一行原因并退回同步路径。另有第二个阻碍点：n-gram 的 `propose()` 在 CPU 上读 `seq.token_ids`，
ZOS 下这个锚点正好还留在显存里。

**完整分析**（两条冲突的机制、为什么"最坏预留"救不了 position、A/B/C 三条出路各要付多少、
为什么不做 B 的收益算术、面试六问）见 [06-ZOS与投机](06-ZOS与投机.md)。

## 7. 怎么验证（三层证据）

```bash
# ① 正确性：两模式逐 token 一致
bash bench/final.sh /tmp/final.txt

# ② 机制：read_samples 是否还需要等
KRITACE_ZOS_STATS=1 KRITACE_ZOS=0 python bench/zos_matrix.py --case short --n 8 --max-tokens 32 --blocks 48
KRITACE_ZOS_STATS=1 KRITACE_ZOS=1 python bench/zos_matrix.py --case short --n 8 --max-tokens 32 --blocks 48

# ③ 气泡：步墙钟 − GPU 忙时
KRITACE_ZOS_GPU=1 python bench/repro_zos.py --rounds 3
```

典型输出（RTX 3050 Laptop 4GB + Qwen3-0.6B，SDPA 回退）：

| 指标 | ZOS=0 | ZOS=1 |
|---|---|---|
| `read_samples` ready 比例 | 3.1% | 100% |
| 累计等待（32 步） | 56.8–76 ms | 0.1–0.3 ms |
| 每步气泡 B | 0.23 ms（bs=8）/ 0.33 ms（bs=32） | ≈ 0 |
| 端到端吞吐 | — | 无显著差异（`B/G ≈ 0.2%`） |

## 8. 什么时候不值得做

`1 + B/G` 是唯一判据：

- 本机一步 130 ms 里 GPU 只忙 15.6 ms（CPU 提交是瓶颈），气泡占 0.2% —— 量不出收益。
- 换到真 flash-attn + CUDA Graph 的机器，`G` 降到 1.5–10 ms、`B` 仍是 0.2–0.3 ms，收益 2%–15%。
- 模型越大 `G` 越大、`B` 不变 ⇒ 大模型上 ZOS 更不值钱。
