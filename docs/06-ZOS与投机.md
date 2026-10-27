# ZOS 与投机解码：为什么这两件事现在不能同时开

> 一句话：**ZOS 要求 CPU 在"还不知道上一步结果"的时候就把下一步排好；投机解码恰恰是"下一步的起点要等上一步验证完才知道"。**
> 前者把未知量压缩到"一个 token id"，后者把未知量放大成"一个长度 j"。这不是实现没写完，是两个前提互斥。

本文只讲这一件事：冲突是什么、为什么绕不过去、三条出路各要付多少、以及为什么最后选了"互斥"。
ZOS 本身的机制见 [02-ZOS](02-ZOS.md)，投机的机制见 [03-投机解码](03-投机解码.md)。

---

## 1. 把两套机制的前提摆在一起

### 1.1 ZOS 的前提：未知量只有一个

ZOS 让 `schedule(t+1)` 发生在 GPU 还在算第 t 步的时候。排一批 decode 只需要四个标量
（`src/compute/batch.py:188` 的 `build_decode`）：

| 需要什么 | 从哪来 | ZOS 下能不能算 |
|---|---|---|
| 位置 `pos` | `num_cached_tokens`（CPU 账本） | 能：`pos = num_cached + 上一批 token 数`（乐观 +1） |
| 线性槽位 `slot` | `block_table[pos // 256] * 256 + pos % 256` | 能：块在调度时就已预留 |
| 注意力长度 `csl` | `pos + 1` | 能 |
| 输入 token | 上一步采样出来的那个 id | **不能** → 留空 + 记行号 `remap`，GPU 端 gather |

只有**一个**未知量，而且它天然是"按行号索引"的，所以可以在 GPU 上补。这就是 ZOS 能成立的全部理由。

### 1.2 投机的前提：下一步的起点是一个区间

投机一步的样子（`src/control/scheduler.py:215` 起）：

```python
draft = self.spec_drafter.propose(seq.token_ids, self.spec_k)   # ① 起草：要对"最新 token"做后缀匹配
n_new = len(draft) + 1                                          # ② 按 K+1 预留 KV
...
j = greedy_accept_length(draft, seg[: len(draft)])               # ③ 验证：接受长度在 GPU 上算出来
emitted = [*draft[:j], int(seg[j])]
seq.num_cached_tokens += j + 1                                  # ④ 下一步的起点 = num_cached + j + 1
```

两个依赖，互相独立：

- **① 需要一个锚点**：草稿是"用最后 n 个 token 去历史里找后继"（`src/spec/ngram.py:49` 的 `propose` → `_match_once` 里 `key = tuple(history[-n:])`）。
- **④ 需要一个长度**：下一步的 `pos` / `slot` / `csl` 全都由 `num_cached_tokens` 推出来，而它只加了 `j+1`，`j` 是 0…K 之间的任意整数。

### 1.3 冲突的判定

ZOS 的 `schedule(t+1)` 发生在 `finish(t)` 之前（`src/control/core.py:202` `take_pending` 与 `:237` `swap_pending` 之间）。
也就是说调度第 t+1 步的那一刻：

- CPU 手上的历史截止到 **t−1**，t 的 token 还在 GPU 的显存里（`_sample_dev`）→ **锚点不存在**；
- CPU 更不知道 t 步有几个草稿被接受 → **j 不存在**。

这两条都不是"精度差一点"，下面分别说。

---

## 2. 冲突一：起草锚点不存在

### 2.1 具体差在哪

设第 t 步已经接受并产出了 token `x`，现在要排第 t+1 步：

```
ZOS 正常 decode：
  CPU 账本 token_ids = [… y]        ← 截止 t−1
  GPU 显存 _sample_dev = 最新采样的 x ← t 步的结果，CPU 还没读

  build_decode: pos = 账本 + 1、input_ids 留空、remap = 行号
  GPU: input_ids[i] = _sample_dev[remap[i]]     ← 只需要"第几行"，不需要它是谁
```

投机的起草器做的是**字符串匹配**，它需要这个 id 本身：

```
spec propose: key = tuple(history[-n:])   # 历史以 y 结尾
              → 找不到 "…y" 这种后缀，或者匹配到的是"y 后面跟 α"，而真实下一步是 "x 后面跟 β"
```

### 2.2 为什么"用旧锚点凑合"不是慢一点，而是全错

草稿错了会怎样？验证还是会给出正确的 token（验证是拿草稿去跟目标的 argmax 比，不接受就退回目标的 token）。
所以**不会算错**，但会：

- 接受长度 `j` 大概率掉到 0 → 这一步退化成普通 decode，还白跑了一列 K+1 的 query；
- 更要命的是**位置账**：这一步的 `n_new = K+1` 已经按"最坏情况"预留并写进 KV，而 `j=0` 意味着只推进 1 个位置，剩下 K 个位置要在下一步被覆盖（`src/control/scheduler.py:427` `_update_spec_batch` 的注释写了这件事：`其余 K-j 个 slot 留在同一页里，下一步的写入会自然覆盖它们`）。ZOS 下"下一步"是已经排好的、位置算死的一批 —— 覆盖关系对不上。

一句话：**投机的"多余草稿占位"依赖下一步能准确地从 `num_cached + j + 1` 开始重写；而 ZOS 恰恰把这一步的位置提前算死了。**

### 2.3 能不能"在途时不投机、只跑普通 decode"？

能，而且很自然：`scheduler.py:193` 现在就有一行 `seq.spec_draft_ids = []`，注释是"ZOS 与投机互斥，这里清掉草稿只是把不变式写死"。

但这样等于**每个在途步都是普通 decode**，而 ZOS 下每个 decode 步都在途 —— 于是投机永远不会生效。
所以这条路等于"A 互斥"，只是换了个说法。

---

## 3. 冲突二：`j` 决定位置，而位置不是"槽位"那种可以多预留的东西

### 3.1 先把"最坏预留"能救什么说清楚

有一类东西确实可以按最坏情况预留：

- **KV 块**：按 K+1 个 token 预留块（`may_append_n(seq, n_new)`），反正块是 256 一格，多留一格不花钱；
- **采样缓冲**：按定宽 `K+1` 准备（`_spec_width`，`src/control/scheduler.py:401`）。

这类"容量"问题，最坏预留就够了。

### 3.2 但 position 是"值"，不是"容量"

下一步要写进 kernel 的量是：

```
positions   = pos              # 这个位置的绝对下标，attention 的 RoPE 用它
cache_seqlens = pos + 1        # 决定这一行要读多少历史 KV
slot_mapping  = f(pos)         # 由 pos 是否跨块边界决定，跨了就换页
```

`pos = num_cached + j + 1`。`j ∈ {0,…,K}`，K=4 时就是 5 个可能值。
你不能"预留 5 个位置" —— 位置是标量，一批里每条序列只有一个。

### 3.3 "先按最坏写，错了再改"为什么也不行

因为**写下去的就是数据本身**：

1. 第 t+1 步的 attention 会用 `csl = pos + 1` 去读历史。如果 `pos` 比真实值大，就会读到还没写入的 KV 槽（内容是被覆盖掉的旧页），**算出来的 logits 直接错**，不是"稍后修正"就能救的；
2. KV 写入是按 `slot_mapping` 落的。位置错 → 落错页 → 后面每一步都在错的页上继续写。

所以这不是"精度问题"，是**正确性问题**。这也是为什么"投机 + ZOS 同时开"不像"投机 + chunked prefill"那样可以凑合。

---

## 4. 附带的第三层：投机的"无损"本身有条件

即使把坐标问题解决（见 §5 的路 B），还有一层更软但更麻烦的东西：**投机的"逐 token 与基线一致"依赖批量不变性**。

投机验证批是 `[last_token, *draft]`，形状是 `bs × (K+1)`；普通 decode 是 `bs × 1`。同一条序列在不同批形状下，logits 可能有极小差异，argmax 在**并列**时就会翻。

本仓库实测（`bench/batch_invariance.py`，8 条 prompt × 32 token，同批 `max_num_seqs=8` vs 串行 `=1`）：

```
[DIFF] prompt5 首次分叉位置=6
       bs8=[293, 271, 1350, 955, 579, 7, 16, 15]
       bs1=[293, 271, 1350, 955, 579, 7, 20, 593]
[batch-invariance] prompts=8 tokens/row=32 total=256 分叉行=1
[margin] steps=297 min=0.000e+00 p1=0.000e+00 p50=3.250e+00
```

`min margin = 0.000e+00`：**真的存在 top1==top2 的精确并列**，不是"接近"。
2×4090D 上也有同类证据（[07-实测汇总-4090D](07-实测汇总-4090D.md) §9.4）：验证路径 top-1=13 与 top-2=11 的 logit 同为 `+20.250`，baseline 只差 0.125，而 bf16 的 ULP 是 0.0625 —— 差两个 ULP 以内的并列，argmax 按实现取小下标就翻。

结论：**"投机是无损的"必须加限定词** —— 在"同一批组成、同一份 kernel、同一条算术顺序"下无损；跨批形状不承诺。这一点跟 ZOS 无关，但会连着 ZOS 一起被问，所以放在这里。

---

## 5. 三条出路

### A. 互斥（当前实现）

```python
# src/control/core.py:92
want_zos = os.environ.get("KRITACE_ZOS", "1").strip().lower() not in ("0", "false", "no")
self.zos = want_zos and self.config.num_speculative_tokens == 0
if want_zos and not self.zos:
    print("[zos] 投机解码开启（num_speculative_tokens>0），ZOS 自动关闭："
          "接受长度 j 在 GPU 上产生，CPU 侧调度无法提前一步拿到。")
```

- 成本：两者不能同时拿收益；
- 收益：两条路径各自都是**被验证过的**（ZOS：三组对拍 sha 全等；投机：K 曲线 + 与 K=0 逐 token 一致）；
- 参数为 0 时 ZOS 照常工作，行为与不写投机时完全一致。

### B. 把坐标修正搬上 GPU（DeepInfer 走的路）

DeepInfer 的 nextn 投机能开 ZOS，是因为它把"未知"全放在显存里，CPU 一个都不问：

| 它有什么 | 对应本仓库缺什么 |
|---|---|
| 草稿由 **MTP 头在 GPU 上产**（`nextn` 层），CPU 不需要锚点 | 本仓库是 n-gram 在 CPU 做字符串匹配 |
| `num_accepted_tokens_gpu` 留在显存，`compute_seqlens_k_zos` 在 GPU 上改写 `seqlens_k` | 本仓库 `j` 只在 `_update_spec_batch` 里以 Python int 出现 |
| `compute_seqlens_q_by_confidence_rates` 在 GPU 上算 varlen 的 `seqlens_q` | 本仓库 `csl` 在 CPU 上按 `pos + 1` 写死 |
| placeholder 步保证 CUDA Graph 形状不断 | 本仓库空批直接 return（`core.py:202`） |

对应到本仓库，要做的是五件事：

1. **device-side n-gram**：GPU 上的后缀匹配（哈希表 + 代数标记，或按序列的环形缓冲 + 窗口比较）。这是最难的一块：`propose` 现在是 O(len) 的 Python 线性扫描，搬到 GPU 要么 O(N·n) 的暴力、要么维护一张会串味的表；
2. **accept 长度在 GPU 上算**：`j` 本来就是"草稿与 argmax 逐位相等的长度"，`(draft_dev == seg_dev).cumprod().sum()` 就够，不需要 D2H；
3. **位置在 GPU 上修正**：`pos = cached_dev + accepted_dev + 1`，`csl`/`slot`/`positions` 三个 kernel 参数都要改成"读显存里的标量"；
4. **CUDA Graph 形状**：ZOS + 投机时批形状在 `bs×1` 与 `bs×(K+1)` 之间跳，要么多捕获几档，要么走 placeholder 步；
5. **不变量重写**：在途行现在假设"只跑 1 个 token"（`scheduler.py:193` 那句清草稿），要换成"在途行跑 K+1 个 token，位置待定"。

工作量估计：**2–3 周**（其中 GPU n-gram 占一半），且做完之后还要重新推一遍 ZOS 的三条不变式（在途不可抢占、块推迟归还、采样双缓冲）在"长度可变"下的版本。

### C. 交替（每一步选一种）

听上去像折中，实际会退化成 A：ZOS 的前提是"每步都在途"，一旦某一步同步收尾，流水线就断了，下一步又得等 —— 省下来的那 0.2 ms 立刻还回去。

---

## 6. 为什么最后没做 B：先把收益上限算清楚

ZOS 的收益上限是 `1 + B/G`（B = CPU 侧一步的调度成本，G = GPU 侧一步的计算时间）：

| 机器 | B | G | 上限 |
|---|---|---|---|
| RTX 3050 4 GB / 0.6B（本机实测） | 0.23–0.33 ms | ~130 ms | **1.002×** |
| 2×4090D / 0.6B（v4 实测 449.7 tok/s ⇒ 2.2 ms/步） | 0.1–0.2 ms（估算） | 2.2 ms | **1.05–1.09×** |
| 8B on 4090D | 同上 | ~16 ms | 1.01× |

而投机的收益是 `(1+αK)/(1+c)`，实测：重复文本 **5.87×**（4090D）、散文 1.75×、代码 1.16×。

也就是说：**ZOS 的上限（≤1.15，多数情况 <1.05）比投机的下限（1.16）还低**。
为了把 ZOS 从 "0.2% 提升" 变成 "和投机叠加"，要付 2–3 周 + 重推三条不变式 + GPU 侧新 kernel，收益端的增量仍然被 `B/G` 卡死。**这笔投资不成立。**

这不是"做不出来"，是"做出来也不值得"。判断标准写在 [02-ZOS §何时不值得做](02-ZOS.md) 里：先量 `B`，`B/G < 5%` 就不要做。

---

## 7. 现在的实现（事实清单）

| 位置 | 是什么 |
|---|---|
| `src/control/core.py:92` | `KRITACE_ZOS` 开关 |
| `src/control/core.py:93` | `self.zos = want_zos and config.num_speculative_tokens == 0` |
| `src/control/core.py:94-98` | 开启投机时打印一行原因（不静默） |
| `src/control/scheduler.py:193` | 在途分支强制 `seq.spec_draft_ids = []`，把"在途行只跑 1 token"写死 |
| `src/control/scheduler.py:215-234` | 普通分支：先 `propose`，再按 `K+1` 预留 |
| `src/control/scheduler.py:140-145` | 有草稿的序列必须走 prefill 通道（只有 varlen attention 能一次算 K+1 个 query） |
| `src/control/scheduler.py:401-425` | `_spec_width` / `prefill_samples`：注意后者返回的是**元素数**不是请求数 |
| `src/control/scheduler.py:427-471` | `_update_spec_batch`：`j` → `num_cached += j+1` → `sync_prefix(num_new_tokens=j+1)` |
| `src/data/batch.py:116` | 投机行的 token 是 `[last_token, *spec_draft_ids]`（长度 K+1） |
| `src/data/batch.py:91-104` | ZOS 在途行：`token_ids=[]`、`num_cached=zos_start_pos`、`remap=行号` |

行为表：

| `KRITACE_ZOS` | `--num-speculative-tokens` | 实际行为 |
|---|---|---|
| 1（默认） | 0 | 零开销调度生效 |
| 1 | >0 | 打印一行，ZOS 关闭，投机生效 |
| 0 | 任意 | 同步调度（投机照常生效） |

---

## 8. 如果哪天真要做，动手顺序

1. **先把 `j` 变成显存档位**（半天）：验证后不 `.tolist()`，`j_dev = (draft_dev == seg_dev).cumprod(dim=1).sum(dim=1)`；`read_samples` 只读 token，不读 `j`；
2. **让 `build_decode` 的三个标量能来自显存**（2–3 天）：`pos/csl/slot` 改成"CPU 给一张标量表 or GPU 现算"，先只支持"在途且无草稿"与"在途且有草稿"两种形状；
3. **写 device-side n-gram**（1–1.5 周）：先用"每序列环形缓冲 + 暴力窗口比较"验证正确性（N=4096、n≤4、8 条序列，成本可接受），再考虑哈希表；
4. **补 placeholder 步**（2–3 天）：空批不再直接 `return`，发一个占位批保证 graph 形状与流水不断（DeepInfer 的做法）；
5. **重推不变式**（2 天）：在途不可抢占 → 在途 + 可变长度；块推迟归还 → 覆盖式写入；双缓冲 → 采样缓冲要能装下 K+1 个 id。

验收口径（与现在一致，别放松）：**同批组成下，ZOS=0 与 ZOS=1 逐 token 一致**，三组 `bench/final.sh` 的 sha 全等。

---

## 9. 面试问答

**Q1：为什么这两个不能同时开？**
A：ZOS 排下一步时只知道"上一批跑了几行"，不知道"上一批产出的是什么"；投机恰恰要拿"上一批产出的那个 token"去起草，而且下一步的起点还依赖验证接受长度 `j`。前者的未知量是一个 id（可以 GPU 侧按行号补），后者的未知量是一个长度（位置是标量，补不了）。所以是前提互斥，不是实现没写完。

**Q2：最坏情况预留不行吗？**
A：容量可以预留（KV 块、采样缓冲都按 K+1 预留了），但 `positions` / `cache_seqlens` 是标量值，不是容量。`csl` 错 → attention 读错历史长度 → logits 直接错，不是"稍后修正"能救的。

**Q3：那像 DeepInfer 那样把修正搬上 GPU 行不行？**
A：行，但那是另一套架构：草稿由 MTP 头在 GPU 上产（不需要 CPU 锚点），`num_accepted_tokens` 留在显存，`compute_seqlens_k_zos` 在 GPU 上改写 `seqlens_k`，还有 placeholder 步保证 CUDA Graph 形状。本仓库是 n-gram 在 CPU 做字符串匹配，要补的是 device-side n-gram + 显存坐标修正 + placeholder + 三条不变式的可变长度版本，估 2–3 周。

**Q4：做出来能快多少？**
A：ZOS 的天花板是 `1 + B/G`，本机 `B/G ≈ 0.2%`，4090D 上按 449.7 tok/s（2.2 ms/步）估 `5–9%`；而投机单独就有 1.16×（代码）到 5.87×（重复文本）。叠加的增量被 `B/G` 卡住，所以我没有做。

**Q5：那你的 ZOS 是不是白做了？**
A：机制层面不是 —— `KRITACE_ZOS_STATS=1` 下 `read_samples` 的"无需等待"比例从 3.1% 涨到 100%，累计等待从 56.8 ms 掉到 0.1 ms，说明气泡确实被填掉了。只是这台 4 GB 卡上 GPU 一步要 130 ms，填掉 0.3 ms 看不出来。要看出来需要 `G` 小、`B/G` 大的机器。

**Q6：投机真的无损吗？**
A：在"同一批组成、同一份 kernel、同一条算术顺序"下，与 K=0 逐 token 一致（我跑过 K=0/3/5/8 三组文本）。但跨批形状不承诺：8 条 prompt 里就有 1 条在第 6 个 token 分叉，而且实测到 `top1−top2 = 0` 的精确并列。无损是"给定批形状的无损"。

---

## 10. 代码地图

| 想看的 | 去哪 |
|---|---|
| 互斥开关与提示 | `src/control/core.py:92-98` |
| 在途行强制清草稿 | `src/control/scheduler.py:184-206` |
| 起草 + 按 K+1 预留 | `src/control/scheduler.py:210-240` |
| 投机批走 prefill 通道 | `src/control/scheduler.py:134-145` |
| 接受长度回写 | `src/control/scheduler.py:443-471` |
| 定宽与元素数 | `src/control/scheduler.py:401-425` |
| n-gram 起草器 | `src/spec/ngram.py:49`（`propose`）/ `:104`（`_last_occurrence`） |
| 接受的判定 | `src/spec/verify.py:91`（`greedy_accept_length`） |
| 投机行怎么组包 | `src/data/batch.py:137-159` |
| zos 在途行怎么组包 | `src/data/batch.py:91-104` |
