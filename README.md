# kritace_vllm

从零实现的 LLM 推理引擎（vLLM 精简复刻），约 1.3 万行，覆盖 PagedAttention、连续批处理、前缀缓存、抢占、CUDA Graph、torch.compile、MoE、TP/PP/DP/EP、投机解码，以及自研的**零开销调度（ZOS）**。

| 主题 | 入口 | 文档 |
|---|---|---|
| 分页 KV + 前缀缓存 | `src/data/kv.py`、`src/data/prefix/` | [01](docs/01-架构总览.md) |
| 连续批处理 / chunked prefill / 抢占 | `src/control/scheduler.py` | [01](docs/01-架构总览.md) |
| CUDA Graph / torch.compile | `src/compute/cuda_graph.py`、`src/compute/model_runner.py` | [01](docs/01-架构总览.md) |
| MoE / TP / PP / DP / EP | `src/moe/`、`src/distributed/` | [04](docs/04-MoE与并行.md) |
| 投机解码（n-gram / 小模型） | `src/spec/` | [03](docs/03-投机解码.md) |
| **零开销调度 ZOS** | `src/control/core.py:173` | **[02](docs/02-ZOS.md)** |

---

## 1. 安装

**环境要求**：Linux、Python 3.10–3.12、NVIDIA GPU（sm_80 以上，4 GB 显存可跑 0.6B）、CUDA 12.x。

```bash
# ① torch：按你的 CUDA 版本装，不要用 pip 默认的 CPU 版
pip install torch --index-url https://download.pytorch.org/whl/cu124

# ② 其余依赖
pip install -r requirements.txt

# ③ flash-attn（强烈建议装；不装也能跑，但慢很多且 CUDA Graph 不可用）
pip install flash-attn --no-build-isolation
```

**实测环境**（本项目所有数字都来自这里）：torch 2.13.0+cu130、transformers 5.17.0、triton 3.7.1、flashinfer-python 0.6.18、flash-attn **未安装**、RTX 3050 Laptop 4 GB。

**模型**：默认从 `~/huggingface/<模型名>` 找（可用 `HF_HOME` 改）。

```bash
python run.py --list                    # 缓存目录 / 显存 / 注册表里的模型 / 哪些能装下
python run.py --download --model Qwen3-0.6B     # 下载到 ~/huggingface/Qwen3-0.6B
python run.py --model /abs/path/to/model        # 也可以直接给本地目录
```

---

## 2. 快速开始

```bash
python run.py --model Qwen3-0.6B --prompt "Hello, my name is" --max-new-tokens 64
```

输出：

```
Loading /home/Kritace/huggingface/Qwen3-0.6B ...
KV pool: 4 blocks × 256 tok (0.11 GiB)
Prompt: Hello, my name is
Hello, my name is Xia, and I'm 17 years old. I am very into the game ...
Warmup: 3.83s  (compile + CUDA Graph capture, excluded)
TTFT:   229.5 ms  (steady prefill + first token)
Decode: 23 tokens in 1.110s (20.7 tok/s)
E2E:    24 tokens in 1.339s (17.9 tok/s)
```

逐行读：

- `KV pool: 4 blocks × 256 tok (0.11 GiB)` —— 按 `gpu_memory_utilization` 自动算出的 KV 池。**太小说明显存被占满了**，先把 yaml 里的比例降到 0.6 或手动给 `num_kvcache_blocks`。
- `Warmup` —— 三次等长 dummy 的预热（跑完整调度循环），CUDA Graph 录图 / torch.compile 编译也在这里；开 `--cg --compile` 时会有几十秒，**不计入下面的计时**。不预热的话第一段 prefill 的首次触达成本会算进 TTFT（实测 TTFT 1530 ms → 229 ms）。
- `TTFT` —— 首 token 延迟（prefill 全流程 + 采样）。
- `Decode` —— 稳态解码吞吐（单请求）。
- `E2E` —— 含 prefill 的总时间。

批量演示（4 条并发）：

```bash
python run.py --model Qwen3-0.6B --batch
```

---

## 3. 开关总表

| 想开什么 | 命令行 | config.yaml | 环境变量 | 默认 |
|---|---|---|---|---|
| **CUDA Graph**（decode） | `--cg` / `--eager` | `enforce_eager: false` | — | **关** |
| **torch.compile**（prefill） | `--compile` / `--no-compile` | `torch_compile: true` | — | **关** |
| compile 档位 | `--compile-mode default\|reduce-overhead\|max-autotune` | `compile_mode` | — | `reduce-overhead` |
| 符号形状 | `--compile-dynamic` / `--no-compile-dynamic` | `compile_dynamic` | — | `true` |
| **投机：n-gram** | `--spec-draft ngram --num-speculative-tokens 4` | `spec_draft: ngram` | — | **关** |
| 投机：小模型 | `--spec-draft /path/to/draft` | `spec_draft: <路径>` | — | **关** |
| n-gram 匹配长度 | — | `spec_ngram_max_n: 4` | — | 4 |
| **ZOS** | — | — | `KRITACE_ZOS=1/0` | **开** |
| ZOS 等待统计 | — | — | `KRITACE_ZOS_STATS=1` | 关 |
| ZOS GPU 忙时探针 | — | — | `KRITACE_ZOS_GPU=1` | 关 |
| NVTX 区间（nsys） | — | — | `KRITACE_NVTX=1` | 关 |
| **前缀缓存** | `--prefix-backend none\|hash\|radix` | `prefix_backend` | — | `hash` |
| 混排 prefill/decode | — | `mix_prefill_decode` | — | `true` |
| token 预算（定 chunk 大小） | — | `max_num_batched_tokens` | — | 8192 |
| 并发序列上限 | `--max-num-seqs` | `max_num_seqs` | — | 256 |
| KV 块数（0=自动） | — | `num_kvcache_blocks` | — | 0 |
| 显存占用比例 | — | `gpu_memory_utilization` | — | 0.85 |
| 上下文长度 | — | `context_len` | — | 4096 |
| **TP / PP / DP** | `--tp/--pp/--dp` | `tp_size/pp_size/dp_size` | — | 1/1/1 |
| EP（MoE 专家并行） | `--ep` | `enable_ep` | — | `false` |
| MoE 后端 | `--moe-backend auto\|cutlass\|triton` | `moe_backend` | `KRITACE_MOE_CUDA=0` | `auto` |
| PP 不均分 | `--pp-layer-counts 12,28` | `pp_layer_counts` | — | 等分 |
| GPU 映射 | `--device-ids 1,0` | `device_ids` | — | 自动 |
| 采样参数 | `--temperature/--top-p/--top-k` | `temperature/top_p/top_k` | — | 0.9/0.95/20 |
| 投机调试 | — | — | `KRITACE_SPEC_DEBUG=1` | 关 |

> 命令行只覆盖 config.yaml 里同名字段；标 "—" 的只能改 yaml 或环境变量。

---

## 4. 每个开关怎么用

### 4.1 CUDA Graph（decode 加速，`--cg`）

**做什么**：decode 每步要发几百个 kernel，Python 逐个 launch 的开销比 GPU 算的还大。CUDA Graph 把整步录成一个图，之后一次 `replay()` 提交。

```bash
python run.py --model Qwen3-0.6B --cg --prompt "hi" --max-new-tokens 64
```

- 在 `ModelRunner.__init__` 里就录完（`src/compute/model_runner.py:213`），按 batch 档位（`CUDAGraphRunner.make_bs_list`）各录一张；实际 batch 落在两档之间时用 `_pad_bs` 向上取档，padding 行的 `slot_mapping=-1`、`cache_seqlens=0`、`block_tables=-1`，保证不写脏数据。
- 多卡时录图前后各加一次 barrier，避免先到的 rank 卡在 warmup 的集合通信里。
- PP 时跨 stage 的 send/recv 在**图外**做，只有层内计算进图。
- **前提**：必须装真 flash-attn。回退实现（`_flash_attn_fallback.py`）里有 `.tolist()` 的 host 同步，图捕获会报 `Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned`。
- 代价：启动慢（录图几十秒）、显存多留一份 workspace。
- 关掉：`--eager`。

### 4.2 torch.compile（prefill 加速，`--compile`）

**做什么**：把 prefill 的整段前向编译成一个 fused 图，消除逐层 Python 开销、融合 pointwise。

```bash
python run.py --model Qwen3-0.6B --compile --cg --prompt "写一段快排"
```

- 只作用于 **prefill 且没有前缀命中** 的那一段（`src/compute/model_runner.py:227`），因为这段 Q 与 K 等长；decode 走 CUDA Graph，两者互补。
- `--compile-mode`：`default`（编译快、优化少）/ `reduce-overhead`（默认）/ `max-autotune`（最慢最好）。**含 TP 时自动改成 `max-autotune-no-cudagraphs`**（compile 自带的 cudagraph 会和引擎自己的 CUDA Graph 打架）。
- `--compile-dynamic`（默认开）：符号形状，一次编译覆盖任意长度，在线服务用这个；`--no-compile-dynamic` 是固定形状重放，新长度会重编译，只适合离线定长。
- 代价：首 chunk 编译 10–60 s（在 warmup 里，不计时）。

### 4.3 投机解码

**n-gram（零额外模型，先跑这个）**：

```bash
python run.py --model Qwen3-0.6B --spec-draft ngram --num-speculative-tokens 4 \
              --prompt "def fib(n): def fib(n):" --max-new-tokens 64
```

- 两个参数都要给：`--spec-draft ngram` 选草稿来源，`--num-speculative-tokens K` 给每步草稿长度（缺 K 等于不启用）。
- n-gram 从序列自身历史里找可复用后缀（`src/spec/ngram.py`），对摘要 / 代码 / few-shot 这类含重复片段的输入有效，对自然文本基本无效。匹配窗口 `spec_ngram_max_n`（默认 4）只能改 yaml。

**小模型草稿**：

```bash
python run.py --model Qwen3-8B --spec-draft ~/huggingface/Qwen3-0.6B \
              --num-speculative-tokens 5
```

- 两个模型必须**同词表**，否则 token id 对不上、接受率恒为 0（不报错，只是白算）。
- draft 参数量要远小于 target，否则"用便宜计算换昂贵访存"的前提不成立。

**行为细节**：带草稿的序列走 prefill 通道（只有 varlen attention 能一次算 K+1 个 query），因此它们**不与普通 decode 混排**（`src/control/scheduler.py:140`）。

**实测**（Qwen3-0.6B / 单卡 / `enforce_eager` / n-gram / 单请求 64 token 贪心 / 3 次均值）：重复文本 K=2 约 1.3×，K=4 约 1.4×；自然文本接近 1.0×。

**限制**：`max_tokens` 可能多 1 个 token（验证批一次回写 `j+1` 个，边界那步越界一格）；**与 ZOS 互斥**（见 4.4）。

### 4.4 ZOS（零开销调度，默认已开）

**做什么**：把"等采样结果"推迟一拍，让 CPU 的调度落在 GPU 算前向的窗口里，GPU 不再空转等 CPU。

```bash
# 默认就是开的；关掉做对照：
KRITACE_ZOS=0 python run.py --model Qwen3-0.6B --prompt "hi" --max-new-tokens 64

# 看"等"是否真的消失（read_samples 的 ready 比例 / 累计等待）
KRITACE_ZOS_STATS=1 KRITACE_ZOS=0 python bench/zos_matrix.py --case short --n 8 --max-tokens 32 --blocks 48
KRITACE_ZOS_STATS=1 KRITACE_ZOS=1 python bench/zos_matrix.py --case short --n 8 --max-tokens 32 --blocks 48

# 量气泡：步墙钟 − GPU 忙时（两模式交错跑 3 轮）
KRITACE_ZOS_GPU=1 python bench/repro_zos.py --rounds 3

# 正确性：两模式逐 token 必须一致
bash bench/final.sh /tmp/final.txt
```

- 单进程、TP、PP 都支持（`(n_p, n_d, slot)` 会透传，PP 下采样 rank 广播 `sample_dev`）。
- **与投机互斥**：接受长度 `j` 在 GPU 上产生，CPU 侧无法提前一步拿到。开投机时会打印一行提示并自动退回同步循环。
- 收益判据：`1 + B/G`。`B`（每步气泡）实测 0.23 ms（bs=8）～0.33 ms（bs=32），与并发无关；`G` 是 GPU 真实算时。本机 `G≈130 ms`（CPU launch-bound）→ 收益 0.2%，量不出来；真 flash-attn + CUDA Graph 的机器上 `G` 降到 1.5–10 ms → 收益 2%–15%。

### 4.5 前缀缓存（`--prefix-backend`）

```bash
python run.py --model Qwen3-0.6B --prefix-backend hash --batch
python run.py --model Qwen3-0.6B --prefix-backend radix --batch
```

- `none`：不做。`hash`（默认）：块级 xxhash 链式索引，命中即复用 KV 块，零拷贝。`radix`：基数树，同样的块级粒度，但支持更细的共享管理。
- 粒度是 `kvcache_block_size`（默认 256），比块短的公共前缀不会被复用。
- 命中会**跳过 prefill 计算**，所以配合 `--compile` 时注意：编译路径只覆盖"无命中"的那段。

### 4.6 连续批处理与 chunked prefill

```yaml
mix_prefill_decode: true      # 同一步里 decode 与 prefill 混排（默认开）
max_num_batched_tokens: 8192  # 一步的 token 预算：决定 prefill chunk 切多大
max_num_seqs: 256             # 同时跑的序列数上限
```

- `mix_prefill_decode: false` 会变成 prefill 优先：来一个 prefill 就把 decode 挤到后面，TTFT 好、ITL 差。
- 长 prompt 被 `max_num_batched_tokens` 切成多块（chunked prefill），每块一次前向；调小显存更友好、调大吞吐更高。
- 显存不够时的兜底是**抢占**：牺牲队尾序列、回到 waiting 队头重算（`scheduler.preempt`）。

### 4.7 KV 池与显存

```yaml
num_kvcache_blocks: 0        # 0 = 按 gpu_memory_utilization 自动算
gpu_memory_utilization: 0.85 # 越大留的 KV 越多，越小越不容易 OOM
kvcache_block_size: 256      # FA2 paged 要求 256 的倍数，别改小
context_len: 4096            # 单序列最长上下文
```

4 GB 卡上 0.6B 的实测：`--blocks 32` 能跑（0.88 GiB KV），`--blocks 512` 直接 OOM。**小显存先降 `gpu_memory_utilization` 到 0.6–0.7，再手动给 `num_kvcache_blocks`**（bench 脚本用 `--blocks`）。

### 4.8 并行（多卡必须走 torchrun）

单进程只能看到一张卡；`--tp/--pp` 在单进程里只是配置，真正跑多卡要 `torchrun`（`run_torchrun.py` 是入口）：

```bash
# TP=2（切 head / 切权重，机内 NVLink 或 PCIe）
torchrun --nproc_per_node 2 run_torchrun.py --model Qwen3-0.6B --tp 2 --prompt "hi"

# PP=2（切层，跨机低带宽更划算）
torchrun --nproc_per_node 2 run_torchrun.py --model Qwen3-0.6B --pp 2

# TP=2 × PP=2（4 卡），PP 不均分（异卡显存）
torchrun --nproc_per_node 4 run_torchrun.py --model Qwen3-0.6B \
         --tp 2 --pp 2 --pp-layer-counts 12,16 --device-ids 0,1,2,3

# MoE + 专家并行（EP 在现有 TP ranks 上切 expert，不扩大 world）
torchrun --nproc_per_node 2 run_torchrun.py --model Qwen1.5-MoE-A2.7B --tp 2 --ep

# 约束：不开 EP 时 moe_intermediate_size % tp_size == 0；开 EP 时 num_experts % tp_size == 0
```

- **DP**（多副本吞吐）：`python run.py --dp 2 --device-ids 0,1` —— 起 `dp_size` 个独立引擎副本（`MPClient` + ZMQ），每个副本占用 `tp×pp` 张卡，副本间不通信。适合"吞吐优先、请求互不相关"。
- rendezvous：单机 spawn 用 `nccl_port`；多机用 `master_addr` / `master_port` / `nccl_ifname`（IB 网卡名）。
- 采样固定在最后一个 PP stage 的 tp0，结果再回传 driver；PP 下 driver 的阻塞等待是已知吞吐上限（见 `src/control/executor.py` 文件头）。
- 4090D 无 NVLink 且 P2P 被禁，TP all-reduce 走主机内存，带宽折损明显。

### 4.9 采样参数

```bash
python run.py --model Qwen3-0.6B --temperature 0 --top-p 0.9 --top-k 20
```

- `--temperature 0` = 贪心（等价 vLLM 的 `greedy`）。
- `min_p` 目前没有命令行，只能在代码里传：`SamplingParams(temperature=0.7, min_p=0.05, max_tokens=128)`。
- 整批都是贪心时会跳过词表投影直接 `argmax(hidden)`，采样开销接近 0。

---

## 5. 常见组合

| 目标 | 命令 |
|---|---|
| 先跑通 | `python run.py --model Qwen3-0.6B` |
| 单卡最快 | `python run.py --model Qwen3-0.6B --cg --compile`（需真 flash-attn，否则 `--cg` 报 §4.1 那个错） |
| 显存紧张 | `python run.py --model Qwen3-0.6B --cg` + yaml 里 `gpu_memory_utilization: 0.65`、`num_kvcache_blocks: 32` |
| 服务化、重复 prompt | `python run.py --prefix-backend hash --cg --compile --batch` |
| 重复文本加速 | `--spec-draft ngram --num-speculative-tokens 4` |
| 验证 ZOS 效果 | `KRITACE_ZOS_STATS=1` + `bench/zos_matrix.py` 两模式对跑 |
| 双卡吞吐 | `torchrun --nproc_per_node 2 run_torchrun.py --tp 2` |
| 双卡、两路独立流量 | `python run.py --dp 2 --device-ids 0,1` |

---

## 6. config.yaml 全字段

| 字段 | 默认 | 说明 |
|---|---|---|
| `model` | `Qwen3-0.6B` | 模型名（查 `~/huggingface/`）或绝对路径 |
| `prompt` / `max_new_tokens` | `"Hello, my name is"` / 200 | 单条推理的输入与长度 |
| `temperature` / `top_p` / `top_k` | 0.9 / 0.95 / 20 | 采样参数 |
| `enforce_eager` | `true` | `false` = 开 decode CUDA Graph |
| `torch_compile` | `false` | `true` = prefill 走 torch.compile |
| `compile_mode` | `reduce-overhead` | 含 TP 时自动改 `max-autotune-no-cudagraphs` |
| `compile_dynamic` | `true` | 符号形状（一次编译覆盖任意长度） |
| `dtype` | `auto` | 可填 `float16` / `bfloat16` |
| `context_len` | 4096 | 单序列最长上下文 |
| `max_num_seqs` | 256 | 并发序列上限 |
| `max_num_batched_tokens` | 8192 | 一步 token 预算（chunk 大小） |
| `mix_prefill_decode` | `true` | prefill/decode 混排 |
| `kvcache_block_size` | 256 | 分页块大小，FA2 paged 要求 256 的倍数 |
| `prefix_backend` | `hash` | `none` / `hash` / `radix` |
| `gpu_memory_utilization` | 0.85 | KV 池显存占比 |
| `num_kvcache_blocks` | 0 | 0 = 自动 |
| `tp_size` / `pp_size` / `dp_size` | 1/1/1 | 张量 / 流水 / 数据并行度 |
| `pp_layer_counts` | `null` | 不均分 PP 每 stage 层数 |
| `device_ids` | `null` | 每进程 GPU 映射，长度应为 `dp×tp×pp` |
| `nccl_port` / `master_addr` / `master_port` / `nccl_ifname` | 29500 / 127.0.0.1 / 29500 / `""` | 分布式 rendezvous |
| `enable_ep` | `false` | MoE 专家并行 |
| `moe_backend` | `auto` | `auto` / `cutlass` / `triton` |
| `spec_draft` | `""` | `ngram` 或草稿模型路径；空 = 关 |
| `num_speculative_tokens` | 0 | 每步草稿长度 K；0 = 关 |
| `spec_ngram_max_n` | 4 | n-gram 最大匹配长度 |

---

## 7. 环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `KRITACE_ZOS` | `1` | ZOS 开关 |
| `KRITACE_ZOS_STATS` | `0` | 打印 `read_samples` 等待统计（ready 比例 / 累计等待 / 平均） |
| `KRITACE_ZOS_GPU` | `0` | cudaEvent 量每步 GPU 忙时（配合步墙钟算气泡） |
| `KRITACE_NVTX` | `0` | 给 `step()` 四段打 NVTX 区间，nsys 用 |
| `KRITACE_SPEC_DEBUG` | `0` | 投机：逐序列打印草稿 / 验证 / 接受长度 |
| `KRITACE_SPEC_LOGITS` | `0` | 投机：打印验证批 logits 对齐信息 |
| `KRITACE_MOE_FORCE_EAGER` | `0` | MoE 强制 eager（排查 CUDA Graph 问题） |
| `KRITACE_MOE_CUDA` | `1` | `0` = 关掉 CUDA 扩展，退回 Triton 后端做对照 |
| `KRITACE_SYS_PROF` | `0` | 全链路分段计时（定位瓶颈用，不作为性能数据） |

---

## 8. 观测与调试

```bash
# ① "等"消失了没
KRITACE_ZOS_STATS=1 python run.py --model Qwen3-0.6B --max-new-tokens 128
#   ZOS=0: calls=32 ready=1 (3.1%) total_wait=56.8ms avg=1.78ms
#   ZOS=1: calls=32 ready=32 (100.0%) total_wait=0.1ms avg=0.005ms

# ② 时间线（需要 nsys；WSLg 里 nsys-ui 能直接弹窗）
KRITACE_NVTX=1 nsys profile -t cuda,nvtx -o /tmp/zos --force-overwrite true \
    python bench/zos_matrix.py --case short --n 8 --max-tokens 32 --blocks 48
nsys stats --report nvtx_sum /tmp/zos.nsys-rep | grep zos
#   同步版 zos.read 中位 1728 µs；ZOS 版 36 µs

# ③ 步边界 GPU 空转（从 nsys sqlite 算）
python bench/gap_step.py /tmp/zos.sqlite
```

四个 NVTX 区间对应的正是 CPU 一步的四件事：`zos.schedule`（调度）、`zos.launch`（组包+发射）、`zos.read`（取结果，唯一阻塞点）、`zos.finish`（认账）。

---

## 9. 测试与基准

```bash
python -m pytest tests/unit -q                    # 33 passed, 1 skipped，不需要 GPU
python -m pytest tests/gpu -q                     # 需要模型权重；没装模型会自动 skip
torchrun --nproc_per_node 2 -m pytest tests/distributed -q

bash bench/final.sh /tmp/final.txt                # ZOS 0/1 逐 token 对拍（三组场景）
python bench/repro_zos.py --rounds 3              # 气泡 / 等待 / 正确性汇总
python bench/cpu_bubble.py                        # 分相计时：B 的量级
python bench/zos_matrix.py --case short --n 8 --max-tokens 32 --blocks 48
python bench/spec_sweep.py --k 1,2,4              # 投机加速比扫描
```

每个数字的口径、噪声处理、为什么本机量不出加速，见 [docs/05-基准与复现](docs/05-基准与复现.md)。

---

## 10. 目录结构

```
src/
  config.py            EngineConfig：所有可调参数与校验
  control/             控制面（不碰张量）
    core.py             唯一的推理循环（schedule → run → read → finish）
    scheduler.py        准入、抢占、KV 记账、ZOS 在途账本
    executor.py         Uniproc / Multiproc（TP×PP），采样 rank 回传
    engine.py client.py mp_client.py    门面与进程内/跨进程客户端
  data/                Sequence、块表、前缀缓存、跨进程 batch
  compute/             worker、model_runner、组包、算子、采样、CUDA Graph
  moe/                 路由、专家、fused 后端、CUDA/Triton kernel
  spec/                n-gram / 模型草稿、验证、接受判定
  distributed/         TP/PP/EP 通信组与控制通道
bench/  tests/  utils/  docs/
```

---

## 11. 已知限制

- **CUDA Graph 需要真 flash-attn**；回退实现里有 host 同步，图捕获未通过。
- **ZOS 与投机互斥**（设计取舍，见 [docs/02 §6](docs/02-ZOS.md)）。
- **投机模式下 `max_tokens` 可能多 1 个 token**（边界回写越界一格）。
- **MoE 的 `moe_sum` 在部分 torch 版本报** `TypeError: sum() got an unexpected keyword argument 'out'`，用 `KRITACE_MOE_CUDA=0` 退回 Triton。
- 只支持 Qwen3 dense / Qwen3-MoE 两种结构；采样只实现贪心 / top-k / top-p / min-p。
- 没有 OpenAI 兼容 server、没有 metrics 导出、没有动态批的超时聚合。
