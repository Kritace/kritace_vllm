# kritace_vllm

从零实现的 LLM 推理引擎（vLLM 精简复刻），单机单卡到单机多卡，代码量约 1.2 万行。

覆盖的核心机制：

| 主题 | 实现要点 |
|---|---|
| 分页 KV 缓存 | 块大小 256，`BlockManager` 管理空闲/已用块，`slot_mapping` 直接写页 |
| 前缀缓存 | `none` / `hash`（xxhash 链式索引）/ `radix`（基数树）三种后端 |
| 连续批处理 | chunked prefill + decode 混排（`mix_prefill_decode`），token 预算准入 |
| 抢占 | 显存不足时牺牲队尾序列（`preempt`），回到 waiting 队头重算 |
| CUDA Graph | decode 按 batch 档位捕获/重放，padding 行填安全值 |
| torch.compile | prefill 首 chunk 编译，符号形状模式 |
| MoE | 稀疏专家 + grouped GEMM（CUDA 扩展 / Triton 两套后端），TP 与 EP 两种切法 |
| 并行 | TP（切头/切权重）、PP（切层，收发中间张量）、DP（多副本） |
| 投机解码 | n-gram / 小模型草稿，一次前向验证 K+1 个位置（走 prefill 通道） |
| 零开销调度（ZOS） | CPU 调度与 GPU 前向重叠，用 GPU 侧 gather 补上「还没采到的 token」 |

设计文档见 [`docs/`](docs/)。

---

## 快速开始

```bash
pip install -r requirements.txt          # 需要 CUDA 版 torch

# 单卡
python run.py --model Qwen/Qwen3-0.6B --prompt "Hello, my name is"

# 多卡（TP=2）
torchrun --nproc_per_node 2 run_torchrun.py --model Qwen/Qwen3-0.6B --tp 2

# MoE + 专家并行
python run.py --model Qwen/Qwen1.5-MoE-A2.7B --tp 2 --ep
```

`config.yaml` 是默认参数，命令行覆盖它。常用开关：`--cg`（decode CUDA Graph）、`--compile`（prefill 编译）、`--prefix-backend hash`、`--spec-k`（投机步长）。

依赖 `flash-attn >= 2.5`（decode 用 `flash_attn_with_kvcache` 的分页 KV）。没装时引擎自动退回 `src/compute/layers/_flash_attn_fallback.py`（纯 torch SDPA，慢但能跑），便于在没有 wheel 的机器上开发。

---

## 目录结构

```
src/
  config.py            EngineConfig：所有可调参数与校验
  control/             控制面（不碰张量）
    engine.py            Engine 门面：tokenizer + 请求生命周期
    client.py/mp_client.py  进程内 / 跨进程客户端
    core.py              EngineCore：唯一的推理循环（调度→发射→取结果→回灌）
    scheduler.py         准入、抢占、KV 记账、ZOS 在途账本
    executor.py          UniprocExecutor / MultiprocExecutor（TP×PP 广播）
  data/                数据结构与 KV 账本
    sequence.py          Sequence：token、状态、块表、位置账本
    batch.py             ScheduledBatch → SchedulerOutput（msgspec 跨进程）
    kv.py/block_manager.py   分页 KV 分配与释放
    prefix/              前缀缓存三种后端
  compute/             计算面
    worker.py            Worker：建环境、装模型、调 Runner
    model_runner.py      前向、采样、CUDA Graph、采样 D2H 分期
    batch.py             组包：把调度结果写进常驻 GPU 缓冲
    layers/              算子（attention / linear / rotary / norm / 全词表并行）
    models/              Qwen3 dense 与 Qwen3-MoE
    sampler.py           贪心 / top-k / top-p / min-p
  moe/                  稀疏专家：路由、专家权重、fused 后端、kernel
  spec/                 投机解码：n-gram 草稿、验证、接受判定
  distributed/          TP/PP/EP 通信组与控制通道
bench/                  对拍与性能脚本（见 docs/05）
tests/                  unit / gpu / distributed 三层
utils/                  模型解析与采样参数（被 src 复用）
```

---

## 零开销调度（ZOS）

一步的墙钟时间可以写成 `T = G + B`：`G` 是 GPU 真正在算的时间，`B` 是 GPU 等 CPU 排下一批的时间。原版循环里 `B` 是赤裸的：

```
schedule → run(发射) → read_samples(等 GPU 算完) → update(回灌)
```

ZOS 把「等」推迟一拍，让 CPU 的调度落进 GPU 的算力窗口里：

```
schedule(t) → run(t) → swap_pending → read_samples(t-1) → finish(t-1)
```

代价是 CPU 账本比 GPU 落后一步，因此需要三条不变式（在途不可抢占、在途块落地前不归还、采样结果双缓冲 + 同流拷贝），以及一个关键补丁：下一步的输入 token 此刻还不知道 → `input_ids` 留空 + 记行号 `remap`，由 GPU 从上一步的**显存内**采样结果里 gather 补齐。

- 开关：`KRITACE_ZOS=0` 关闭（默认开）。与投机解码互斥，开了投机自动退回同步路径。
- 机制、代码地图、验证方法：见 [`docs/02-ZOS.md`](docs/02-ZOS.md)。

---

## 环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `KRITACE_ZOS` | `1` | 零开销调度总开关 |
| `KRITACE_ZOS_STATS` | `0` | 打印 `read_samples` 的等待统计（ready 比例 / 累计等待） |
| `KRITACE_ZOS_GPU` | `0` | 用 cudaEvent 量每步 GPU 忙时（配合步墙钟算气泡） |
| `KRITACE_NVTX` | `0` | 给 `step()` 的四段打 NVTX 区间（nsys 用） |
| `KRITACE_SPEC_DEBUG` | `0` | 投机：逐序列打印草稿/验证/接受长度 |
| `KRITACE_SPEC_LOGITS` | `0` | 投机：打印验证批的 logits 对齐信息 |
| `KRITACE_MOE_FORCE_EAGER` | `0` | MoE 强制 eager（排查 CUDA Graph 问题） |
| `KRITACE_MOE_CUDA` | `1` | `0` 关掉 CUDA 扩展，退回 Triton 后端做对照 |
| `KRITACE_SYS_PROF` | `0` | 全链路分段计时（只用于定位瓶颈，不作为性能数据） |

---

## 测试

```bash
python -m pytest tests/unit -q                      # 33 passed, 1 skipped（无需 GPU）
python -m pytest tests/gpu -q                       # 需要模型权重与本机 GPU
torchrun --nproc_per_node 2 -m pytest tests/distributed -q
```

模型默认从 `HF_HOME` 找 `Qwen3-0.6B` / `Qwen1.5-MoE-A2.7B`；缺失的用例会自动 skip，也可用 `--model`（跑脚本）或 `tests/helpers/paths.py`（测试）指定路径。

---

## 基准脚本

```bash
bash bench/final.sh /tmp/final.txt                  # ZOS 0/1 逐 token 对拍（三组场景）
python bench/repro_zos.py --rounds 3                # 交错跑，输出气泡/等待/正确性指标表
python bench/cpu_bubble.py                          # 分相计时：B 的量级
python bench/spec_sweep.py --k 1,2,4                # 投机加速比扫描
```

每个数字的口径、复现命令和本机实测见 [`docs/05-基准与复现.md`](docs/05-基准与复现.md)。

---

## 已知限制

- **CUDA Graph 需要真 flash-attn**：纯 torch 回退实现里有 host 同步，图捕获未验证。
- **ZOS 与投机不能同时开**（设计取舍，见 docs/02 §6）。
- **投机模式下 `max_tokens` 可能多 1 个 token**：验证批一次回写 `j+1` 个 token，边界那一步会越界一格（继承自投机路径的既有行为，未修）。
- **CUDA 扩展未覆盖所有 torch 版本**：`moe_sum` 在部分版本报 `sum() got an unexpected keyword argument 'out'`，用 `KRITACE_MOE_CUDA=0` 走 Triton 后端。
- 只支持 Qwen3 dense / Qwen3-MoE 两种结构；采样只实现了贪心、top-k、top-p、min-p。
