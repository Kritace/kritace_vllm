# 3050test —— 本机（RTX 3050 Laptop 4 GB / WSL2）的性能实验记录

这里放**手动测试**的产物：cudaEvent 探针日志、cpu_bubble 分段、nsys 报告、实验笔记。
目的不是复现 4090D 的数字，而是**自己走一遍测量流程，并学会读 nsys / profiler / ncu 的输出**。

> 本机数字**不可与 `docs/07-实测汇总-4090D.md` 直接比较**：显存 4 GB、无 NVLink、
> 未装 flash-attn（走 `_flash_attn_fallback.py` 的 SDPA 路径）、WSL2。

## 目录

```
bench/3050test/
├── README.md      本文件：约定 + 工具手册
├── logs/          cudaEvent / zos-stats / cpu_bubble 等文本日志（*.log）
├── nsys/          .nsys-rep 及 nsys export 出来的 .sqlite
├── ncu/           .ncu-rep（本机 WSL 采不到，见"工具可用性"）
└── notes/         每次实验的记录（模板：notes/实验记录模板.md）
```

## 工具可用性（2026-10-30 本机实测）

| 工具 | 状态 | 能回答的问题 |
|---|---|---|
| 项目探针 `KRITACE_ZOS_GPU` / `KRITACE_ZOS_STATS` | ✅ | 每步 GPU 忙多久、读采样结果要不要等 |
| `src/utils/profiler.py`（Timer） | ✅ | 代码块级 CPU 耗时（`torch.cuda.Event` 成对） |
| `bench/cpu_bubble.py` | ✅ | schedule / run / read / finish 四段耗时 |
| **nsys 2025.3.2**（`/usr/local/cuda-13.0/bin/nsys`） | ✅ | CUDA API + kernel 时间线、NVTX 区间、GPU 忙闲 |
| `torch.profiler` | ✅ | Python 调用栈 + CUDA kernel 的轻量时间线 |
| **ncu 2026.3.1** | ❌ `ERR_NVGPUCTRPERM` | 需要 GPU 性能计数器，WSL 下无权限 |

`ncu` 报错原文：

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU
Performance Counters on the target device 0.
```

处理办法（三选一）：

1. **换原生 Linux**（租的 4090D 机器）跑 ncu —— 最省事，WSL 官方不支持 ncu；
2. Windows 侧放开计数器权限：注册表
   `HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak\RmProfilingAdminOnly = 0`，重启
   （原生 Windows 有效，WSL 内多数情况仍不可用）；
3. 用 **nsys 的 kernel 汇总**替代 ncu 的"时间维度"部分（见下），
   只是看不到 SM 利用率 / occupancy / 访存吞吐这些硬件指标。

## 命名规范

```
<工具>_<模式>_n<并发>_k<生成长度>_<HHMM>.<后缀>
```

例：

- `logs/event_zos0_n1_k128_1130.log`
- `logs/event_zos1_n1_k128_1132.log`
- `nsys/nsys_zos1_n1_k32_1140.nsys-rep`
- `nsys/nsys_zos1_n1_k32_1140.sqlite`

规则：**模式、并发、生成长度三者必须能从文件名读出来**，否则一周后无法配对比较。

## 实验阶梯（按顺序做）

### E1 · cudaEvent 探针（已完成基线）

```bash
cd ~/kritace_vllm
PYTHONPATH=. KRITACE_ZOS=0 KRITACE_ZOS_STATS=1 KRITACE_ZOS_GPU=1 \
  python bench/zos_compare.py --n 1 --max-tokens 32 --context-len 1024 --blocks 8 \
  2>&1 | tee bench/3050test/logs/event_zos0_n1_k32_$(date +%H%M).log
```

要抄的数：`tok_sha` / `steps` / `step_ms` / `ready%` / `avg_wait` / `avg_gpu`，
气泡 = `step_ms − avg_gpu`。

⚠️ 两个开关是**模块级常量**（`src/compute/model_runner.py:60-66`），必须在 `python`
启动前设置；结果在 `EngineCore.destroy()` 打印（`src/control/core.py:274-277`），
**不能 Ctrl-C**。

### E2 · 四段 CPU 耗时（不需要 nsys）

```bash
PYTHONPATH=. KRITACE_ZOS=0 python bench/cpu_bubble.py --n 8 --max-tokens 32 \
  2>&1 | tee bench/3050test/logs/cpububble_zos0_n8_k32_$(date +%H%M).log
PYTHONPATH=. KRITACE_ZOS=1 python bench/cpu_bubble.py --n 8 --max-tokens 32 \
  2>&1 | tee bench/3050test/logs/cpububble_zos1_n8_k32_$(date +%H%M).log
```

输出 `schedule / run / read / finish / step` 五个中位数。
`ZOS=0` 时"GPU 每步空泡 ≈ finish + schedule + run"（见 `bench/cpu_bubble.py` 文件头）。

### E3 · nsys + NVTX 时间线（**本机可做**）

```bash
KRITACE_NVTX=1 nsys profile -t cuda,nvtx,osrt --force-overwrite true \
  -o bench/3050test/nsys/nsys_zos1_n1_k32_$(date +%H%M) \
  python bench/zos_compare.py --n 1 --max-tokens 32 --context-len 1024 --blocks 8

# ① NVTX 区间汇总：zos.schedule / zos.launch / zos.read / zos.finish 的 count/avg/med
nsys stats --report nvtx_sum bench/3050test/nsys/nsys_zos1_n1_k32_XXXX.nsys-rep

# ② kernel 汇总（ncu 的"时间维度"替代品）
nsys stats --report cuda_gpu_kern_sum bench/3050test/nsys/nsys_zos1_n1_k32_XXXX.nsys-rep

# ③ 图形界面（WSLg）
nsys-ui bench/3050test/nsys/nsys_zos1_n1_k32_XXXX.nsys-rep
```

三条要点：

- `-t nvtx` **必须写**，否则 `NVTX_EVENTS` 表是空的；
- 开了 `--cg`（CUDA Graph）时要加 `--cuda-graph-trace=node`，否则图内的 kernel
  会被折叠成一次 graph launch；
- 采集本身有开销，**同一种采集方式下的相对对比**才有意义，别拿采集前后的绝对值比。

### E4 · 从 nsys sqlite 做自定义计算

```bash
nsys export --type sqlite --force-overwrite true \
  -o bench/3050test/nsys/zos1_n1_k32.sqlite \
  bench/3050test/nsys/nsys_zos1_n1_k32_XXXX.nsys-rep

python bench/gap_step.py bench/3050test/nsys/zos1_n1_k32.sqlite
```

`bench/gap_step.py:20-34` 里的两条 SQL 是模板：

```sql
SELECT start, end FROM NVTX_EVENTS WHERE text='zos.launch' AND end IS NOT NULL ORDER BY start;
SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start;
```

### E5 · torch.profiler（轻量、纯 Python）

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True) as prof:
    ...  # 这里跑一步 Engine.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("bench/3050test/logs/step_trace.json")   # 拖进 chrome://tracing 或 perfetto
```

### E6 · ncu（换机器再做）

```bash
# 原生 Linux + 已放开计数器权限
ncu --set full --target-processes all -f -o bench/3050test/ncu/smoke python your_script.py
ncu --import smoke.ncu-rep --page details
```

## 每次实验记什么（notes/）

见 `notes/实验记录模板.md`。**最低要求**：命令全文、环境变量、六个关键数字、结论一句话、
异常现象。没有这四项，数字一周后就废了。

## 与 git 的关系

`.gitignore` 已忽略 `*.log`、`*.nsys-rep`、`*.ncu-rep`，所以产物不会进版本库；
`README.md` / `notes/*.md` 会被 git 看到。若想让笔记也不进库：

```bash
echo "bench/3050test/" >> .git/info/exclude
```
