# 设计文档

| 文档 | 讲什么 |
|---|---|
| [01-架构总览](01-架构总览.md) | 分层、请求生命周期、数据结构、组包/注意力/采样/CUDA Graph、四种并行、与 vLLM 的简化差异 |
| [02-ZOS](02-ZOS.md) | 零开销调度：气泡从哪来、五段改法、为什么只有 token id 拿不到、三条不变式、代码地图、与投机的冲突、何时不值得做 |
| [03-投机解码](03-投机解码.md) | 位置对齐、无损接受判定、n-gram 与小模型两种草稿、与调度器的耦合、实测 |
| [04-MoE与并行](04-MoE与并行.md) | 路由与 grouped GEMM、TP/EP/PP/DP、与投机的耦合风险、已验证矩阵 |
| [05-基准与复现](05-基准与复现.md) | 对拍、机制指标、气泡量级、为什么本机量不出加速、测量防坑、换机器怎么复现 |

代码入口：`src/control/core.py`（唯一推理循环）、`src/control/scheduler.py`（准入与账本）、`src/compute/model_runner.py`（前向与采样分期）。

## 环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `KRITACE_ZOS` | `1` | 零开销调度开关 |
| `KRITACE_ZOS_STATS` | `0` | `read_samples` 等待统计 |
| `KRITACE_ZOS_GPU` | `0` | cudaEvent 量 GPU 忙时 |
| `KRITACE_NVTX` | `0` | `step()` 四段打 NVTX（nsys 用） |
| `KRITACE_SPEC_DEBUG` | `0` | 投机：逐序列打印草稿/接受长度 |
| `KRITACE_SPEC_LOGITS` | `0` | 投机：打印验证批 logits 对齐 |
| `KRITACE_MOE_FORCE_EAGER` | `0` | MoE 强制 eager |
| `KRITACE_MOE_CUDA` | `1` | `0` 退回 Triton 后端 |
| `KRITACE_SYS_PROF` | `0` | 全链路分段计时 |
