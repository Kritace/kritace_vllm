# MoE 与并行

## 1. MoE 前向

```
hidden → Router(linear) → top-k → 重归一化 → 按专家分组 → grouped GEMM → 加权求和 → hidden
```

- 路由：`src/moe/block.py`（`MoEBlock`），专家权重 `src/moe/experts.py`。
- 分组 GEMM 两套后端（`KRITACE_MOE_CUDA` 切换）：
  - CUDA 扩展 `src/moe/kernels/cuda.py` + `moe_kernels.cu`：`moe_align_block_size` 把 token 按专家重排，`BLOCK_M` 分块跑，最后 `moe_sum` 加权归并。
  - Triton `src/moe/kernels/triton.py`：同样的算法，纯 Python 侧可调试。
- 融合层 `src/moe/fused.py`：把 align / GEMM / reduce 串成一次调用，减少 Python 开销。

## 2. 四种并行

| 维度 | 切什么 | 通信 | 约束 |
|---|---|---|---|
| TP | head / 权重矩阵 | 每层 all-reduce | 不开 EP 时 MoE 按中间维切：`moe_intermediate_size % tp_size == 0` |
| EP | 专家 | 出口 all-reduce（token 留本地，**未做** all-to-all） | `num_experts % tp_size == 0` |
| PP | 层（可不等分） | p2p 收发 `IntermediateTensors` | 采样固定在最后一段的 tp0 |
| DP | 整副本 | 无 | 各副本独立进程组 |

校验在 `src/config.py`：`is_moe` 检测、EP/非 EP 的整除断言、`pp_layer_counts`。

## 3. 与投机的耦合

投机批形状是 `[bs, K+1]`：这 K+1 个 token 会全部穿过 MoE 层。

- dense：K+1 个位置复用同一批权重，验证几乎免费。
- MoE：只读 top-k 专家，K+1 个 token 触达的**不同专家数**变多 → 权重读取真的增加；EP 下出口 all-reduce 的数据量也变 K+1 倍。
- 同一序列的连续 K+1 个 token 大概率命中同一批专家 ⇒ EP 下容易出现热点卡；被拒的草稿是纯浪费。
- **批量不变性未验证**：`moe_align_block_size` 按专家重排、`BLOCK_M` 随 token 数变化，累加顺序可能改变；近 tie 时 argmax 可能翻转，从而破坏"投机与直接解码逐 token 一致"。
- CUDA Graph 形状：定宽 `[bs, K+1]` 让档位数乘 K+1，MoE 还要按形状池化 workspace。

## 4. 已验证与未验证

| 项 | 状态 |
|---|---|
| dense TP=2 / PP=2 | 有测试（`tests/distributed/`），本机单卡只跑过代码路径 |
| MoE 单卡前向 | `tests/gpu/test_moe_e2e.py`，本机无 MoE 权重未跑 |
| MoE + EP | `tests/distributed/test_moe_parallel.py`，需 2 卡 |
| MoE + CUDA Graph | `tests/gpu/test_moe_cudagraph.py`，需真 flash-attn |
| MoE + 投机 | **未组合过**（见 §3 的风险） |

## 5. 已知问题

- `moe_sum` 在部分 torch 版本报 `TypeError: sum() got an unexpected keyword argument 'out'`（`src/moe/kernels/cuda.py`）。用 `KRITACE_MOE_CUDA=0` 退回 Triton 后端。
- 4090D 无 NVLink 且 P2P 被禁，TP 的 all-reduce 走主机内存，带宽折损明显。
