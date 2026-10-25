# ============================================================================
# 【本文件是做什么的】激活层的薄封装：把注册好的自定义算子
# `torch.ops.vllm.silu_and_mul` 包成一个 nn.Module，模型侧就能像用普通层一样用它。
#
# 【为什么还要多这一层壳】直接调算子也能跑，包成 Module 有两个实际好处：
#     ① 模型代码统一：`self.act = SiluAndMul()` 与其它层写法一致，不用到处 import 算子；
#     ② module 边界：torch.compile 以 Module 为粒度做图切分，裸函数调用不会形成边界。
#
# 【输入布局：gate 在前、up 在后】最后一维是 2×中间维的拼接，
#     约定是 [gate | up]，输出 `silu(gate) * up`。这个顺序由模型侧的权重加载决定
#     （gate_proj 写第 0 段、up_proj 写第 1 段），算子自己不做拆分判断。
#
# ⚠️ MoE 那边另有一个顺序相反的算子：`moe.kernels.cuda.silu_and_mul` 走的是
#   [up | gate]（`up * silu(gate)`），因为专家权重 w13 的布局是 up 在前。
#   两个顺序都对，但用错地方不会报错，只会算出错误的数值 —— 排查时先看这里。
# ============================================================================

from __future__ import annotations
import torch
from torch import nn


# 极薄的一层壳：不持任何参数、不做形状检查，纯粹把调用转发给自定义算子。
# 之所以还要有这一层，是为了让模型侧能像用普通 nn 模块那样用（统一接口、
# 也方便被 torch.compile 识别为 module 边界）。
class SiluAndMul(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 直接调注册好的算子。输入必须已经按 [.., 2*intermediate] 排好
        # （拼接顺序见 qwen3.py 里 gate_up_proj 的输出布局），算子自己不做拆分。
        return torch.ops.vllm.silu_and_mul(x)
