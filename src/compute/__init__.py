# ============================================================================
# 【本文件是做什么的】计算面的对外接口：三个符号 ——
#     ModelRunner     —— 组包 + 前向 + 采样，跑一步的执行器
#     CUDAGraphRunner —— decode 的录图与重放
#     Sampler         —— 采样 kernel
#
# 【它和 control/ 的界线是什么】就一条：**碰不碰张量**。
#     control/ 只做调度与进程编排，不碰张量、也不知道模型长什么样；
#     compute/ 才持有模型、KV 显存与所有 GPU 缓冲。
# 这条界线是三层架构的基础 —— 顺着它就能判断一段代码该放哪边。
# ============================================================================

"""Compute plane: model runner, CUDA Graph, sampler, layers, models."""

from __future__ import annotations

from .cuda_graph import CUDAGraphRunner
from .model_runner import ModelRunner
from .sampler import Sampler

# 对外契约。
__all__ = ["CUDAGraphRunner", "ModelRunner", "Sampler"]
