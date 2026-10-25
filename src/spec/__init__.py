"""投机解码（Speculative Decoding）。

   ngram.py   序列自身的 n-gram 后缀匹配，零额外模型
   draft.py   小模型自回归生成草稿，与 target 共享 GPU
   verify.py  接受 / 拒绝判定与修正分布重采样
   engine.py  一步的编排：草稿 → 验证 → 接受/拒绝 → 产出 token

设计要点见 ``docs/03-投机解码.md``：decode 阶段是访存受限，
投机解码用"多算几个位置"换取"每步的权重读取被摊薄"。

推荐的上手顺序：先跑 n-gram 版本（``spec_draft="ngram"``），
它不需要第二个模型就能把 KV 账本、接受判定、回滚逻辑全部验证一遍。
"""

from __future__ import annotations

# 两个草稿实现导出在同一个名字下：SpecEngine 只认 propose() 这个接口，
# 所以"用 n-gram 还是用 draft 模型"对上层是完全透明的（config.spec_draft 决定）。
from .draft import DraftForward, ModelDrafter
from .engine import Drafter, SpecEngine, SpecStepResult, TargetForward
from .ngram import NgramDrafter
# 接受判定的五个函数都导出：除引擎自用外，测试与调参时也要单独调它们
# （例如只想知道某个草稿在给定分布下的接受率）。
from .verify import (
    accept_probability,
    apply_top_k_top_p,
    greedy_accept_length,
    residual_distribution,
    sample_accept,
)

__all__ = [
    "Drafter",
    "DraftForward",
    "ModelDrafter",
    "NgramDrafter",
    "SpecEngine",
    "SpecStepResult",
    "TargetForward",
    "accept_probability",
    "apply_top_k_top_p",
    "greedy_accept_length",
    "residual_distribution",
    "sample_accept",
]
