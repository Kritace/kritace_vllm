# ==============================================================================
# draft model 草稿：用一个小模型自回归生成候选 token。
#
# 与 ngram.py 的接口完全一致（propose(token_ids, k) -> list[int]），
# 因此 SpecEngine 不需要区分草稿来源 —— 这也是把两者分成两个文件、
# 却又共用同一个 Protocol 的原因。
#
# 两个硬约束（推导见 docs/03-投机解码.md）：
#   1. draft 必须与 target **同词表**。否则 token id 对不上，
#      验证从第一步就是错的，而且不会报错，只会表现为接受率恒为 0。
#   2. draft 参数量必须远小于 target。K 次草稿各读一遍 draft 权重，
#      若 draft 接近 target 大小，"用便宜计算换昂贵访存"的前提就不成立。
# ==============================================================================
from __future__ import annotations

from typing import Callable

import torch

__all__ = ["DraftForward", "ModelDrafter"]

DraftForward = Callable[[list[int], int], torch.Tensor]
"""草稿模型的前向回调，签名 ``(tokens, start_pos) -> logits[vocab]``。

语义约定：
  - ``tokens`` 是**本次要处理的** token（通常只放 1 个新 token）
  - ``start_pos`` 是这批 token 在序列里的起始位置，供 RoPE 与写 KV 使用
  - 返回**最后一个位置**的一维 logits，形状 ``[vocab]``

由调用方（ModelRunner）注入，从而把本模块与并行切分、CUDA Graph
等实现细节解耦，也便于在 CPU 上用假函数单测。
"""


class ModelDrafter:
    """用小模型自回归生成草稿。

    与 ``NgramDrafter`` 的取舍：本类每步都要跑一次前向，K 越大草稿越贵；
    n-gram 几乎零成本，但只在输入含重复片段时才命中。
    """

    def __init__(
        self,
        forward: DraftForward,
        *,
        all_greedy: bool = True,
        temperature: float = 1.0,
        eos_token_id: int | None = None,
    ) -> None:
        # 参数自检：temperature <= 0 意味着"纯贪心"，那就必须显式声明 all_greedy=True，
        # 免得出现"设了温度 0 却走采样路径"这种自相矛盾的组合。
        if temperature <= 0 and not all_greedy:
            raise ValueError(
                "temperature <= 0 时应当走贪心路径，请设 all_greedy=True"
            )
        # 前向回调由外部注入（ModelRunner）—— 本模块因此不需要知道并行切分 / CUDA Graph 的存在。
        self.forward = forward
        self.all_greedy = all_greedy
        self.temperature = temperature
        self.eos_token_id = eos_token_id

    def propose(self, token_ids: list[int], k: int) -> list[int]:
        """自回归跑 k 步，返回最多 k 个草稿 token。

        中途生成出 EOS 就提前停下 —— 继续往下草稿没有意义，
        而且会让上层以为序列还能延长。
        """
        # 空序列或 k<=0 直接返回空草稿（与 NgramDrafter 保持同样的契约）。
        if k <= 0 or not token_ids:
            return []

        out: list[int] = []
        # 从最后一个 token 开始自回归 —— 草稿的起点必须是"序列已有的最后一步"，
        # 而不是新起一个位置。
        cur = token_ids[-1]
        # start_pos = 最后一个 token 的下标。它要一路传给 draft 的前向：
        #   ① RoPE 需要真实位置（位置编错了整条草稿都是废的）；
        #   ② 写 KV 需要知道写到哪一槽 —— 草稿的 KV 必须与目标模型的位置对齐，
        #      否则验证阶段读到的 KV 是错位的。
        start_pos = len(token_ids) - 1
        # 自回归 K 步（中途遇到 EOS 会提前 break）。
        for _ in range(k):
            # 每步只喂 1 个 token（[cur]）—— 草稿模型是标准的自回归解码，
            # 这次前向的 logits 会用来选下一个 token，同时把 KV 写进序列。
            logits = self.forward([cur], start_pos)
            if logits.dim() != 1:
                raise RuntimeError(
                    f"draft forward 应返回 [vocab] 的一维 logits，"
                    f"实际形状 {tuple(logits.shape)}"
                )
            # 贪心或温度 0 时用 argmax；否则按温度采样（见 _pick）。
            tok = self._pick(logits)
            out.append(tok)
            # cur 立刻变成刚选出的 token，供下一次前向使用 ——
            # 这一步保证了草稿是"一条连贯的续写"，而不是 k 个互相独立的猜测。
            cur = tok
            # 位置同步前移，保证每一步的 RoPE / KV 位置都正确。
            start_pos += 1
            if self.eos_token_id is not None and tok == self.eos_token_id:
                # 【遇到 EOS 提前停】继续往下草稿没有意义（序列就要结束了），
                # 而且会让上层以为还能延长。停在 EOS 本身是安全的：
                # 它照常参与验证，被接受就正常结束序列。
                break
        return out

    # 选一个 token。两条路径与 target 侧的语义保持一致：
    # 贪心 = argmax；采样 = 按温度 softmax 后 multinomial。
    def _pick(self, logits: torch.Tensor) -> int:
        # 温度 0 视同贪心（与 engine.py 里 _probs 的特判呼应）。
        if self.all_greedy or self.temperature <= 0:
            return int(logits.argmax().item())
        # 采样路径：温度必须先作用于 logits 再 softmax。
        # ⚠️ 这里的温度必须与 target 侧**用同一个值**，否则接受判定 min(1, p/q) 不成立。
        probs = torch.softmax(logits / self.temperature, dim=-1)
        return int(torch.multinomial(probs, 1).item())
