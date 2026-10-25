# ==============================================================================
# 投机解码一步的编排：草稿 → 验证 → 接受/拒绝 → 产出 token。
#
# 本类不直接接触 Worker / CUDA Graph / 并行通信，只依赖两个回调：
#   - drafter.propose(token_ids, k)  -> list[int]        草稿
#   - target_forward(token_ids)      -> Tensor[K+1, V]   验证 logits
# 因此可以在 CPU 上用假模型把整条链路单测掉
# （tests/unit/test_spec_engine.py），这也是推荐的上手顺序。
#
# 位置对齐关系（最容易写错的地方，单独说明）：
#
#   已有序列      t_1 ... t_L
#   验证输入      [t_L, d_1, d_2, ..., d_K]     共 K+1 个位置
#   输出位置      pos 0 预测 d_1
#                 pos 1 预测 d_2
#                 ...
#                 pos K-1 预测 d_K
#                 pos K   给出"额外 token"（全接受时白送的那个）
#
# 于是"接受的草稿数 j"与"最终产出"有一个统一表达：
#
#   emitted = draft[:j] + [argmax(logits[j])]
#
#  - j = 0（第一个就被拒）-> 只产出 target 的修正 token
#  - j = K（全部接受）    -> 产出 draft 全部 + 第 K 个位置白送的 token
# ==============================================================================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import torch

from .verify import greedy_accept_length, sample_accept

# 三个对外名字：草稿接口协议、单步结果、步进器。
__all__ = ["Drafter", "SpecStepResult", "SpecEngine"]


# 【为什么用 Protocol 而不是抽象基类】结构化类型：任何满足该方法的对象都能当草稿器用，
# 不需要显式继承（`NgramDrafter` 与 `ModelDrafter` 都不继承它）。
# 好处有二：① 实现方不必为了类型检查 import 这个模块；
#          ② 单测里塞一个假草稿器（返回固定 token 的 lambda）就能覆盖整条链路。
class Drafter(Protocol):
    """草稿生成器的统一接口。

    ``NgramDrafter`` 与 ``ModelDrafter`` 都满足这个协议，
    上层 ``SpecEngine`` 因此不需要关心草稿是查表还是跑模型得到的。
    """

    # 契约：返回最多 k 个候选，不足 k 也没关系（草稿器尽力而为）。
    # 返回空列表表示"这次没有草稿"，调用方应当退回普通 decode ——
    # 这与"草稿全被拒"是两件完全不同的事（后者仍会产出一个 token）。
    def propose(self, token_ids: list[int], k: int) -> list[int]:
        """返回最多 k 个候选 token；没有可用的草稿时返回空列表。"""
        ...


TargetForward = Callable[[list[int]], torch.Tensor]


@dataclass
# ============================================================================
# 一步投机解码的全部结果。
# 四个字段合起来就能算出所有需要的统计量（接受率、每步产出、KV 要预留几位）。
# ============================================================================
class SpecStepResult:
    """一步投机解码的结果。

    ``draft`` 为空表示这次没拿到草稿（比如 n-gram 没匹配上），
    调用方应当退回普通 decode —— 此时 ``emitted`` 为空，
    不能把"没有草稿"当成"草稿全被拒"。
    """

    # 这一步用的草稿（空 = 没拿到草稿）。
    draft: list[int] = field(default_factory=list)
    # 被 target 接受的草稿个数 j。
    num_accepted: int = 0
    # 最终写进序列的 token：长度 1 ~ K+1（最少一个修正 token，最多全部草稿 + 白送的一个）。
    emitted: list[int] = field(default_factory=list)
    # 提交给 target 验证的位置数 = K+1（不是 k！）。
    # KV 侧要按这个数预留槽位（见 data/kv.py 的 can_append_n）。
    num_verified: int = 0
    """实际提交给 target 验证的位置数（= len(draft) + 1）。"""

    @property
    # 有草稿才算一次有效的投机步 —— 调用方用它区分「退回普通 decode」与「全被拒」。
    def has_draft(self) -> bool:
        return len(self.draft) > 0

    @property
    # 接受率 α = 接受的草稿数 / 草稿长度。
    # 它是判断"投机这一步划不划算"的核心指标：期望产出 ≈ 1 + α + α² + …（全接受时还会白送一个）。
    def acceptance(self) -> float:
        """接受率，没有草稿时返回 0。"""
        return self.num_accepted / len(self.draft) if self.draft else 0.0


# ============================================================================
# 投机解码的编排器：只管"草稿 → 验证 → 接受/产出"这三步的先后与取值，
# 不碰 Worker / CUDA Graph / 并行通信，也不碰 KV Cache 的物理布局。
# 依赖只有两个回调（drafter 与 target_forward），所以能在 CPU 上单测。
# ============================================================================
class SpecEngine:
    """投机解码的步进器。

    参数
    ----
    drafter : 草稿来源（n-gram 或小模型）
    num_speculative_tokens : 每步草稿长度 K
    all_greedy : 是否为贪心解码。贪心时退化成"比较 argmax"，
        不必构造概率分布，更快也更稳。
    """

    def __init__(
        self,
        drafter: Drafter,
        *,
        num_speculative_tokens: int,
        all_greedy: bool = True,
    ) -> None:
        # K 至少为 1：K=0 等于没有投机，应该干脆不开这个特性。
        if num_speculative_tokens < 1:
            raise ValueError(
                f"num_speculative_tokens must be >= 1, got {num_speculative_tokens}"
            )
        # 草稿器由外部注入（n-gram 或小模型），本类不关心它是哪种。
        self.drafter = drafter
        self.k = num_speculative_tokens
        self.all_greedy = all_greedy
        # 采样路径下 draft 与 target 必须用同一温度；贪心路径用不到。
        # 采样路径的温度；贪心路径用不到它，默认 1.0 即可。
        self._temperature: float = 1.0

    # 【构造验证输入】`[last_token, *draft]`
    # 为什么要重复最后一个已生成的 token：这样 target 的第 i 个输出位置
    # 就直接对应"第 i 个草稿该不该接受"，不必额外保存上一步的 logits。
    # 代价是多算一个位置（K 变 K+1），换来实现上的对称。
    def build_verify_input(self, token_ids: list[int], draft: list[int]) -> list[int]:
        """构造验证批次的输入：``[last_token, *draft]``。

        为什么带上 last_token：这样输出位置 0 就直接对应"d_1 该不该接受"，
        不需要额外保存上一步的 logits。代价是多算一个位置，
        但换来实现上的对称性（K+1 个位置，K+1 个输出），不难理解。
        """
        # 空序列说明调用方状态不对：至少要有一个已生成的 token 才能构造验证输入。
        if not token_ids:
            raise ValueError("token_ids 不能为空")
        # ★ 最容易写错的一行（位置对齐）：
        #     输入  = [t_L, d_1, ..., d_K]        （K+1 个位置）
        #     输出  = pos 0 判 d_1、pos 1 判 d_2、…、pos K 给出"白送的那个"
        # 于是"j 个被接受"时统一写成 `emitted = draft[:j] + [argmax(logits[j])]`。
        return [token_ids[-1], *draft]

    # ============================================================================
    # 一步投机解码：拿草稿 → 交 target 验证 → 判定 → 产出 token。
    # ============================================================================
    def step(
        self,
        token_ids: list[int],
        target_forward: TargetForward,
        *,
        generator: torch.Generator | None = None,
    ) -> SpecStepResult:
        """执行一步投机解码。

        ``target_forward`` 接收验证输入（长度 K+1），
        返回形状 ``[K+1, vocab]`` 的 logits。
        """
        # 第一步：要草稿（可能少于 K 个，也可能没有）。
        draft = self.drafter.propose(token_ids, self.k)
        # 没有草稿 ⇒ 直接返回空结果，由调用方退回普通 decode。
        # 关键区别：这里不是"草稿全被拒"（那还会产出一个修正 token），
        # 而是"这一步什么都没做"。混淆这两者会让调度器少跑一次前向。
        if not draft:
            return SpecStepResult()

        # 第二步：把 [last_token, *draft] 交给 target 跑一次前向。
        verify_input = self.build_verify_input(token_ids, draft)
        logits = target_forward(verify_input)
        # 第三步：校验形状。位置数错一个就会出现"用 pos i 的判断去接受 d_{i+1}"
        # 这种静默错位 —— 数值还在、序列全歪，所以这里必须硬校验。
        expected = len(verify_input)
        # 期望恰好 K+1 个位置（见 build_verify_input 的对齐说明）。
        if logits.shape[0] != expected:
            raise RuntimeError(
                f"target_forward 应返回 {expected} 个位置的 logits，"
                f"实际得到 {logits.shape[0]}"
            )

        # 注意用 len(draft) 而不是 self.k：草稿可能不足 K 个。
        k = len(draft)
        # 【贪心路径】temperature == 0 / greedy 时走这里：
        # 直接比较"草稿 token"与"target 的 argmax"，不必构造概率分布。
        if self.all_greedy:
            argmax = logits.argmax(dim=-1).tolist()
            # argmax[:k] 只取前 k 个位置（第 K 个位置是「白送 token」的候选，不参与接受判定）。
            j = greedy_accept_length(draft, argmax[:k])
            # ★ 统一表达（j 取任何值都对）：
            #     j = 0 → 只产出 target 给的修正 token（等于退回普通 decode 的产出）
            #     j = K → 产出全部草稿 + 第 K 个位置白送的那个
            # 这就是"投机解码至少不亏"的原因：每步最少产出 1 个 token。
            emitted = [*draft[:j], int(argmax[j])]
            return SpecStepResult(
                draft=draft, num_accepted=j, emitted=emitted, num_verified=expected
            )

        # 【采样路径】先把 logits 转成带温度的概率（q 与 p 必须在同一温度下）。
        draft_probs = _probs(logits, self._temperature)
        j, extra = sample_accept(
            draft,
            draft_probs[:k],
            draft_probs,
            generator=generator,
        )
        # sample_accept 约定必返回额外 token（全接受时从 target 第 K 个位置采一个）。
        assert extra is not None, "sample_accept 必须返回额外 token"
        emitted = [*draft[:j], extra]
        return SpecStepResult(
            draft=draft, num_accepted=j, emitted=emitted, num_verified=expected
        )

    # 【温度必须两边一致】draft 与 target 若用不同温度，p 与 q 不在同一尺度上，
    # min(1, p/q) 的推导前提就不成立 —— 表面能跑，实际已经不再无损。
    def set_temperature(self, temperature: float) -> None:
        """设置采样温度（仅采样路径生效）。

        ⚠️ draft 与 target 必须使用**同一温度**，否则接受概率
        min(1, p/q) 失去意义 —— 两个分布不在同一尺度上。
        """
        # 下界 1e-6：避免后续除法除零（真正的 temperature == 0 由 _probs 特判）。
        self._temperature = max(temperature, 1e-6)


# temperature <= 0 时退化成 one-hot（即"纯贪心"）：把最大项置 1、其余 0。
# 这样采样路径与贪心路径共用同一套代码，不需要在调用处写分支。
def _probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        idx = logits.argmax(dim=-1, keepdim=True)
        out = torch.zeros_like(logits)
        # scatter 构造 one-hot：好处是不需要知道 argmax 的具体位置，逐元素表达。
        return out.scatter_(-1, idx, 1.0)
    return torch.softmax(logits / temperature, dim=-1)
