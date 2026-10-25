# ==============================================================================
# 投机解码「一步编排」的单元测试。
#
# 用假 drafter + 假 target_forward 把 SpecEngine 的逻辑隔离出来测，
# 不需要 GPU、不需要真实模型。重点验证三件事：
#   1. 全接受时一步产出 K+1 个 token（这就是收益的来源）
#   2. 第 j 个被拒时，产出 draft[:j] 加上 target 在位置 j 的修正 token
#   3. 没有草稿时必须返回空结果，而不能被当成"草稿全被拒"
# ==============================================================================
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.spec import SpecEngine  # noqa: E402

VOCAB = 8
K = 3


class FakeDrafter:
    """固定返回预设草稿的 drafter，用来精确控制测试场景。"""

    def __init__(self, draft: list[int]) -> None:
        self.draft = draft
        self.calls: list[tuple[list[int], int]] = []

    def propose(self, token_ids: list[int], k: int) -> list[int]:
        self.calls.append((list(token_ids), k))
        return self.draft[:k]


def make_target(expected: list[int]):
    """构造 target_forward：第 i 个位置的 argmax 恰好是 expected[i]。

    输入长度应与 expected 等长（SpecEngine 会校验这一点）。
    """

    def forward(tokens: list[int]) -> torch.Tensor:
        logits = torch.full((len(tokens), VOCAB), -10.0)
        for i in range(len(tokens)):
            logits[i, expected[i % len(expected)]] = 10.0
        return logits

    return forward


def _engine(draft: list[int]) -> SpecEngine:
    return SpecEngine(FakeDrafter(draft), num_speculative_tokens=K, all_greedy=True)


# ---------------------------------------------------------------------------
# 位置对齐：验证输入必须是 [last_token, *draft]
# ---------------------------------------------------------------------------


def test_verify_input_prepends_last_token():
    e = _engine([1, 2, 3])
    assert e.build_verify_input([7, 8, 9], [1, 2, 3]) == [9, 1, 2, 3]


def test_verify_input_needs_nonempty_tokens():
    e = _engine([1])
    with pytest.raises(ValueError):
        e.build_verify_input([], [1])


def test_target_forward_length_is_checked():
    """target_forward 返回的位置数不对时必须报错，而不是静默算错。"""
    e = _engine([1, 2, 3])

    def bad_forward(tokens: list[int]) -> torch.Tensor:
        return torch.zeros(len(tokens) + 2, VOCAB)

    with pytest.raises(RuntimeError):
        e.step([5], bad_forward)


# ---------------------------------------------------------------------------
# 接受 / 拒绝
# ---------------------------------------------------------------------------


def test_full_accept_emits_k_plus_one():
    """全部接受：产出 K 个草稿 + 第 K 个位置白送的 token = K+1 个。"""
    draft = [1, 2, 3]
    # 验证输入是 [last, 1, 2, 3] -> 位置 0..3 的 argmax 依次为 1,2,3,4
    target = make_target([1, 2, 3, 4])
    res = _engine(draft).step([9], target)

    assert res.num_accepted == K
    assert res.emitted == [1, 2, 3, 4]
    assert len(res.emitted) == K + 1
    assert res.num_verified == K + 1


def test_first_reject_emits_single_corrected_token():
    """第 0 个就被拒：只产出 target 的修正 token。"""
    draft = [1, 2, 3]
    # 位置 0 的 argmax 是 5，与草稿的 1 不同 -> 立即拒绝
    target = make_target([5, -1, -1, -1])
    res = _engine(draft).step([9], target)

    assert res.num_accepted == 0
    assert res.emitted == [5]


def test_partial_accept_stops_at_mismatch():
    """接受前 2 个，第 3 个不匹配 -> 产出前 2 个 + 修正 token。"""
    draft = [1, 2, 3]
    target = make_target([1, 2, 7, -1])
    res = _engine(draft).step([9], target)

    assert res.num_accepted == 2
    assert res.emitted == [1, 2, 7]


def test_num_verified_equals_draft_plus_one():
    draft = [1, 2, 3]
    target = make_target([1, 2, 3, 4])
    res = _engine(draft).step([9], target)
    assert res.num_verified == len(draft) + 1


# ---------------------------------------------------------------------------
# 没有草稿时的退化行为
# ---------------------------------------------------------------------------


def test_no_draft_returns_empty_and_skips_target():
    """n-gram 没匹配上时必须直接返回空，绝不能白白跑一次 target 前向。"""
    called = {"n": 0}

    def target(tokens: list[int]) -> torch.Tensor:
        called["n"] += 1
        return torch.zeros(len(tokens), VOCAB)

    res = _engine([]).step([9], target)
    assert not res.has_draft
    assert res.emitted == []
    assert called["n"] == 0


def test_drafter_receives_k_and_history():
    d = FakeDrafter([1, 2, 3])
    e = SpecEngine(d, num_speculative_tokens=K, all_greedy=True)
    e.step([4, 5, 6], make_target([1, 2, 3, 4]))
    assert d.calls == [([4, 5, 6], K)]


# ---------------------------------------------------------------------------
# 接受率的统计口径
# ---------------------------------------------------------------------------


def test_acceptance_is_ratio_over_draft_length():
    draft = [1, 2, 3]
    full = _engine(draft).step([9], make_target([1, 2, 3, 4]))
    assert full.acceptance == pytest.approx(1.0)

    partial = _engine(draft).step([9], make_target([1, 2, 7, -1]))
    assert partial.acceptance == pytest.approx(2 / 3)

    none = _engine(draft).step([9], make_target([5, -1, -1, -1]))
    assert none.acceptance == pytest.approx(0.0)


def test_invalid_k_rejected():
    with pytest.raises(ValueError):
        SpecEngine(FakeDrafter([]), num_speculative_tokens=0)
