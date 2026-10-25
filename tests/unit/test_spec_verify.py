# ==============================================================================
# 投机解码接受判定的单元测试。
#
# 最关键的一条是 test_*_matches_target_distribution_*：
# 用大量采样做统计验证，确认"草稿 + 接受/拒绝"的最终输出分布
# 与直接采样 target 一致。这是投机解码"无损"的定义本身，
# 比逐行审查代码可靠得多。
#
# 全部测试只用 CPU，不依赖 GPU。
# ==============================================================================
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.spec import (  # noqa: E402
    NgramDrafter,
    accept_probability,
    apply_top_k_top_p,
    greedy_accept_length,
    residual_distribution,
    sample_accept,
)

N_SAMPLES = 60_000
ATOL = 0.02


def _simulate_one_step(
    p: torch.Tensor,
    q: torch.Tensor,
    *,
    n: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """模拟 n 次"投机一步"，返回最终输出 token 的经验分布。

    一步的流程：草稿按 q 采样 -> 按 min(1, p/q) 决定接受 ->
    被拒则从 (p-q) 的正部归一化后重采样。
    """
    counts = torch.zeros_like(p)
    resid = residual_distribution(p, q)
    for _ in range(n):
        tok = int(torch.multinomial(q, 1, generator=generator).item())
        r = torch.rand((), generator=generator)
        if r <= accept_probability(p[tok], q[tok]):
            out = tok
        else:
            out = int(torch.multinomial(resid, 1, generator=generator).item())
        counts[out] += 1
    return counts / n


# ---------------------------------------------------------------------------
# 基础性质
# ---------------------------------------------------------------------------


def test_residual_distribution_sums_to_one():
    p = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
    q = torch.tensor([[0.3, 0.6, 0.1], [0.4, 0.4, 0.2]])
    r = residual_distribution(p, q)
    assert torch.allclose(r.sum(dim=-1), torch.ones(2), atol=1e-6)


def test_residual_distribution_is_nonnegative():
    p = torch.tensor([0.1, 0.2, 0.7])
    q = torch.tensor([0.8, 0.1, 0.1])
    r = residual_distribution(p, q)
    assert (r >= 0).all()
    # p 与 q 差距最大的位置应当拿到最大的残差质量
    assert r.argmax().item() == 2


def test_residual_falls_back_to_p_when_p_equals_q():
    """p == q 时 (p-q) 的正部全为 0，必须回退到 p 而不是产生 NaN。"""
    p = torch.tensor([0.5, 0.5])
    r = residual_distribution(p, p)
    assert torch.isfinite(r).all()
    assert torch.allclose(r.sum(), torch.tensor(1.0))


def test_accept_probability_handles_zero_draft_prob():
    """草稿概率为 0 时接受概率记为 1，不因除零而拒绝。"""
    got = accept_probability(torch.tensor(0.4), torch.tensor(0.0))
    assert got.item() == pytest.approx(1.0)


def test_accept_probability_clamped_at_one():
    """草稿过于保守（p > q）时接受概率封顶为 1。"""
    got = accept_probability(torch.tensor(0.9), torch.tensor(0.1))
    assert got.item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 分布还原：投机解码无损性的统计验证
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "p_list,q_list",
    [
        ([0.6, 0.3, 0.1], [0.3, 0.6, 0.1]),  # 草稿在 B 上过度自信
        ([0.7, 0.2, 0.1], [0.1, 0.2, 0.7]),  # 草稿严重偏离
        ([0.4, 0.35, 0.25], [0.34, 0.33, 0.33]),  # 接近但仍不同
        ([1.0, 0.0, 0.0], [0.2, 0.5, 0.3]),  # 目标退化为 one-hot
    ],
)
def test_output_distribution_matches_target(p_list, q_list):
    p = torch.tensor(p_list)
    q = torch.tensor(q_list)
    g = torch.Generator().manual_seed(1234)
    est = _simulate_one_step(p, q, n=N_SAMPLES, generator=g)
    assert torch.allclose(est, p, atol=ATOL), (
        f"投机输出的经验分布 {est.tolist()} 偏离 target {p.tolist()}，"
        "说明接受/拒绝规则写错了"
    )


def test_accept_rate_is_sum_of_min():
    """接受率应当等于 sum(min(p, q)) —— 与第 4 节的推导一致。"""
    p = torch.tensor([0.6, 0.3, 0.1])
    q = torch.tensor([0.3, 0.6, 0.1])
    expected = float(torch.minimum(p, q).sum())

    g = torch.Generator().manual_seed(7)
    accepted = 0
    n = N_SAMPLES
    for _ in range(n):
        tok = int(torch.multinomial(q, 1, generator=g).item())
        if torch.rand((), generator=g) <= accept_probability(p[tok], q[tok]):
            accepted += 1
    assert accepted / n == pytest.approx(expected, abs=0.015)


# ---------------------------------------------------------------------------
# 贪心路径
# ---------------------------------------------------------------------------


def test_greedy_accept_all():
    assert greedy_accept_length([1, 2, 3], [1, 2, 3]) == 3


def test_greedy_accept_prefix():
    assert greedy_accept_length([1, 2, 9], [1, 2, 3]) == 2


def test_greedy_reject_first():
    assert greedy_accept_length([9, 2, 3], [1, 2, 3]) == 0


def test_greedy_empty_draft():
    assert greedy_accept_length([], []) == 0


def test_sample_accept_full_accept_returns_extra_token():
    """全部接受时，第 K 个位置还要白送一个 token。"""
    k = 3
    draft_ids = [1, 2, 3]
    onehot = torch.zeros(k + 1, 5)
    onehot[:, :4] = 1.0  # 让 draft 与 target 在这 4 个位置上完全一致
    draft_probs = onehot[:k]
    target_probs = onehot
    g = torch.Generator().manual_seed(0)
    j, extra = sample_accept(draft_ids, draft_probs, target_probs, generator=g)
    assert j == k
    assert extra is not None and 0 <= extra < 5


# ---------------------------------------------------------------------------
# 截断：draft 与 target 必须用同一套
# ---------------------------------------------------------------------------


def test_apply_top_k_top_p_normalizes():
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    out = apply_top_k_top_p(probs, top_k=2)
    assert out.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert (out[2:] == 0).all()


def test_apply_min_p_zeroes_small_probabilities():
    probs = torch.tensor([0.7, 0.2, 0.05, 0.05])
    out = apply_top_k_top_p(probs, min_p=0.5)
    assert out[0] > 0
    assert (out[1:] == 0).all()


# ---------------------------------------------------------------------------
# n-gram 草稿
# ---------------------------------------------------------------------------


def test_ngram_proposes_repeat_successor():
    d = NgramDrafter(max_n=3)
    assert d.propose([5, 9, 2, 7, 5, 9, 2], k=2) == [7, 5]


def test_ngram_returns_empty_on_no_history():
    d = NgramDrafter(max_n=3)
    assert d.propose([1, 2, 3], k=2) == []


def test_ngram_respects_k():
    d = NgramDrafter(max_n=2)
    got = d.propose([1, 2, 1, 2, 1, 2], k=3)
    assert len(got) == 3


def test_ngram_rejects_bad_bounds():
    with pytest.raises(ValueError):
        NgramDrafter(max_n=1, min_n=3)


def test_ngram_short_sequence_is_safe():
    d = NgramDrafter(max_n=4)
    assert d.propose([7], k=4) == []
