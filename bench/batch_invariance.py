"""批量不变性 + tie 裕度：验证"同一个 prompt 单独跑 vs 与别人一起跑"会不会给出不同 token。

两种模式：
  A) max_num_seqs=8，8 条 prompt 同时进批 → 每步 batch=8
  B) max_num_seqs=1，8 条 prompt 逐条串行（每条都单独占一个 batch=1）
对每一行比较 token 序列，并打印首次分叉位置。

同时（KRITACE_MARGIN=1 由本脚本设置）记录贪心步里 top1-top2 的 logit 差值：
裕度远大于 0 → 累加顺序扰不动 argmax；裕度在 1e-6 量级 → 一次重排就可能翻转。
"""
import os
import sys

sys.path.insert(0, "/home/Kritace/kritace_vllm")
os.environ.setdefault("KRITACE_MARGIN", "1")

from src import Engine, EngineConfig  # noqa: E402
from src.compute import model_runner as mr  # noqa: E402

PROMPTS = [
    "The capital of France is",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    return",
    "1, 1, 2, 3, 5, 8, 13,",
    "Q: What is the boiling point of water at sea level?\nA:",
    "Translate to English: 今天天气很好，我们一起去公园散步吧。",
    "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return",
    "The three laws of thermodynamics are:",
    "Write a haiku about autumn:\n",
]


def _close(eng) -> None:
    for name in ("shutdown", "close", "destroy"):
        fn = getattr(eng, name, None)
        if callable(fn):
            fn()
            return


def build(max_seqs: int) -> Engine:
    return Engine(
        EngineConfig(
            model=os.path.expanduser("~/huggingface/Qwen3-0.6B"),
            enforce_eager=True,
            max_num_seqs=max_seqs,
            num_kvcache_blocks=64,
            prefix_backend="none",
        )
    )


def run_batch(n_tok: int) -> list[list[int]]:
    eng = build(8)
    seqs = [eng.add_request(p, mr_tokens(n_tok)) for p in PROMPTS]
    while not all(s.is_finished for s in seqs):
        eng.step()
    out = [list(s.token_ids[-n_tok:]) for s in seqs]
    _close(eng)
    return out


def run_single(n_tok: int) -> list[list[int]]:
    eng = build(1)
    out = []
    for p in PROMPTS:
        seq = eng.add_request(p, mr_tokens(n_tok))
        while not seq.is_finished:
            eng.step()
        out.append(list(seq.token_ids[-n_tok:]))
    _close(eng)
    return out


def mr_tokens(n: int):
    from src import SamplingParams

    return SamplingParams(max_tokens=n, temperature=0.0)


if __name__ == "__main__":
    n_tok = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    mr._MARGIN.clear()
    a = run_batch(n_tok)
    b = run_single(n_tok)
    total = 0
    bad = 0
    for i, (x, y) in enumerate(zip(a, b)):
        total += n_tok
        if x != y:
            bad += 1
            first = next((k for k in range(n_tok) if x[k] != y[k]), -1)
            print(f"[DIFF] prompt{i} 首次分叉位置={first} bs8={x[:8]} bs1={y[:8]}")
    print(f"[batch-invariance] prompts={len(PROMPTS)} tokens/row={n_tok} total={total} 分叉行={bad}")
    if bad == 0:
        print("  两种 batch 组成下逐 token 完全一致")
    stats = mr.margin_stats()
    if stats:
        print(f"[margin] steps={stats['n']} min={stats['min']:.3e} p1={stats['p1']:.3e} "
              f"p50={stats['p50']:.3e}")
