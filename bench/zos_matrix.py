"""ZOS 对拍矩阵：KRITACE_ZOS=0/1 同一组参数各跑一次，逐 token 比较。

用法::

    KRITACE_ZOS=0 PYTHONPATH=. python bench/zos_matrix.py --case mixed
    KRITACE_ZOS=1 PYTHONPATH=. python bench/zos_matrix.py --case mixed

输出一行 ``CASE <name> sha=<...>`` 与完整 tokens JSON，外层脚本 diff 两者。
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))


import argparse
import hashlib
import json
import os
import time

from src import Engine, EngineConfig, SamplingParams

SHORT = [
    "The capital of France is",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    return",
    "1, 1, 2, 3, 5, 8, 13,",
    "Q: What is the boiling point of water at sea level?\nA:",
]
SENT = (
    "The history of computing is a history of abstraction. "
    "Each layer hides the one below it, and each hidden layer costs something. "
)


def long_prompt(repeat: int) -> str:
    return SENT * repeat


def build_prompts(case: str, n: int, repeat: int) -> list[str]:
    if case == "chunk":            # 一条长 prompt 被 max-batched-tokens 切成多块
        return [long_prompt(repeat)] * n
    if case == "mixed":            # 长 prompt + 短 prompt 同批
        return [long_prompt(repeat)] + (SHORT * 8)[: max(n - 1, 0)]
    return (SHORT * 16)[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B"))
    ap.add_argument("--case", default="short")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--blocks", type=int, default=64)
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--max-batched-tokens", type=int, default=8192)
    ap.add_argument("--context-len", type=int, default=2048)
    ap.add_argument("--prefix-backend", default="none")
    ap.add_argument("--long-repeat", type=int, default=6)
    ap.add_argument("--eager", type=int, default=1)
    # 投机解码：>0 时开 K 个草稿（n-gram），用于验证「ZOS 与投机互斥」。
    ap.add_argument("--spec-k", type=int, default=0)
    args = ap.parse_args()

    mode = os.environ.get("KRITACE_ZOS", "0")
    prompts = build_prompts(args.case, args.n, args.long_repeat)
    config = EngineConfig(
        model=args.model,
        enforce_eager=bool(args.eager),
        max_num_seqs=max(args.max_num_seqs, len(prompts)),
        max_num_batched_tokens=args.max_batched_tokens,
        context_len=args.context_len,
        num_kvcache_blocks=args.blocks,
        gpu_memory_utilization=0.9,
        prefix_backend=args.prefix_backend,
        num_speculative_tokens=args.spec_k,
        spec_draft="ngram" if args.spec_k else None,
    )
    eng = Engine(config)
    try:
        seqs = [
            eng.add_request(
                p,
                SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True),
            )
            for p in prompts
        ]
        t0 = time.perf_counter()
        steps = 0
        while any(not s.is_finished for s in seqs):
            eng.step()
            steps += 1
        elapsed = time.perf_counter() - t0
        outs = [list(s.token_ids[s.num_prompt_tokens :]) for s in seqs]
        prompt_tok = sum(s.num_prompt_tokens for s in seqs)
    finally:
        eng.destroy()

    digest = hashlib.sha1(repr(outs).encode()).hexdigest()[:12]
    print(
        f"CASE {args.case} n={args.n} mode={mode} steps={steps} prompt_tok={prompt_tok} "
        f"gen_tok={sum(len(o) for o in outs)} elapsed={elapsed:.3f}s sha={digest}"
    )
    print("TOKENS " + json.dumps(outs))


if __name__ == "__main__":
    main()
