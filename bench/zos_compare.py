"""ZOS 对拍与计时：同一负载分别用 KRITACE_ZOS=0/1 跑，比 token 与吞吐。

用法（在项目根目录）::

    KRITACE_ZOS=0 PYTHONPATH=. python bench/zos_compare.py --n 1 --max-tokens 64
    KRITACE_ZOS=1 PYTHONPATH=. python bench/zos_compare.py --n 1 --max-tokens 64

对拍口径：temperature=0 + ignore_eos，逐 token 必须完全一致；吞吐口径：
TTFT（首 token 时间）与 decode tok/s（总耗时扣掉首 token 之后的部分）。
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))


import argparse
import hashlib
import os
import time

from src import Engine, EngineConfig, SamplingParams

PROMPTS = [
    "The capital of France is",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    return",
    "Explain the difference between a process and a thread in one paragraph:",
    "1, 1, 2, 3, 5, 8, 13,",
    "Q: What is the boiling point of water at sea level?\nA:",
    "写一段关于秋天的短句：",
    "The transformer architecture consists of",
    "SELECT name, COUNT(*) FROM users GROUP BY name HAVING COUNT(*) >",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B"))
    ap.add_argument("--n", type=int, default=1, help="并发请求数")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--context-len", type=int, default=1024)
    ap.add_argument("--blocks", type=int, default=128)
    ap.add_argument("--max-num-seqs", type=int, default=8)
    args = ap.parse_args()

    mode = os.environ.get("KRITACE_ZOS", "0")
    prompts = (PROMPTS * 8)[: args.n]
    config = EngineConfig(
        model=args.model,
        enforce_eager=True,
        max_num_seqs=max(args.max_num_seqs, args.n),
        context_len=args.context_len,
        num_kvcache_blocks=args.blocks,
        gpu_memory_utilization=0.9,
        prefix_backend="none",
    )
    eng = Engine(config)
    try:
        seqs = [
            eng.add_request(
                p, SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)
            )
            for p in prompts
        ]
        t0 = time.perf_counter()
        t_first = None
        steps = 0
        while any(not s.is_finished for s in seqs):
            eng.step()
            steps += 1
            if t_first is None and all(len(s.token_ids) > s.num_prompt_tokens for s in seqs):
                t_first = time.perf_counter() - t0
        t_all = time.perf_counter() - t0
        outs = [list(s.token_ids[s.num_prompt_tokens :]) for s in seqs]
    finally:
        eng.destroy()

    total = sum(len(o) for o in outs)
    n_prompt = sum(len(s.token_ids) - s.num_prompt_tokens for s in seqs) - total
    digest = hashlib.sha1(repr(outs).encode()).hexdigest()[:12]
    decode_t = max(t_all - (t_first or 0.0), 1e-9)
    extra = max(total - len(outs), 1)
    print(
        f"mode={mode} n={args.n} max_tokens={args.max_tokens} steps={steps} "
        f"prompt_tok={n_prompt} gen_tok={total} tok_sha={digest}"
    )
    print(
        f"TTFT={1000.0 * (t_first or 0.0):.1f}ms total={1000.0 * t_all:.1f}ms "
        f"decode_tok_s={extra / decode_t:.2f} step_ms={1000.0 * t_all / max(steps, 1):.2f}"
    )
    print("tokens:", outs)


if __name__ == "__main__":
    main()
