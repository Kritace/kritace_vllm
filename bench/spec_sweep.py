"""投机解码的 K 收益曲线 —— 端到端吞吐 + 贪心一致性。

**为什么必须按文本类型分开测**：n-gram 草稿靠"序列自身的历史"命中，
所以收益**完全取决于输入里有没有重复片段**。只报一个高重复提示词的数字
会严重高估（本机实测高重复 5.87× vs 代码 1.16×，差 5 倍）。

**同时做贪心一致性比对**：投机在贪心下应与基线逐 token 相同，
但**这个等价性只在 top-2 logits 间隔大于数值误差时成立** ——
自然散文上实测到过两个 kernel 的 logits 被舍入成完全相等的值，
`argmax` 并列取小下标导致分叉（详见 docs/03-投机解码.md）。
所以本脚本把每种文本的 K=0 结果留作基线逐项比对，不一致时打 `★不一致★`
**而不是判为失败** —— 它本身就是一条值得记录的实测边界。

用法::

    PYTHONPATH=. python bench/spec_sweep.py --devices 0
    PYTHONPATH=. python bench/spec_sweep.py --devices 0 --ks 0,3,5,8,12 --n 128
    PYTHONPATH=. python bench/spec_sweep.py --devices 0 --out bench/out/spec_sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# bench/ 不是包，同目录的 vs_vllm.py 直接 import；它内部会自己把 ROOT 加进 sys.path。
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tests.helpers.paths import DENSE_MODEL  # noqa: E402
from vs_vllm import _ours  # noqa: E402

# 三类文本，覆盖"重复度"这个唯一决定收益的变量。
#
# 选择依据（实测的 α 与 E[token] 见 docs/03-投机解码.md）：
#   高重复：模板化的问答对 → n-gram 几乎必中，用来给出**收益上限**
#   散文  ：自然语言 → 中等重复度，最接近真实业务场景
#   代码  ：几乎无重复片段 → 用来给出**收益下限**（也是"K 加大没用"的例子）
PROMPTS: dict[str, str] = {
    "重复(问答模板)": (
        "Q: What is the capital of France?\nA: The capital of France is Paris.\n"
        "Q: What is the capital of Germany?\nA: The capital of Germany is Berlin.\n"
        "Q: What is the capital of Italy?\nA: The capital of Italy is Rome.\n"
        "Q: What is the capital of Spain?\nA: The capital of Spain is Madrid.\n"
        "Q: What is the capital of Portugal?\nA: The capital of Portugal is"
    ),
    "散文(维基风)": (
        "Machine learning is a field of study that gives computers the ability to "
        "learn without being explicitly programmed. The term was coined in 1959 by "
        "Arthur Samuel, an American pioneer in the field of computer gaming and "
        "artificial intelligence. The synonym self-taught programming was also used"
    ),
    "代码(Python)": (
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n"
        "    pivot = arr[len(arr) // 2]\n"
        "    left = [x for x in arr if x < pivot]\n"
        "    middle = [x for x in arr if x == pivot]\n"
        "    right = [x for x in arr if x > pivot]\n    return"
    ),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--devices", default="0")
    p.add_argument("--ks", default="0,3,5,8", help="要扫的 K，逗号分隔（0 = 关闭投机）")
    p.add_argument("--spec-max-n", type=int, default=4, help="n-gram 的最大 n")
    p.add_argument("--n", type=int, default=64, help="每个请求生成多少 token")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--model", default=str(DENSE_MODEL))
    p.add_argument("--out", default="", help="把逐行结果写成 JSON")
    args = p.parse_args()

    devices = [int(x) for x in args.devices.split(",") if x.strip() != ""]
    ks = [int(x) for x in args.ks.split(",") if x.strip() != ""]
    if not Path(args.model).is_dir():
        raise SystemExit(f"missing model: {args.model}")

    rows: list[dict] = []
    for tag, prompt in PROMPTS.items():
        base_tokens: list[int] | None = None
        for k in ks:
            r = _ours(
                args.model,
                prompt,
                n=args.n,
                warmup=args.warmup,
                reps=args.reps,
                devices=devices,
                tp=args.tp,
                enable_ep=False,
                moe_backend="triton",
                spec_k=k,
                spec_ngram_max_n=args.spec_max_n,
            )
            if k == 0:
                base_tokens = r["tokens"]
                match = ""
            elif r["tokens"] == base_tokens:
                match = "一致"
            else:
                # 不是失败：数值并列导致的 argmax 翻转，见模块 docstring。
                match = "★不一致★"
            rows.append({"prompt": tag, "K": k, **r, "greedy_match": match})
            print(
                f"  {tag:16s} K={k:2d}  decode={r['decode_tok_s']:7.1f} tok/s  "
                f"e2e={r['e2e_s']:.2f}s  ttft={r['ttft_ms']:.1f}ms  {match}",
                flush=True,
            )

    print()
    print("=== decode 吞吐加速比（相对各自 K=0）===")
    base = {r["prompt"]: r["decode_tok_s"] for r in rows if r["K"] == 0}
    for tag in PROMPTS:
        if tag not in base:
            continue
        cells = []
        for k in ks:
            hit = [r for r in rows if r["prompt"] == tag and r["K"] == k]
            cells.append(f"K={k}: {hit[0]['decode_tok_s'] / base[tag]:5.2f}x" if hit else "")
        print(f"  {tag:16s} " + "  ".join(cells))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        print(f"\n  逐行结果已写入 {out}")


if __name__ == "__main__":
    main()
