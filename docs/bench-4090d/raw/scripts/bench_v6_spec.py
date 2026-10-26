"""v6 投机解码的端到端测量：K 的收益曲线 + 贪心一致性验证。

为什么要三类提示词：n-gram 草稿靠"序列自身的历史"命中，收益**强烈依赖文本重复度**。
只报一个高重复提示词的数字会严重高估；所以同时报散文与代码两类。
"""
import json
import sys

ROOT = "/root/dzyy-vllm/vllm-v6"
sys.path.insert(0, ROOT)
from importlib.util import module_from_spec, spec_from_file_location

spec = spec_from_file_location("vs_vllm", f"{ROOT}/bench/vs_vllm.py")
mod = module_from_spec(spec)
sys.modules["vs_vllm"] = mod
spec.loader.exec_module(mod)

PROMPTS = {
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
MODEL = "/root/huggingface/Qwen3-0.6B"
KS = [0, 3, 5, 8]
N = 64

rows = []
for tag, prompt in PROMPTS.items():
    base_tokens = None
    for k in KS:
        r = mod._ours(
            MODEL, prompt, n=N, warmup=2, reps=3,
            devices=[0], tp=1, enable_ep=False, moe_backend="triton",
            spec_k=k, spec_ngram_max_n=4,
        )
        same = "" if k == 0 else ("一致" if r["tokens"] == base_tokens else "★不一致★")
        if k == 0:
            base_tokens = r["tokens"]
        rows.append({"prompt": tag, "K": k, **r, "greedy_match": same})
        print(f"  {tag:16s} K={k}  decode={r['decode_tok_s']:7.1f} tok/s  "
              f"e2e={r['e2e_s']:.2f}s  ttft={r['ttft_ms']:.1f}ms  {same}", flush=True)

with open("/tmp/v6_spec.json", "w") as f:
    json.dump(rows, f, ensure_ascii=False, indent=1)

print()
print("=== 汇总：decode 吞吐加速比（相对各自 K=0）===")
base = {}
for r in rows:
    if r["K"] == 0:
        base[r["prompt"]] = r["decode_tok_s"]
for tag in PROMPTS:
    line = f"  {tag:16s}"
    for k in KS:
        v = next(r["decode_tok_s"] for r in rows if r["prompt"] == tag and r["K"] == k)
        line += f"  K={k}: {v / base[tag]:5.2f}x"
    print(line)
