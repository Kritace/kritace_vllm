"""决定性实验：区分"q_len=4 的 kernel 数值"与"j>0 的多 token 记账"。

四个对照（同一散文提示词，64 token，贪心）：
  A) K=0                    基线
  B) q_len=2, j≡0           强制草稿 [0]        —— 无记账参与
  D) q_len=4, j≡0           强制草稿 [0,0,0]    —— 仍无记账参与，但 kernel 形状同投机
  C) q_len=4, j 变化        真实 n-gram 草稿

A==B==D 而 C 不同  → 记账 bug（j>0 才触发），kernel 数值无辜
D != A             → varlen(q=4) 与 decode 的 bf16 数值不同，属 kernel 差异
"""
import sys
sys.path.insert(0, "/root/dzyy-vllm/vllm-v6")

from src import Engine, EngineConfig, SamplingParams
from src.spec import NgramDrafter

_PROSE = (
    "Machine learning is a field of study that gives computers the ability to "
    "learn without being explicitly programmed. The term was coined in 1959 by "
    "Arthur Samuel, an American pioneer in the field of computer gaming and "
    "artificial intelligence. The synonym self-taught programming was also used"
)
MODEL = "/root/huggingface/Qwen3-0.6B"
N = 64
_ORIG = NgramDrafter.propose


def run(k, force=None):
    if force is None:
        NgramDrafter.propose = _ORIG
    else:
        NgramDrafter.propose = lambda self, ids, kk: list(force)
    eng = Engine(EngineConfig(
        model=MODEL, device_ids=[0], tp_size=1, enforce_eager=True,
        max_num_seqs=1, context_len=2048, num_kvcache_blocks=64,
        gpu_memory_utilization=0.85, prefix_backend="none",
        spec_draft="ngram" if k else "", num_speculative_tokens=k,
    ))
    try:
        seq = eng.add_request(_PROSE, SamplingParams(temperature=0.0, max_tokens=N, ignore_eos=True))
        steps = 0
        while not seq.is_finished and steps < 300:
            eng.step(); steps += 1
        return list(seq.token_ids[seq.num_prompt_tokens:]), steps
    finally:
        eng.destroy()
        NgramDrafter.propose = _ORIG


def show(tag, tok, steps, ref):
    if ref is None:
        print(f"  {tag:26s} steps={steps:3d}  (基线)", flush=True)
        return
    if tok == ref:
        print(f"  {tag:26s} steps={steps:3d}  与基线一致 ✓", flush=True)
    else:
        i = next((j for j, (x, y) in enumerate(zip(ref, tok)) if x != y), min(len(ref), len(tok)))
        print(f"  {tag:26s} steps={steps:3d}  ✗ 首个不同={i}", flush=True)
        print(f"      基线[{max(0,i-3)}:{i+4}] = {ref[max(0,i-3):i+4]}", flush=True)
        print(f"      本组[{max(0,i-3)}:{i+4}] = {tok[max(0,i-3):i+4]}", flush=True)


base, sb = run(0)
show("A) K=0 基线", base, sb, None)

b, sb2 = run(1, [0])
show("B) q_len=2, j≡0", b, sb2, base)

d, sd = run(3, [0, 0, 0])
show("D) q_len=4, j≡0", d, sd, base)

c, sc = run(3)
show("C) q_len=4, 真实草稿", c, sc, base)
