"""判别：散文提示词下的不一致，是"多 token 记账 bug"还是"kernel 数值差异"。

思路：让 n-gram 草稿恒返回 1 个几乎不会被预测的 token（0），于是
  - 前向仍走 varlen attention（q_len = 2）
  - 但接受长度 j 恒为 0 → **完全没有多 token 记账参与**：
    每步只写 1 个有效 KV、只推进 1 个 token，与 K=0 语义完全相同
若此时输出与 K=0 仍不一致 → 差异来自 kernel 数值（varlen vs decode）
若一致 → 多 token 路径确有 bug
"""
import sys
sys.path.insert(0, "/root/dzyy-vllm/vllm-v6")

from src import Engine, EngineConfig, SamplingParams
from src.spec import NgramDrafter

PROMPT = (
    "Machine learning is a field of study that gives computers the ability to "
    "learn without being explicitly programmed. The term was coined in 1959 by "
    "Arthur Samuel, an American pioneer in the field of computer gaming and "
    "artificial intelligence. The synonym self-taught programming was also used"
)
MODEL = "/root/huggingface/Qwen3-0.6B"
N = 64


def run(k, force_draft=None):
    if force_draft is not None:
        NgramDrafter.propose = lambda self, ids, kk: list(force_draft)
    eng = Engine(EngineConfig(
        model=MODEL, device_ids=[0], tp_size=1, enforce_eager=True,
        max_num_seqs=1, context_len=2048, num_kvcache_blocks=64,
        gpu_memory_utilization=0.85, prefix_backend="none",
        spec_draft="ngram" if k else "", num_speculative_tokens=k,
    ))
    try:
        seq = eng.add_request(PROMPT, SamplingParams(temperature=0.0, max_tokens=N, ignore_eos=True))
        steps = 0
        while not seq.is_finished and steps < 300:
            eng.step(); steps += 1
        return list(seq.token_ids[seq.num_prompt_tokens:]), steps
    finally:
        eng.destroy()


base, sb = run(0)
print(f"  A) K=0 基线            : steps={sb:3d}  tokens={base[:12]}...", flush=True)

b2, sb2 = run(0)
print(f"  A') K=0 复跑（确定性）  : 与 A 一致={b2 == base}", flush=True)

# 强制草稿恒为 [0]，K=1 → q_len=2 的 varlen，但 j 恒为 0
c, sc = run(1, force_draft=[0])
print(f"  B) varlen q=2, j≡0     : steps={sc:3d}  与 A 一致={c == base}", flush=True)
if c != base:
    i = next((j for j, (x, y) in enumerate(zip(base, c)) if x != y), -1)
    print(f"     首个不同(生成段)={i}", flush=True)
    lo, hi = max(0, i - 3), i + 4
    print(f"     A[..]={base[lo:hi]}", flush=True)
    print(f"     B[..]={c[lo:hi]}", flush=True)

d, sd = run(3)
e, se = run(3)
print(f"  C) K=3 正常投机        : steps={sd:3d}  与 A 一致={d == base}", flush=True)
print(f"  C') K=3 复跑（确定性）  : 与 C 一致={e == d}", flush=True)
