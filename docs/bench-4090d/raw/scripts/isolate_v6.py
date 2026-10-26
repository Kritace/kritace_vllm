"""隔离复现：在**干净历史**下重放投机验证的那一步。

做法：把 prompt 截到目标位置前一个 token（于是 prefill 后 cached 恰好等于我们要的位置），
再用 monkeypatch 强制草稿 = **基线真值**的后 3 个 token，于是第一个投机步就是待复现的那一步。

若干净历史下 seg 完全等于真值 → 说明问题由"投机历史留下的 KV"引起；
若干净历史下就已经错 → 说明这一步的前向本身有问题（与历史无关）。
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
_ORIG = NgramDrafter.propose


def mk(k):
    return Engine(EngineConfig(
        model=MODEL, device_ids=[0], tp_size=1, enforce_eager=True,
        max_num_seqs=1, context_len=2048, num_kvcache_blocks=64,
        gpu_memory_utilization=0.85, prefix_backend="none",
        spec_draft="ngram" if k else "", num_speculative_tokens=k,
    ))


# ---- 先拿基线真值 ----
eng = mk(0)
seq = eng.add_request(PROMPT, SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True))
while not seq.is_finished:
    eng.step()
pl = seq.num_prompt_tokens
prompt_ids = list(seq.token_ids[:pl])
g = list(seq.token_ids[pl:])
eng.destroy()
print(f"  prompt={pl}  真值前 12 个 = {g[:12]}", flush=True)

# ---- 逐位置复现 ----
for k in (10, 20, 30, 40, 50):
    if k + 5 >= len(g):
        continue
    draft = [g[k + 1], g[k + 2], g[k + 3]]
    want = [g[k + 1], g[k + 2], g[k + 3], g[k + 4]]
    NgramDrafter.propose = lambda self, ids, kk, _d=draft: list(_d)
    eng = mk(3)
    try:
        seq = eng.add_request(prompt_ids + g[:k],
                              SamplingParams(temperature=0.0, max_tokens=6, ignore_eos=True))
        while not seq.is_finished:
            eng.step()
    finally:
        NgramDrafter.propose = _ORIG
        eng.destroy()

print()
print("  上面每行的 [spec] 里：cached = 复现位置，seg 四元组应与真值一致")
print("  真值对照（生成段下标 -> token）：")
for k in (10, 20, 30, 40, 50):
    if k + 5 >= len(g):
        continue
    print(f"    k={k:2d} 期待 seg = {[g[k+1], g[k+2], g[k+3], g[k+4]]}")
