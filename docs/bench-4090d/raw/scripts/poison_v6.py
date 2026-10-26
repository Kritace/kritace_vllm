"""证伪实验：paged varlen 核是否读了"写指针之外"的槽？

把整个 KV pool 预填成 100.0（正常应写入 0~几 的量级）。
若核只读 cu_seqlens_k 范围内的键 → 结果不受影响（毒值从未被读）。
若核读了越界槽 → 结果立刻崩坏。
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


def find_mr(eng):
    """找到 model_runner（不同版本属性链可能不同，逐个试）。"""
    cands = []
    o = eng
    for a in ("_client", "engine_core", "executor", "worker", "model_runner"):
        if hasattr(o, a):
            o = getattr(o, a)
            cands.append((a, type(o).__name__))
        else:
            cands.append((a, "MISSING"))
    print("  属性链:", " -> ".join(f"{a}({t})" for a, t in cands), flush=True)
    return o


# 基线真值
eng = mk(0)
seq = eng.add_request(PROMPT, SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True))
while not seq.is_finished:
    eng.step()
pl = seq.num_prompt_tokens
prompt_ids = list(seq.token_ids[:pl])
g = list(seq.token_ids[pl:])
eng.destroy()
print(f"  prompt={pl}  真值 g[41:46] = {g[41:46]}", flush=True)

K = 3
DRAFT = [g[41], g[42], g[43]]        # 真值草稿 → 应全中
WANT = [g[41], g[42], g[43], g[44]]

for poison, tag in ((None, "不投毒（对照）"), (100.0, "毒化整个 KV pool = 100.0")):
    NgramDrafter.propose = lambda self, ids, kk, _d=DRAFT: list(_d)
    eng = mk(K)
    try:
        if poison is not None:
            mr = find_mr(eng)
            mr.kv_cache.fill_(poison)
            print(f"  已把 kv_cache 填成 {poison}，shape={tuple(mr.kv_cache.shape)}", flush=True)
        seq = eng.add_request(prompt_ids + g[:40],
                              SamplingParams(temperature=0.0, max_tokens=6, ignore_eos=True))
        while not seq.is_finished:
            eng.step()
    finally:
        NgramDrafter.propose = _ORIG
        eng.destroy()
    print(f"  --- {tag} ---", flush=True)

print()
print(f"  期待 [spec] seg = {WANT}")
