"""把两种情况在 cached=96 那一步的元数据与 logits 直接对拍。

情况 U：真实 n-gram 草稿（会偏离）
情况 R：真值草稿（不偏离）
同一步的 input tokens 与 L/cached 完全相同，差异只可能来自元数据或 KV 内容。
"""
import os, sys
ROOT = "/root/dzyy-vllm/vllm-v6"
sys.path.insert(0, ROOT)
os.environ["V6_SPEC_DEBUG"] = "1"
os.environ["V6_SPEC_LOGITS"] = "1"
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


eng = mk(0)
seq = eng.add_request(PROMPT, SamplingParams(temperature=0.0, max_tokens=60, ignore_eos=True))
while not seq.is_finished:
    eng.step()
pl = seq.num_prompt_tokens
g = list(seq.token_ids[pl:])
eng.destroy()

for tag in ("U 真实草稿", "R 真值草稿"):
    if tag.startswith("R"):
        NgramDrafter.propose = lambda self, ids, kk: (
            list(g[len(ids) - pl : len(ids) - pl + kk])
            if 0 <= len(ids) - pl and len(ids) - pl + kk <= len(g) else []
        )
    else:
        NgramDrafter.propose = _ORIG
    eng = mk(3)
    import io, contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            seq = eng.add_request(PROMPT, SamplingParams(temperature=0.0, max_tokens=45, ignore_eos=True))
            while not seq.is_finished:
                eng.step()
    finally:
        NgramDrafter.propose = _ORIG
        eng.destroy()

    lines = buf.getvalue().splitlines()
    print(f"\n  ================= {tag} =================")
    for i, l in enumerate(lines):
        if "cache_seqlens=[96]" in l:
            for k2 in range(max(0, i - 8), min(len(lines), i + 6)):
                s = lines[k2]
                if s.startswith("[spec-dbg]") or s.startswith("[wire]") or s.startswith("[spec] "):
                    print("   ", s[:190])
            break
    print(f"    该次运行的生成段前 12 个 = {list(seq.token_ids[pl:pl+12])}")
