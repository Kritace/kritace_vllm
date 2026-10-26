import sys
sys.path.insert(0, "/root/dzyy-vllm/vllm-v6")
from src import Engine, EngineConfig, SamplingParams

PROMPT = (
    "Machine learning is a field of study that gives computers the ability to "
    "learn without being explicitly programmed. The term was coined in 1959 by "
    "Arthur Samuel, an American pioneer in the field of computer gaming and "
    "artificial intelligence. The synonym self-taught programming was also used"
)
MODEL = "/root/huggingface/Qwen3-0.6B"
N = 64

def run(k):
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
        gen = list(seq.token_ids[seq.num_prompt_tokens:])
        return gen, steps, seq.num_prompt_tokens
    finally:
        eng.destroy()

g0, s0, pl = run(0)
print(f"  prompt_tokens={pl}  K=0: steps={s0} gen={len(g0)}", flush=True)
for k in (3,):
    gk, sk, _ = run(k)
    same = g0 == gk
    print(f"  K={k}: steps={sk} gen={len(gk)}  一致={same}", flush=True)
    if not same:
        i = next((i for i,(x,y) in enumerate(zip(g0,gk)) if x!=y), min(len(g0),len(gk)))
        print(f"    首个不同下标(生成段)={i}", flush=True)
        lo, hi = max(0, i-4), i+5
        print(f"    K=0[..]: {g0[lo:hi]}", flush=True)
        print(f"    K={k}[..]: {gk[lo:hi]}", flush=True)
