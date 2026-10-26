import sys
sys.path.insert(0, "/root/dzyy-vllm/vllm-v6")
from src import Engine, EngineConfig, SamplingParams

PROMPT = (
    "Q: What is the capital of France?\nA: The capital of France is Paris.\n"
    "Q: What is the capital of Germany?\nA: The capital of Germany is Berlin.\n"
    "Q: What is the capital of Italy?\nA: The capital of Italy is Rome.\n"
    "Q: What is the capital of Spain?\nA: The capital of Spain is Madrid.\n"
    "Q: What is the capital of Portugal?\nA: The capital of Portugal is"
)
eng = Engine(EngineConfig(
    model="/root/huggingface/Qwen3-0.6B", device_ids=[0], tp_size=1,
    enforce_eager=True, max_num_seqs=1, context_len=2048,
    num_kvcache_blocks=64, gpu_memory_utilization=0.85, prefix_backend="none",
    spec_draft="ngram", num_speculative_tokens=3,
))
try:
    print("=== 第一次请求（warmup 风格，4 token）===", flush=True)
    s1 = eng.add_request(PROMPT, SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True))
    n = 0
    while not s1.is_finished and n < 40:
        eng.step(); n += 1
    print(f"  warmup 结束 n_steps={n} is_finished={s1.is_finished}", flush=True)

    print("=== 第二次请求（新建 Sequence）===", flush=True)
    s2 = eng.add_request(PROMPT, SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True))
    print(f"  新 seq id={s2.seq_id} len={len(s2.token_ids)} draft={s2.spec_draft_ids}", flush=True)
    n = 0
    while not s2.is_finished and n < 80:
        eng.step(); n += 1
    print(f"  结束 n_steps={n} is_finished={s2.is_finished} gen={s2.num_completion_tokens}", flush=True)
except Exception as e:
    print(f"!!! 异常: {type(e).__name__}: {e}", flush=True)
finally:
    eng.destroy()
