"""打印失配步的注意力元数据与 top-3 logits。

散文提示词，K=3，只关心 L 接近 97 的那一步（基线在此处 pos98 = 13，投机给 11）。
"""
import os, sys
ROOT = "/root/dzyy-vllm/vllm-v6"
sys.path.insert(0, ROOT)
os.environ["V6_SPEC_DEBUG"] = "1"
os.environ["V6_SPEC_LOGITS"] = "1"
from src import Engine, EngineConfig, SamplingParams

PROMPT = (
    "Machine learning is a field of study that gives computers the ability to "
    "learn without being explicitly programmed. The term was coined in 1959 by "
    "Arthur Samuel, an American pioneer in the field of computer gaming and "
    "artificial intelligence. The synonym self-taught programming was also used"
)
eng = Engine(EngineConfig(
    model="/root/huggingface/Qwen3-0.6B", device_ids=[0], tp_size=1,
    enforce_eager=True, max_num_seqs=1, context_len=2048,
    num_kvcache_blocks=64, gpu_memory_utilization=0.85, prefix_backend="none",
    spec_draft="ngram", num_speculative_tokens=3,
))
try:
    seq = eng.add_request(PROMPT, SamplingParams(temperature=0.0, max_tokens=50, ignore_eos=True))
    while not seq.is_finished:
        eng.step()
    print("GEN:" + ",".join(map(str, seq.token_ids[seq.num_prompt_tokens:])))
finally:
    eng.destroy()
