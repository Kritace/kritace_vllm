"""散文提示词下 K=0 与 K=3 的逐步对比，聚焦"部分接受"路径。

输出：K=0 的逐步产出 + K=3 的 [spec] 明细，然后对齐找首个偏离步。
"""
import json, os, subprocess, sys

ROOT = "/root/dzyy-vllm/vllm-v6"
sys.path.insert(0, ROOT)
from src import Engine, EngineConfig, SamplingParams

PROMPT = (
    "Machine learning is a field of study that gives computers the ability to "
    "learn without being explicitly programmed. The term was coined in 1959 by "
    "Arthur Samuel, an American pioneer in the field of computer gaming and "
    "artificial intelligence. The synonym self-taught programming was also used"
)
MODEL = "/root/huggingface/Qwen3-0.6B"
N = 64


def mk(k):
    return Engine(EngineConfig(
        model=MODEL, device_ids=[0], tp_size=1, enforce_eager=True,
        max_num_seqs=1, context_len=2048, num_kvcache_blocks=64,
        gpu_memory_utilization=0.85, prefix_backend="none",
        spec_draft="ngram" if k else "", num_speculative_tokens=k,
    ))


# ---- K=0：逐步产出 ----
eng = mk(0)
seq = eng.add_request(PROMPT, SamplingParams(temperature=0.0, max_tokens=N, ignore_eos=True))
base_steps = []
while not seq.is_finished:
    outs, _ = eng._client.step() if False else (None, None)
    eng.step()
    # 从 seq 上取本步新增（step 后立即快照）
eng_based = list(seq.token_ids[seq.num_prompt_tokens:])
pl = seq.num_prompt_tokens
eng.destroy()
print(f"  prompt={pl}  K=0 生成 {len(eng_based)} 个", flush=True)

# ---- K=3：带 debug 逐步明细 ----
env = os.environ.copy()
env["V6_SPEC_DEBUG"] = "1"
code = f'''
import sys
sys.path.insert(0, "{ROOT}")
from src import Engine, EngineConfig, SamplingParams
PROMPT = {PROMPT!r}
eng = Engine(EngineConfig(model="{MODEL}", device_ids=[0], tp_size=1,
    enforce_eager=True, max_num_seqs=1, context_len=2048, num_kvcache_blocks=64,
    gpu_memory_utilization=0.85, prefix_backend="none",
    spec_draft="ngram", num_speculative_tokens=3))
seq = eng.add_request(PROMPT, SamplingParams(temperature=0.0, max_tokens={N}, ignore_eos=True))
while not seq.is_finished:
    eng.step()
import json
print("TOKENS:" + json.dumps(list(seq.token_ids[seq.num_prompt_tokens:])))
eng.destroy()
'''
r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
spec_lines = [l for l in r.stdout.splitlines() if l.startswith("[spec]")]
tok_line = [l for l in r.stdout.splitlines() if l.startswith("TOKENS:")]
spec_tokens = json.loads(tok_line[0][7:]) if tok_line else []

print(f"  K=3 生成 {len(spec_tokens)} 个，[spec] 步数={len(spec_lines)}", flush=True)
if spec_tokens and spec_tokens != eng_based:
    i = next((j for j, (x, y) in enumerate(zip(eng_based, spec_tokens)) if x != y), -1)
    print(f"  首个不同(生成段)={i}  绝对位置={pl + i}", flush=True)

print()
print("  === K=3 的 [spec] 明细（部分接受的步用 ★ 标出）===")
for l in spec_lines:
    j = int(l.split(" j=")[1].split()[0])
    mark = "★部分" if 0 < j < 3 else ("全中" if j == 3 else "全拒")
    print(f"  {mark}  {l}", flush=True)
