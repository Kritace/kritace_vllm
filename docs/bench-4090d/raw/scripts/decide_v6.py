"""判定位置 98 应该是 13 还是 11。

三个口径，同一段上下文（提示词 56 token + 基线已生成的 42 个 token = 位置 0..97）：
  A) 一次性 prefill 98 个 token（cached=0，不复用任何缓存）→ 预测位置 98
  B) prefill 97 个 token 再逐 token 生成 2 个 → 第 2 个也是位置 98，但由 decode kernel 算出
  C) 投机路径（已知 = 11）
"""
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
G = SamplingParams(temperature=0.0, max_tokens=N, ignore_eos=True)


def mk(spec_k=0):
    return Engine(EngineConfig(
        model=MODEL, device_ids=[0], tp_size=1, enforce_eager=True,
        max_num_seqs=1, context_len=2048, num_kvcache_blocks=64,
        gpu_memory_utilization=0.85, prefix_backend="none",
        spec_draft="ngram" if spec_k else "", num_speculative_tokens=spec_k,
    ))


def gen(eng, prompt, sp):
    seq = eng.add_request(prompt, sp)
    while not seq.is_finished:
        eng.step()
    return list(seq.token_ids)


# ---- 基线 K=0 ----
eng = mk(0)
full = gen(eng, PROMPT, G)
prompt_ids = full[: full.index(full[0])]  # 占位，下面重新取
eng.destroy()

eng = mk(0)
seq = eng.add_request(PROMPT, G)
while not seq.is_finished:
    eng.step()
pl = seq.num_prompt_tokens
base = list(seq.token_ids[pl:])
eng.destroy()
print(f"  prompt={pl}  基线生成 {len(base)} 个；位置98 的基线值 = {base[42]}", flush=True)

CTX = None  # 占位

# ---- A) 一次性 prefill 98 个 token ----
eng = mk(0)
seq = eng.add_request(PROMPT, G)
while not seq.is_finished:
    eng.step()
ctx_ids = list(seq.token_ids[: pl + 42])       # 位置 0..97
eng.destroy()
print(f"  上下文长度 = {len(ctx_ids)}（位置 0..{len(ctx_ids)-1}）", flush=True)

eng = mk(0)
seq = eng.add_request(ctx_ids, SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
while not seq.is_finished:
    eng.step()
a_tok = seq.token_ids[-1]
eng.destroy()
print(f"  A) 一次性 prefill        → 位置98 = {a_tok}", flush=True)

# ---- B) prefill 97 个 token 再生成 2 个（decode kernel）----
eng = mk(0)
seq = eng.add_request(ctx_ids[:-1], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True))
while not seq.is_finished:
    eng.step()
b_ids = list(seq.token_ids[len(ctx_ids) - 1:])
eng.destroy()
print(f"  B) prefill97+逐token生成 → 位置97={b_ids[0]}  位置98={b_ids[1]}", flush=True)

# ---- C) 投机 ----
eng = mk(3)
seq = eng.add_request(PROMPT, G)
while not seq.is_finished:
    eng.step()
c = list(seq.token_ids[seq.num_prompt_tokens:])
eng.destroy()
print(f"  C) 投机 K=3             → 位置97={c[41]}  位置98={c[42]}", flush=True)

print()
print(f"  判定：A={a_tok}  B={b_ids[1]}  C={c[42]}   基线={base[42]}")
if a_tok == b_ids[1] == base[42] and c[42] != a_tok:
    # ⚠️ 这一步只能得出"投机路径与另外两条路径不同"，**不能**据此判为账本 bug：
    #    另一种可能是两个 attention kernel 的 bf16 数值差异翻转了 argmax。
    #    区分二者要靠下一步 —— 对拍同一步的 top-3 logits（见 compare_step.py）。
    #    （本项目实际就是后者：compare_step.py 显示 13@+20.250 与 11@+20.250 完全并列）
    print("  ⇒ A 与 B 一致、C 偏离：两条基准路径都回 13，只有投机回 11")
    print("     但这还不足以判定成因 —— 需对拍 top-3 logits（见 compare_step.py）")
elif a_tok == c[42] != b_ids[1]:
    print("  ⇒ 规范参考与投机一致、decode 核偏离 → 指向 kernel 数值差异")
else:
    print("  ⇒ 三者不一致，需要更细的 logits 比对")
