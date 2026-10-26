"""在真实 GPU 上测 n-gram 投机的接受率。

思路：贪心解码下 target 的 argmax 就等于实际生成的 token，
所以"草稿 token 会不会被接受"可以直接在**真实生成的序列**上模拟 ——
不必改引擎、不必暴露 logits、也不必把前向跑两遍。

这样得到的 α 正是 docs/00-投机解码.md 第 7 节收益公式里唯一的未知量：
    E[每步产出] = (1 - α^(K+1)) / (1 - α)
n-gram 的草稿开销接近 0，所以"加速 ≈ E"。
"""
import sys
sys.path.insert(0, "/root/dzyy-vllm/vllm-v6")

from src import Engine, EngineConfig, SamplingParams
from src.spec import NgramDrafter

MODEL = "/root/huggingface/Qwen3-0.6B"
GEN = 256

PROMPTS = {
    "repeat-block": "甲乙丙丁戊己庚辛壬癸" * 6 + "\n请接着重复上面的内容：\n" + "甲乙丙丁戊己庚辛壬癸" * 3,
    "code-like": (
        "def add(a, b):\n    return a + b\n\n\n"
        "def mul(a, b):\n    return a * b\n\n\n"
        "def sub(a, b):\n    return a - b\n\n\n"
    ),
    "prose": "深度学习模型的推理分为预填充与解码两个阶段。预填充阶段并行处理整个输入序列，解码阶段逐个生成输出 token。解码阶段每生成一个 token 都要把模型权重完整读取一遍，因此它受限于显存带宽而不是算力。",
}


def simulate(tokens, prompt_len, *, max_n, k):
    """在真实 token 序列上模拟 n-gram 投机，返回统计量。"""
    d = NgramDrafter(max_n=max_n)
    st = {"steps": 0, "accepted": 0, "draft_total": 0, "no_draft": 0}
    i = prompt_len
    while i < len(tokens) - 1:
        draft = d.propose(tokens[:i], k)
        if not draft:
            st["no_draft"] += 1
            i += 1
            continue
        acc = 0
        for j, tok in enumerate(draft):
            if i + j < len(tokens) and tok == tokens[i + j]:
                acc += 1
            else:
                break
        st["steps"] += 1
        st["accepted"] += acc
        st["draft_total"] += len(draft)
        i += acc + 1
    return st


def expected_tokens(alpha, k):
    # alpha == 1 表示草稿全部被接受，此时每一步都产出 K+1 个 token；
    # 下面那个等比数列公式在 alpha == 1 时是 0/0，必须单独处理。
    if alpha >= 1.0:
        return float(k + 1)
    if alpha <= 0:
        return 1.0
    return (1 - alpha ** (k + 1)) / (1 - alpha)


engine = Engine(
    EngineConfig(
        model=MODEL, enforce_eager=False, dtype="bfloat16",
        context_len=4096, max_num_seqs=4, max_num_batched_tokens=4096,
        prefix_backend="none", gpu_memory_utilization=0.6,
        num_kvcache_blocks=128,
    )
)
print(f"{'prompt':14s} {'K':>2s} {'命中步':>7s} {'无草稿':>7s} {'α':>7s} {'E[token]':>9s}", flush=True)
print("-" * 56, flush=True)

for name, text in PROMPTS.items():
    seq = engine.add_request(
        text, SamplingParams(temperature=0.0, max_tokens=GEN, ignore_eos=True)
    )
    while not seq.is_finished:
        engine.step()
    tokens = list(seq.token_ids)
    plen = seq.num_prompt_tokens
    print(f"[{name}] prompt={plen} gen={len(tokens) - plen}", flush=True)

    for k in (1, 2, 4, 6, 8):
        st = simulate(tokens, plen, max_n=4, k=k)
        alpha = st["accepted"] / st["draft_total"] if st["draft_total"] else 0.0
        print(
            f"{name:14s} {k:2d} {st['steps']:7d} {st['no_draft']:7d} "
            f"{alpha:7.3f} {expected_tokens(alpha, k):9.2f}",
            flush=True,
        )
    print(flush=True)

engine.destroy()
print("DONE", flush=True)
