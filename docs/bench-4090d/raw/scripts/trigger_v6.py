"""找触发条件：让草稿每步返回**真值续写**，于是轨迹与基线重合，每步的 seg 都可对照真值。

对照组：
  R1) 全程真值草稿（应全中）            → seg 是否始终等于真值
  R2) 在 L=97 处注入一次"部分接受"      → 之后 seg 是否开始偏离真值
  R3) 不注入，但强制 K 个草稿里有 1 个错（模拟自然的部分接受）
"""
import re
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
K = 3
_ORIG = NgramDrafter.propose


def mk(k):
    return Engine(EngineConfig(
        model=MODEL, device_ids=[0], tp_size=1, enforce_eager=True,
        max_num_seqs=1, context_len=2048, num_kvcache_blocks=64,
        gpu_memory_utilization=0.85, prefix_backend="none",
        spec_draft="ngram" if k else "", num_speculative_tokens=k,
    ))


# ---- 基线真值 ----
eng = mk(0)
seq = eng.add_request(PROMPT, SamplingParams(temperature=0.0, max_tokens=60, ignore_eos=True))
while not seq.is_finished:
    eng.step()
pl = seq.num_prompt_tokens
prompt_ids = list(seq.token_ids[:pl])
g = list(seq.token_ids[pl:])
eng.destroy()
print(f"  prompt={pl} 真值长度={len(g)}", flush=True)

_CTX = {"pl": pl, "g": g, "corrupt": None}


def gt_propose(self, ids, k):
    """返回真值续写；在指定 L 处把第 2 个元素改错以触发部分接受。"""
    L = len(ids)
    m = L - _CTX["pl"]
    if m < 0 or m + K >= len(_CTX["g"]):
        return []
    d = list(_CTX["g"][m : m + K])
    if _CTX["corrupt"] is not None and L == _CTX["corrupt"]:
        d[1] = 0          # 第 2 个必错 → j=1（部分接受），但轨迹不变
    return d


def run(tag, corrupt):
    _CTX["corrupt"] = corrupt
    NgramDrafter.propose = gt_propose
    eng = mk(K)
    try:
        seq = eng.add_request(PROMPT, SamplingParams(temperature=0.0, max_tokens=45, ignore_eos=True))
        while not seq.is_finished:
            eng.step()
    finally:
        NgramDrafter.propose = _ORIG
        eng.destroy()


# 用 stdout 里的 [spec] 行做对照：正则抓 L / cached / seg / j
import io, contextlib
for tag, corrupt in (("R1 全程真值草稿", None), ("R2 在 L=97 注入部分接受", 97)):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run(tag, corrupt)
    lines = [l for l in buf.getvalue().splitlines() if l.startswith("[spec]")]
    print(f"\n  === {tag} ===  （共 {len(lines)} 个验证步）")
    bad = 0
    for l in lines:
        L = int(re.search(r"L=(\d+)", l).group(1))
        seg = [int(x) for x in re.search(r"seg=\[([^\]]*)\]", l).group(1).split(",") if x.strip()]
        j = int(re.search(r" j=(\d+)", l).group(1))
        m = L - pl
        want = g[m : m + 4]
        ok = seg == want
        if not ok:
            bad += 1
            print(f"    ✗ L={L:3d} j={j} 实得 seg={seg}  真值={want}")
    print(f"    → 不匹配的步数: {bad}/{len(lines)}")
