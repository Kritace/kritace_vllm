"""逐步对比：找出投机从哪一步开始偏离基线。

不比较最终序列（那只能给出"第 70 个 token 不同"），而是**逐步记录**：
   - 基线：每步产出的 token
   - 投机：每步的草稿、接受数 j、实际产出

这样能直接看到：是"第一次就错"（位置对齐错），
还是"前 N 步对、之后错"（KV 账本随步数累积偏移）。
"""
import json
import sys

sys.path.insert(0, "/root/dzyy-vllm/vllm-v6")

from src import Engine, EngineConfig, SamplingParams

MODEL = "/root/huggingface/Qwen3-0.6B"
PROMPT = (
    "甲乙丙丁戊己庚辛壬癸" * 4
    + "\n请继续输出同样长度的内容：\n"
    + "甲乙丙丁戊己庚辛壬癸" * 2
)
N = 48


def run(k: int, out_path: str) -> None:
    cfg = EngineConfig(
        model=MODEL, enforce_eager=True, dtype="bfloat16",
        context_len=4096, max_num_seqs=4, max_num_batched_tokens=4096,
        prefix_backend="none", gpu_memory_utilization=0.6,
        num_kvcache_blocks=128,
        spec_draft="ngram" if k > 0 else "",
        num_speculative_tokens=k, spec_ngram_max_n=4,
    )
    engine = Engine(cfg)
    seq = engine.add_request(
        PROMPT, SamplingParams(temperature=0.0, max_tokens=N, ignore_eos=True)
    )
    prompt_len = seq.num_prompt_tokens
    trace = []
    while not seq.is_finished:
        before = len(seq.token_ids)
        engine.step()
        after = len(seq.token_ids)
        trace.append(
            {
                "n_new": after - before,
                "draft": list(seq.spec_draft_ids),
                "accepted": seq.spec_accepted,
                "new_ids": list(seq.token_ids[before:after]),
            }
        )
    engine.destroy()
    json.dump(
        {"k": k, "prompt_len": prompt_len, "token_ids": list(seq.token_ids),
         "trace": trace},
        open(out_path, "w"),
    )
    print(f"  k={k} prompt={prompt_len} steps={len(trace)}", flush=True)


if __name__ == "__main__":
    run(int(sys.argv[1]), sys.argv[2])
