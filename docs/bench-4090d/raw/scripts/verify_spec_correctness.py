"""v6 投机解码的真机验收。

判据：**贪心解码下，投机只改变"怎么算"，不改变"算什么"**。
所以关闭投机得到的 token 序列，与开启投机得到的序列，必须完全相同。
这条判据不需要知道任何内部实现，是最硬的一条。

用法：python verify_spec_correctness.py <spec_k> <out.json>
      spec_k=0 表示关闭投机（基线）
"""
import json
import sys

sys.path.insert(0, "/root/dzyy-vllm/vllm-v6")

from src import Engine, EngineConfig, SamplingParams

MODEL = "/root/huggingface/Qwen3-0.6B"
# 刻意用含重复片段的 prompt —— n-gram 草稿要靠重复才能命中。
# 若用普通文本，草稿会一直为空，链路退化成普通解码，测了等于没测。
PROMPT = (
    "甲乙丙丁戊己庚辛壬癸" * 4
    + "\n请继续输出同样长度的内容：\n"
    + "甲乙丙丁戊己庚辛壬癸" * 2
)
N = 48


def main() -> int:
    k = int(sys.argv[1])
    out_path = sys.argv[2]

    cfg = EngineConfig(
        model=MODEL,
        enforce_eager=True,  # 先不开 CUDA Graph：图捕获的形状问题留到下一步
        dtype="bfloat16",
        context_len=4096,
        max_num_seqs=4,
        max_num_batched_tokens=4096,
        prefix_backend="none",
        gpu_memory_utilization=0.6,
        num_kvcache_blocks=128,
        spec_draft="ngram" if k > 0 else "",
        num_speculative_tokens=k,
        spec_ngram_max_n=4,
    )
    engine = Engine(cfg)
    seq = engine.add_request(
        PROMPT, SamplingParams(temperature=0.0, max_tokens=N, ignore_eos=True)
    )

    steps = 0
    while not seq.is_finished:
        engine.step()
        steps += 1

    ids = list(seq.token_ids)
    # ⚠️ 必须显式释放：否则进程退出时 NCCL 进程组不销毁、显存不归还，
    # 下一个紧接着启动的实验会直接 OOM，而且报错指向它自己，极难定位。
    engine.destroy()
    json.dump(
        {
            "k": k,
            "steps": steps,
            "n_tokens": len(ids),
            "last_accepted": getattr(seq, "spec_accepted", 0),
            "token_ids": ids,
        },
        open(out_path, "w"),
    )
    print(f"  k={k} steps={steps} tokens={len(ids)} 完成", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
