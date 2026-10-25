"""验证前缀缓存在这套配置下到底能不能命中（与 ZOS 无关，只是把事实量出来）。"""
import os
import sys

sys.path.insert(0, "/home/Kritace/kritace_vllm")

from src import Engine, EngineConfig  # noqa: E402

SENT = (
    "The history of computing is a history of abstraction. "
    "Each layer hides the one below it, and each hidden layer costs something. "
)


def run(prefix_backend: str, repeat: int) -> None:
    cfg = EngineConfig(
        model=os.path.expanduser("~/huggingface/Qwen3-0.6B"),
        enforce_eager=True,
        max_num_seqs=1,
        num_kvcache_blocks=48,
        prefix_backend=prefix_backend,
    )
    eng = Engine(cfg)
    prompt = SENT * repeat
    n_tok = len(eng.tokenizer(prompt)["input_ids"]) if hasattr(eng, "tokenizer") else -1
    print(f"[{prefix_backend}] repeat={repeat} prompt_tokens~{n_tok}")
    for i in range(2):
        out = list(eng.generate(prompt, max_new_tokens=2, temperature=0.0))
        print(f"  req{i}: cached={eng.last_num_cached_tokens} out={out[:4]}")
    eng.shutdown() if hasattr(eng, "shutdown") else None


if __name__ == "__main__":
    repeat = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    for pb in ("none", "hash"):
        run(pb, repeat)
