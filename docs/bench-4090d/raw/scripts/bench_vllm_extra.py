"""官方 vLLM 扩展对照：量化（AWQ-INT4）/ 投机解码。

独立脚本，**不改项目代码**。指标口径与 vllm-v4/bench/bench_vllm.py 对齐
（TTFT / decode / batch8），结果追加到同一个 bench.jsonl，方便统一汇总。

投机解码要额外跑 `--prompt real`：伪随机 dummy token 会让 draft 的接受率
严重失真，必须用真实文本才反映实际收益。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

BENCH = Path("/root/dzyy-vllm/vllm-v4/bench")
sys.path.insert(0, str(BENCH))
from common import (  # noqa: E402
    BATCH_SIZE,
    MAX_TOKENS,
    PROMPT_LEN,
    append_jsonl,
    base_record,
    dummy_ids,
    mode_name,
)

REAL_TEXT = (
    "vLLM 是一个高吞吐、低延迟的大语言模型推理与服务引擎。它的核心思想是把操作系统里的"
    "虚拟内存分页机制搬到 KV cache 上：把每个序列的 KV cache 切成固定大小的 block，"
    "用一张 block table 把逻辑块映射到物理块。这样多个序列可以共享同一份物理块，"
    "前缀相同的请求不必重复计算。调度器每跑完一步就重新组批，把新来的请求插进来、"
    "把结束的请求踢出去，这就是连续批处理。连续批处理把 GPU 从静态批处理的空转里解放出来，"
    "是这类推理框架相比传统逐请求推理最大的差异。解码阶段每一小步都要把全部模型权重"
    "从显存读一遍，所以它受限于显存带宽而不是算力；推测解码用一个小草稿模型先猜若干 token，"
    "再用大模型一次前向并行验证，把这一步的权重读取成本摊薄到多个 token 上。"
) * 4


def real_prompt_ids(tokenizer, n: int, k: int) -> list[list[int]]:
    """生成 k 个等长但起点不同的真实文本 prompt（避免 prefix cache 全命中）。"""
    ids = tokenizer.encode(REAL_TEXT)
    need = n + k + 8
    if len(ids) < need:
        ids = ids * (need // max(1, len(ids)) + 1)
    return [ids[i : i + n] for i in range(k)]


def make_llm(args):
    from vllm import LLM

    kw: dict = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        enforce_eager=not args.cg,
        dtype="bfloat16",
        # 投机解码要同时装 target 与 draft 两个模型，两者都要 KV cache，
        # 24G 卡上 8B + 0.6B 必须把 max_model_len 压下来，否则 vLLM 会
        # 直接拒绝启动（"estimated maximum model length is 2368"）。
        max_model_len=args.max_model_len,
        max_num_seqs=16,
        gpu_memory_utilization=args.gpu_mem,
        trust_remote_code=True,
    )
    if args.quant:
        kw["quantization"] = args.quant
    if args.draft:
        kw["speculative_config"] = {
            "model": args.draft,
            "num_speculative_tokens": args.num_spec,
        }
    try:
        return LLM(**kw)
    except TypeError:  # 旧版 API 回退
        if not args.draft:
            raise
        kw.pop("speculative_config")
        kw["speculative_model"] = args.draft
        kw["num_speculative_tokens"] = args.num_spec
        return LLM(**kw)


def warmup(llm, SamplingParams, prompts, *, rounds: int = 2) -> None:
    for _ in range(rounds):
        llm.generate(
            prompts=[{"prompt_token_ids": prompts[0]}],
            sampling_params=SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True),
        )
    torch.cuda.synchronize()


def measure(llm, SamplingParams, prompts) -> tuple[float, float, float]:
    p0 = prompts[0]
    t0 = time.perf_counter()
    llm.generate(
        prompts=[{"prompt_token_ids": p0}],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
    )
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0

    t0 = time.perf_counter()
    llm.generate(
        prompts=[{"prompt_token_ids": p0}],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True),
    )
    torch.cuda.synchronize()
    e2e = time.perf_counter() - t0
    decode_tps = (MAX_TOKENS - 1) / (e2e - ttft) if e2e > ttft else 0.0

    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = llm.generate(prompts=[{"prompt_token_ids": p} for p in prompts[:BATCH_SIZE]], sampling_params=sp)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    n = sum(len(o.outputs[0].token_ids) for o in outs)
    return ttft * 1000.0, decode_tps, (n / wall if wall > 0 else 0.0)


def main() -> int:
    p = argparse.ArgumentParser(description="official vLLM extra: quant / spec-decode")
    p.add_argument("--model", required=True)
    p.add_argument("--draft", default=None)
    p.add_argument("--num-spec", type=int, default=5)
    p.add_argument("--quant", default=None)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--cg", action="store_true")
    p.add_argument("--cuda-devices", default=None)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=4096,
                   help="投机时 24G 卡装不下两个模型的 4096 上下文，需要调小")
    p.add_argument("--prompt", choices=["dummy", "real"], default="dummy")
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    if args.cuda_devices:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    mode = mode_name(cg=bool(args.cg), compile=False)
    rec = base_record(
        engine="vllm-extra", model=args.model, tp=args.tp, pp=1, dp=1, mode=mode, tag=args.tag
    )
    print(
        f"=== {args.tag}: {Path(args.model).name} tp={args.tp} mode={mode} "
        f"quant={args.quant} draft={args.draft} prompt={args.prompt} ===",
        flush=True,
    )

    try:
        from vllm import SamplingParams

        llm = make_llm(args)
        if args.prompt == "real":
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            prompts = real_prompt_ids(tok, PROMPT_LEN, BATCH_SIZE)
        else:
            prompts = [dummy_ids(PROMPT_LEN, salt=1000 + i) for i in range(BATCH_SIZE)]

        warmup(llm, SamplingParams, prompts)
        ttft, dec, batch = measure(llm, SamplingParams, prompts)
        rec["ttft_ms"] = round(ttft, 2)
        rec["decode_tps"] = round(dec, 1)
        rec["batch_tps"] = round(batch, 1)
        print(f"{args.tag} | ttft={ttft:.2f}ms | decode={dec:.1f} | batch8={batch:.1f}", flush=True)
    except Exception as exc:  # noqa: BLE001
        rec["ok"] = False
        rec["error"] = f"{type(exc).__name__}: {exc}"
        print(f"FAIL {args.tag}: {rec['error']}", flush=True)
        append_jsonl(rec)
        return 1

    append_jsonl(rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
