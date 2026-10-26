"""DP=2 的同口径单卡对照（n_reqs=32 聚合吞吐）。

不改项目代码：直接复用项目自己的 EngineConfig，把 dp_size 置 1、
其余参数完全对齐 dp=2 用例（尤其 max_num_batched_tokens=8192，
make_engine 在 dp=1 时会自动降到 4096，那样就不是同口径了）。
"""
import sys
sys.path.insert(0, "/root/dzyy-vllm/vllm-v4")
sys.path.insert(0, "/root/dzyy-vllm/vllm-v4/bench")

import torch
from bench_v4 import _free_tcp_port
from measure import measure_dp_throughput, warmup_dp
from src import Engine, EngineConfig

MODEL = "/root/huggingface/Qwen3-0.6B"
N = 32

for mode, cg in (("eager", False), ("cg", True)):
    engine = Engine(
        EngineConfig(
            model=MODEL, tp_size=1, pp_size=1, dp_size=1,
            enforce_eager=not cg, torch_compile=False, dtype="bfloat16",
            context_len=4096, max_num_seqs=16,
            max_num_batched_tokens=8192,      # 与 dp=2 用例严格一致
            prefix_backend="none", gpu_memory_utilization=0.6,
            num_kvcache_blocks=64, device_ids=[0], pp_layer_counts=None,
            master_addr="127.0.0.1", nccl_port=_free_tcp_port(),
            nccl_ifname="lo",
        )
    )
    try:
        warmup_dp(engine, N)
        tps, n_tok = measure_dp_throughput(engine, N)
        print(f"0.6b-dp1-{mode} | dp=1 | reqs={N} | {n_tok} tok | {tps:.1f} tok/s", flush=True)
    finally:
        engine.destroy()
