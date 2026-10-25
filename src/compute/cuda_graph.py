# ============================================================================
# 【本文件是做什么的】把 decode 的**一步前向**录成 CUDA Graph，之后每步只做
# 「改输入内容 + 重放整张图」，一次提交跑完整步。
#
# 【它省掉的不是计算，是 CPU 侧的下发工作】一步 decode 要下发几百个 kernel，
# 每个都要在 Python 侧走一遍（取参数、查 dispatch、建积木）。GPU 算得很快、
# CPU 反而下不过来，于是 GPU 大部分时间在等 CPU。录成图之后，一次 replay
# 就把这几百个 kernel 一口气全推下去。
#
# 【v4 相对 v3 改了什么：从"自己什么都管"变成"只接收一个计算体"】
#     v3：Runner 自己持有 model / Batch，捕获与重放都在它内部完成。
#     v4：只收一个 **档位表** 和一个 **计算体 body(bs)** —— 也就是
#         「给定 batch size，把这一小段前向跑一遍」这个回调。
#   为什么要这么改？因为 PP 引入了**跨进程的收发**：收到的 hidden 不能在图里
#   去 recv（图里不能有主机侧阻塞调用），只能放在图外做。于是结构变成：
#       图外 recv  →  图内 forward（读写固定 buffer）  →  图外 send
#   这就是 piecewise 的含义 —— 把图切成"能在图里跑的部分"。
#
# 【三个必须记住的约束】
#     1. 图里所有张量的**地址**都被写死了。所以输入必须来自预分配缓冲，
#        每步只改内容；输出对象也必须由本类长期持有（重放是往它里面原地写）。
#     2. 捕获期间不能有会新分配显存的算子，否则那块显存落在图的私有池里，
#        下次重放会被覆盖 —— 报错形如 "overwritten by a subsequent run"。
#     3. 录制前必须先热身跑一遍：触发算子编译、惰性初始化，
#        否则这些"第一次"的动作会被录进图里，重放时行为不对。
#
# 【为什么档位是稀疏的】给每个 batch size 都录一张图，启动时间与显存都受不了，
# 所以只录 1/2/4/8/16/…/max_bs；真实 bs 向上取整到最近的档位（多算几条空序列）。
#
# 【⚠️ 别踩】`destroy` 必须显式做 —— 图的显存池不会因为图对象被回收而释放。
#
# 【读这个文件的顺序】
#     __init__（档位表 + 池）
#       → capture（降序录制、共用池、两步走）
#       → replay / _pad_bs（重放 + 向上取整）
#       → make_bs_list（档位怎么定）
# ============================================================================

from __future__ import annotations

from bisect import bisect_left
from typing import Any, Callable

import torch

from ..distributed import graph_capture


# decode 专用。构造它的时候必须传一个已经建好的静态 Batch（输入 buffer 的来源）。
# decode 专用：把一步前向录成图，之后每步只改输入内容、重放整张图。
class CUDAGraphRunner:
    # v4 的形态变了：不再自己持有模型与 Batch，只收一个档位表和一个计算体。
    def __init__(self, bs_list: list[int], device: torch.device):
        # 这样 PP 的收发可以留在图外，图里只放纯计算。
        self.device = device
        # 档位表排好序：后面用二分找「最小的 ≥ 真实 bs 的档位」。
        self.bs_list = sorted(bs_list)
        # 每个档位一张图。
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        # 输出对象也必须按档位长期持有 —— 重放是往它里面原地写，地址不能变。
        self.outputs: dict[int, Any] = {}
        # 所有图共用同一个显存池：各自开池会成倍吃显存。
        self._graph_pool = None

    # 捕获：为每个档位各录一张图。只在启动时做一次。
    def capture(self, body: Callable[[int], Any]) -> None:
        # 起步先同步 + 清缓存，让录图起点的显存状态干净。
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        # 从大到小录：第一张分配显存池，后面的小图复用它。
        for bs in sorted(self.bs_list, reverse=True):
            # 先热身跑一遍：触发算子编译与惰性初始化，把这些「第一次」排除在图外。
            body(bs)  # warmup：初始化算子与 NCCL 集合通信顺序
            # 每个档位一张独立的图对象。
            graph = torch.cuda.CUDAGraph()
            # 录图前后要对齐 TP / PP 各 rank —— 图里有 all-reduce，步调必须一致。
            with graph_capture(self.device):
                # 进了这一层，所有 kernel 都被录进图里，不再真正执行。
                with torch.cuda.graph(graph, pool=self._graph_pool):
                    # 记下输出对象（重放时它会被原地改写）。
                    self.outputs[bs] = body(bs)
            # 第一张图录完就把它的池记下来，后面所有图都复用。
            # 第一张图录完后把它的池记下来，后面所有图都用它。
            if self._graph_pool is None:
                self._graph_pool = graph.pool()
            self.graphs[bs] = graph
        # 整批录完再同步一次。
        torch.cuda.synchronize(self.device)

    # 重放：按档位取图，一次提交跑完整步。
    def replay(self, bs: int) -> tuple[Any, int]:
        # 向上取整到已录制的档位 —— 多算几条空序列，换掉整张图的重录。
        pbs = self._pad_bs(bs)
        self.graphs[pbs].replay()
        # 返回（该档位的输出对象，实际用到的档位）—— 调用方要自己切回真实行数。
        return self.outputs[pbs], pbs

    # 二分找「第一个 ≥ bs 的档位」：向上取整，多算几条空序列换掉整张图的重录。
    # 档位选择：找到第一个 >= bs 的已录档位。
    def _pad_bs(self, bs: int) -> int:
        return self.bs_list[bisect_left(self.bs_list, bs)]

    # 显式清理：图对象和它的显存池都不能靠 GC 兜底。
    # 显式清理：图的显存池不会因为图对象被回收而释放。
    def destroy(self) -> None:
        self.graphs.clear()
        self.outputs.clear()
        self._graph_pool = None

    # 生成录制档位表。
    @staticmethod
    # 档位表：小并发密集、高并发稀疏。
    def make_bs_list(max_bs: int) -> list[int]:
        # 小档位密集（1/2/4/8，覆盖低并发场景），之后每 16 一档（高并发下精度不重要）。
        bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        bs = [b for b in bs if b <= max_bs]
        # 保证最大档位一定存在，否则 can_use 会对齐不到图。
        if max_bs not in bs:
            bs.append(max_bs)
        return bs
