# ============================================================================
# 【本文件是做什么的】把 distributed/ 这一层的**对外接口**汇总到一个命名空间里。
#
# 【为什么要这么写】v4 的并行代码分成四个文件（进程组 / 通信原语 / 控制通道 /
#   PP 工具），但上层（Worker、Linear、ModelRunner）只该关心"有哪些能力"，
#   不该关心它们分别放在哪个文件里。所以这里统一 re-export。
#
# 【这一层的四个能力面】对应下面四段 import：
#     communication_op —— TP 的集合通信（all_reduce / all_gather / reduce_scatter）
#     与 PP 的点对点（send_tensors / recv_tensors）
#     control_channel  —— driver ↔ worker 的字节通道
#     parallel_state   —— TP / PP 进程组的建立、销毁、getter，以及图捕获前后的对齐
#     utils            —— PP 的层区间划分、跨 stage 激活容器、占位层
#
# 【`__all__` 的作用】它是这一层的**对外契约**：包外代码只用列出的名字。
#   注意 rufi/静态检查会据此判断"未使用导入"，所以新增对外符号时要同步补进来。
# ============================================================================

from __future__ import annotations

from .communication_op import (
    all_gather,
    all_reduce,
    recv_tensors,
    reduce_scatter,
    send_tensors,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from .control_channel import ControlChannel
from .parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    # v5 新增的三个 EP 名字：
    #     expert_parallel_enabled  —— 查"当前是否 EP"（MoE 侧决定 all-reduce 顺序）
    #     get_ep_group             —— 拿 EP 组（EP 时就是 TP 组，否则 size=1 占位组）
    #     set_expert_parallel      —— 唯一的下发开关（由 Worker 在初始化时调用）
    # 放在这里统一导出，MoE 侧就不需要知道它们住在 parallel_state 里。
    expert_parallel_enabled,
    get_ep_group,
    get_pp_group,
    get_tp_group,
    get_world_group,
    graph_capture,
    init_distributed_environment,
    initialize_model_parallel,
    is_initialized,
    set_expert_parallel,
    set_nccl_env,
)
from .utils import (
    IntermediateTensors,
    PPMissingLayer,
    divide,
    ensure_divisibility,
    get_pp_indices,
    set_pp_layer_ranges,
)

# 对外契约：包外只用这里列出的名字。
__all__ = [
    "all_gather",
    "all_reduce",
    "recv_tensors",
    "reduce_scatter",
    "send_tensors",
    "tensor_model_parallel_all_gather",
    "tensor_model_parallel_all_reduce",
    "tensor_model_parallel_reduce_scatter",
    "expert_parallel_enabled",
    "get_ep_group",
    "get_pp_group",
    "get_tp_group",
    "get_world_group",
    "set_expert_parallel",
    "ControlChannel",
    "graph_capture",
    "init_distributed_environment",
    "initialize_model_parallel",
    "is_initialized",
    "destroy_distributed_environment",
    "destroy_model_parallel",
    "set_nccl_env",
    "IntermediateTensors",
    "PPMissingLayer",
    "divide",
    "ensure_divisibility",
    "get_pp_indices",
    "set_pp_layer_ranges",
]
