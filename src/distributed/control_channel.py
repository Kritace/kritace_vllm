# ============================================================================
# 【本文件是做什么的】一条**只传字节**的通道：把 driver 侧（调度器所在进程）与
# worker 侧（跑计算的进程）之间的控制消息搬过去 —— 广播一步的调度结果、
# 回传采样出的 token id。
#
# 【为什么需要它】v4 把"调度"与"计算"拆到不同进程（control/ 与 compute/）。
# 两边要交换的东西形状不固定（一次调度结果里有几十个字段），而 torch.distributed
# 只能传**张量**。所以这里的做法是：
#     ① 上层把控制消息序列化成一段 bytes
#     ② 本文件把 bytes 包成 uint8 张量传过去
#     ③ 接收端再还原成 bytes
#
# 【为什么用 Gloo 而不是 NCCL】消息在 CPU 上、而且都很小。
#   Gloo 天生处理 CPU 张量，小消息延迟也更低；NCCL 只擅长 GPU 大张量。
#
# 【⚠️ 协议上必须"先长度、后内容"】接收端不知道要收多少字节，
#   所以每次都先传一个**定长的长度张量**，再按它开缓冲收正文。
#   长度用 0 表示"没有数据"，接收端据此直接返回 None / b""。
#
# 【src / dst 是 WORLD rank】不是组内序号 —— 这条通道建在 WORLD 的 Gloo 组上，
#   跟 TP/PP 那套"组内 rank"口径是两回事，别混。
#
# 【DP 不走这里】DP 的各副本是独立进程组，用 ZMQ 通信（见 control/mp_client.py），
#   与这个通道无关。
# ============================================================================

from __future__ import annotations

import torch
import torch.distributed as dist

# ==============================================================================
# 同 Engine 内 driver ↔ worker 的 Gloo 字节通道。
#
# MultiprocExecutor / worker_loop 用来广播 SchedulerOutput、回传 token id。
# src/dst 是 WORLD rank。DP 的 ZMQ 不走这里。
# ==============================================================================


# bytes → uint8 张量：torch.distributed 只认张量，字节流必须先包一层。
def _cpu_u8(data: bytes) -> torch.Tensor:
    # frombuffer 得到的是只读视图，clone 一份才是可写的独立张量。
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()


# 反向：张量 → 连续内存 → bytes。contiguous 是必须的（numpy() 只认连续内存）。
def _to_bytes(buf: torch.Tensor) -> bytes:
    return buf.contiguous().numpy().tobytes()


class ControlChannel:
    # group 直接用 WORLD 的 Gloo 组 —— 这是**同 Engine 内部** driver ↔ worker 的通道。
    def __init__(self, group) -> None:
        self.group = group

    # broadcast：先广播长度、再广播正文。长度是定长张量，两端不用协商格式。
    def broadcast(self, data: bytes | None, src: int = 0) -> bytes | None:
        # 先取自己的 rank：下面 src 那个进程要走不同分支。
        rank = dist.get_rank(self.group)
        # 长度张量的 dtype 与形状两端必须完全一致，Gloo 对这个很严格。
        size = torch.tensor([len(data) if data is not None else 0], dtype=torch.long)
        # 第一步：广播长度。
        dist.broadcast(size, src=src, group=self.group)
        # 长度只有 src 算得有意义，其它 rank 拿到的是广播过来的值。
        n = int(size.item())
        # 长度为 0 表示「没有数据」，避免空张量参与 broadcast。
        if n == 0:
            return None
        # src 自己也要调 broadcast（集合通信是所有人的事），但它不需要收。
        if rank == src:
            dist.broadcast(_cpu_u8(data), src=src, group=self.group)
            return None
        # 非 src 侧：先按长度开好缓冲，再收正文。
        buf = torch.empty(n, dtype=torch.uint8)
        dist.broadcast(buf, src=src, group=self.group)
        # 返回拷贝出来的 bytes —— buf 是临时张量，不能直接暴露给调用方。
        return _to_bytes(buf)

    # send：点对点，同样是「先长度、后正文」。
    def send(self, data: bytes, dst: int) -> None:
        # 先把长度发过去，接收端据此开缓冲。
        dist.send(
            torch.tensor([len(data)], dtype=torch.long), dst=dst, group=self.group
        )
        # 空 payload 只发长度 0，不再发一次空张量。
        if data:
            dist.send(_cpu_u8(data), dst=dst, group=self.group)

    # recv 与 send 完全对称。
    def recv(self, src: int) -> bytes:
        # 长度先收，才知道要开多大的 buffer。
        size = torch.tensor([0], dtype=torch.long)
        dist.recv(size, src=src, group=self.group)
        n = int(size.item())
        # 长度 0 → 返回空 bytes。
        if n == 0:
            return b""
        # 再收正文。
        buf = torch.empty(n, dtype=torch.uint8)
        dist.recv(buf, src=src, group=self.group)
        # 转成 bytes 返回。
        return _to_bytes(buf)
