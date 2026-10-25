# ============================================================================
# 【本文件是做什么的】DP（数据并行）的实现层。dp>1 时，门面进程不再自己持有
# EngineCore，而是拉起 dp_size 个**副本进程** —— 每个副本都是一个完整的引擎，
# 内部再自行展开 TP / PP。
#
# 【DP 为什么不用集合通信】这是 DP 与 TP/PP 的本质区别：
#     TP/PP：同一条请求被切开，各卡**必须互相通信**才能算出结果；
#     DP   ：各副本各跑各的请求，**彼此不需要任何通信**，只是"多开几份"。
# 所以这里的通信只是"门面把请求投递给某个副本"，用 ZMQ 的 IPC 通道就够了，
# 完全不需要 NCCL。DP 的加速比接近线性也是这个原因（见 docs/05-基准与复现.md）。
#
# 【为什么要给每个副本改配置】三件事必须分开，否则副本之间会互相干扰：
#     ① dp 归 1      —— 否则副本自己又会去开副本，递归下去；
#     ② 切 device_ids —— 每个副本只拿属于自己那一段（tp × pp 张）卡；
#     ③ 错开 nccl_port —— 多个副本的 FileStore rendezvous 会撞同一个文件。
#
# 【为什么用 ipc:// 而不是 tcp://】同机进程之间，Unix 域套接字比 TCP 快，
# 而且不占端口、不需要处理端口冲突与防火墙。
#
# 【⚠️ 四个容易踩的点】
#     ① 副本进程必须 `daemon=False` —— **守护进程不允许再有子进程**，
#        而副本内部还要再 spawn Worker。设成 daemon 会在第二层直接失败。
#     ② `ready.wait(timeout=60)` 不能省：子进程初始化失败时，
#        没有超时的等待会让主进程永远卡住（而且没有任何报错）。
#     ③ `_pending` 必须是集合：一条请求的"完成"可能出现在**任意一个**副本的输出里。
#     ④ 空闲时 `sleep(0.0002)`：纯忙等会把一个核跑满，而这一步本来没有活干。
#
# 【读这个文件的顺序】
#     RequestWire / CoreRequest / CoreOutput（跨进程只能传数据）
#       → run_engine_core_proc（副本里跑什么）
#       → MPClient.__init__ / _replica_config（怎么切配置、怎么建通道）
#       → add_request（round-robin 投递）
#       → step / _drain（先等一路、再收齐已就绪的）
#       → destroy（逆序关闭）
# ============================================================================

from __future__ import annotations

import copy
import multiprocessing as mp
import tempfile
import time

import msgspec
import torch
import zmq

from ..config import EngineConfig
from ..data import EngineCoreOutput, Sequence
from .client import EngineCoreClient
from .core import EngineCore

# ==============================================================================
# MPClient：dp>1 时 Engine 到各副本 EngineCore 的调用层。
#
# make_client 在 dp_size>1 时构造本类。之后 Engine 的 add_request / step /
# has_unfinished / destroy 都走这里，不再持有本进程 EngineCore。
#
# 每个副本是独立进程里的完整引擎，
# 经 _replica_config 把 dp 归 1、切开 device_ids 与 nccl_port。
# 门面只做 round-robin 投递和输出汇总；副本内部的 TP/PP 仍由 Executor 展开。
# ==============================================================================


# ------------------------------------------------------------------ #
# 门面 ↔ 副本 的消息
# ------------------------------------------------------------------ #


# 跨进程只能传数据、不能传对象 ⇒ 把 Sequence 需要的字段拷一份过来。
class RequestWire(msgspec.Struct):
    """一次 add_request 所需的字段。跨进程不能传 Sequence 指针，因此拷贝 token 与采样参数。"""

    seq_id: int
    token_ids: list[int]
    block_size: int
    temperature: float
    top_p: float | None
    top_k: int | None
    min_p: float | None
    max_tokens: int
    ignore_eos: bool

    # 两个方向的转换：出站用 from_sequence，副本侧入站用 to_sequence。
    @classmethod
    def from_sequence(cls, seq: Sequence) -> "RequestWire":
        return cls(
            seq_id=seq.seq_id,
            token_ids=list(seq.token_ids),
            block_size=seq.block_size,
            temperature=seq.temperature,
            top_p=seq.top_p,
            top_k=seq.top_k,
            min_p=seq.min_p,
            max_tokens=seq.max_tokens,
            ignore_eos=seq.ignore_eos,
        )

    # 副本侧重建 Sequence。num_prompt_tokens 取 token 总数 —— 请求到达时 prompt 已经完整。
    def to_sequence(self) -> Sequence:
        return Sequence(
            token_ids=list(self.token_ids),
            block_size=self.block_size,
            seq_id=self.seq_id,
            num_prompt_tokens=len(self.token_ids),
            last_token=self.token_ids[-1],
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            max_tokens=self.max_tokens,
            ignore_eos=self.ignore_eos,
        )


# 门面 → 副本：只有 add 与 shutdown 两种操作。
class CoreRequest(msgspec.Struct):
    """门面 → 副本。"""

    op: str  # "add" | "shutdown"
    req: RequestWire | None = None


# 副本 → 门面：本步的增量输出，外加一个统计量。
class CoreOutput(msgspec.Struct):
    """副本 → 门面：本步 EngineCore.step 的产出。"""

    # 只传「增量」—— 跨进程不适合传大对象。
    outputs: list[EngineCoreOutput]
    metric: int = 0


# ------------------------------------------------------------------ #
# 副本进程
# ------------------------------------------------------------------ #


# 一个 DP 副本的主循环：收请求 → 有活就 step → 把结果推回门面。
def run_engine_core_proc(
    config: EngineConfig,
    *,
    eos_token_id: int,
    device: str,
    dtype: torch.dtype,
    in_addr: str,
    out_addr: str,
    ready,  # multiprocessing.Event
) -> None:
    """
        单个 DP 副本：本进程 EngineCore + 一对 ZMQ socket。

        先排空入口上的 add/shutdown，有未完成请求则 step 一次并把结果推给门面；
        队列空时短睡，避免空转占满 CPU。
    """
    # 副本内部绝不能再是 dp>1，否则会递归地开子进程。
    assert config.dp_size == 1, "replica config must flatten dp_size to 1"
    core = EngineCore(
        config, eos_token_id=eos_token_id, device=torch.device(device), dtype=dtype
    )
    # 两个 socket 都由副本 bind：in 收门面推来的请求，out 把结果推回去。
    ctx = zmq.Context()
    in_sock = ctx.socket(zmq.PULL)
    in_sock.bind(in_addr)
    out_sock = ctx.socket(zmq.PUSH)
    out_sock.bind(out_addr)

    # poller 用来「非阻塞地看看入口还有没有积压」。
    poller = zmq.Poller()
    poller.register(in_sock, zmq.POLLIN)
    # 编解码器预先建好。
    enc = msgspec.msgpack.Encoder()
    dec = msgspec.msgpack.Decoder(CoreRequest)

    # 告诉门面「我起来了」—— 否则主进程只能靠 sleep 猜。
    ready.set()
    try:
        # 这样既不被慢副本拖住，也不会漏掉已经就绪的结果。
        while True:
            # 内层循环：把入口积压的命令一次排空（不阻塞），顺便处理 shutdown。
            shutdown = False
            while True:
                socks = dict(poller.poll(0))
                if in_sock not in socks:
                    break
                msg = dec.decode(in_sock.recv())
                # shutdown 优先：即使队列里还有请求也立刻退出。
                if msg.op == "shutdown":
                    shutdown = True
                    break
                if msg.op == "add" and msg.req is not None:
                    core.add_request(msg.req.to_sequence())
            if shutdown:
                break

            # 有活干就 step 一次，把结果推回去。
            if core.has_unfinished():
                outputs, metric = core.step()
                out_sock.send(enc.encode(CoreOutput(outputs=outputs, metric=metric)))
            else:
                time.sleep(0.0002)
    finally:
        core.destroy()
        in_sock.close(0)
        out_sock.close(0)
        ctx.term()


# ------------------------------------------------------------------ #
# 门面客户端
# ------------------------------------------------------------------ #


# 门面侧客户端：管 dp_size 个副本进程 + 轮询投递 + 汇总输出。
class MPClient(EngineCoreClient):
    """
        spawn dp_size 个副本进程，ZMQ IPC 投递请求、汇总各副本的 step 输出。

        add_request 按 round-robin 选副本；step 阻塞等到至少一路输出，再非阻塞收齐当前已就绪的结果。
    """

    # 每个副本一套独立的 socket 对，各自绑定自己的 CUDA 设备。
    def __init__(
        self,
        config: EngineConfig,
        *,
        eos_token_id: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.config = config
        # 这个类只在 dp>1 时被构造。
        self.dp_size = config.dp_size
        assert self.dp_size > 1

        self._ctx = zmq.Context()
        self._enc = msgspec.msgpack.Encoder()
        self._dec = msgspec.msgpack.Decoder(CoreOutput)
        self._in: list[zmq.Socket] = []
        self._out: list[zmq.Socket] = []
        self._poller = zmq.Poller()
        self._procs: list[mp.Process] = []
        self._pending: set[int] = set()
        self._next = 0

        # 用临时目录放 IPC 文件：退出时不必清理端口。
        ctx = mp.get_context("spawn")
        tmp = tempfile.mkdtemp(prefix="kritace_dp_")
        # 每个副本：切好的配置 + 一对 IPC 地址 + 一个 ready 事件。
        for r in range(self.dp_size):
            replica_cfg = self._replica_config(r)
            in_addr = f"ipc://{tmp}/in_{r}.ipc"
            out_addr = f"ipc://{tmp}/out_{r}.ipc"
            ready = ctx.Event()
            # daemon=False 是硬要求：副本内部还要再 spawn Worker。
            p = ctx.Process(
                target=run_engine_core_proc,
                args=(),
                kwargs=dict(
                    config=replica_cfg,
                    eos_token_id=eos_token_id,
                    device=f"cuda:{replica_cfg.device_ids[0]}",
                    dtype=dtype,
                    in_addr=in_addr,
                    out_addr=out_addr,
                    ready=ready,
                ),
                # 副本内 TP/PP 还会再 spawn Worker；daemon 进程不能有子进程。
                # 守护进程不允许有子进程，所以这里必须 False。
                daemon=False,
            )
            p.start()
            self._procs.append(p)
            # 等子进程就绪；超时就报错。
            ready.wait(timeout=60)
            # 不报错的话，后面会卡在「没有对端」的 socket 上，且看不出原因。
            if not ready.is_set():
                raise RuntimeError(f"DP replica {r} failed to initialize")

            # 门面侧反向的 socket：往副本推请求、从副本收结果，并注册进 poller。
            in_sock = self._ctx.socket(zmq.PUSH)
            in_sock.connect(in_addr)
            out_sock = self._ctx.socket(zmq.PULL)
            out_sock.connect(out_addr)
            self._in.append(in_sock)
            self._out.append(out_sock)
            self._poller.register(out_sock, zmq.POLLIN)

    # 第 r 个副本的配置：dp 归 1、切 device_ids、错开 nccl_port。
    def _replica_config(self, r: int) -> EngineConfig:
        """第 r 个副本：dp=1，GPU 切成 tp×pp 一段，nccl_port 错开以免 FileStore 撞车。"""
        # 一个副本要占 tp × pp 张卡。
        per = self.config.tp_size * self.config.pp_size
        # 显式给了 device_ids 就按段切。
        if self.config.device_ids is not None:
            devices = self.config.device_ids[r * per : (r + 1) * per]
            if len(devices) != per:
                # 长度对不上说明配置写错了，直接报错。
                raise ValueError(
                    f"device_ids length {len(self.config.device_ids)} "
                    f"incompatible with dp={self.dp_size} × (tp×pp={per})"
                )
        else:
            start = r * per
            devices = list(range(start, start + per))
        # 复制配置 —— 不能改共享的那一份。
        replica = copy.copy(self.config)
        # dp 必须归 1，否则副本会自己再开副本。
        replica.dp_size = 1
        replica.device_ids = devices
        # nccl_port 每个副本错开 16，避免 rendezvous 文件撞车。
        replica.nccl_port = self.config.nccl_port + r * 16
        return replica

    # 投递一次请求：round-robin 选副本，并把 seq_id 记进待完成集合。
    def add_request(self, seq: Sequence) -> None:
        self._pending.add(seq.seq_id)
        # 轮询指针在 dp_size 上循环 —— 不感知各副本负载，纯粹轮着发。
        r = self._next
        self._next = (self._next + 1) % self.dp_size
        req = CoreRequest(op="add", req=RequestWire.from_sequence(seq))
        self._in[r].send(self._enc.encode(req))

    # 门面的一步 = 把所有已就绪副本的输出收一次。
    def step(self) -> tuple[list[EngineCoreOutput], int]:
        return self._drain(block=True)

    # 只要还有请求没被标完成，就算未完成。
    def has_unfinished(self) -> bool:
        return bool(self._pending)

    # 先阻塞等**至少一路**有输出，再把当前已就绪的全部收完。
    def _drain(self, *, block: bool) -> tuple[list[EngineCoreOutput], int]:
        """block=True 时先等到任意一路有输出，再把当前已就绪的副本结果收完。"""
        outputs: list[EngineCoreOutput] = []
        metric = 0
        # 第一次可以无限等（否则会空转），之后一律非阻塞。
        first = True
        while True:
            timeout = None if (block and first) else 0
            socks = dict(self._poller.poll(timeout))
            if not socks:
                break
            for sock in socks:
                msg = self._dec.decode(sock.recv())
                outputs.extend(msg.outputs)
                metric += msg.metric
                # 谁的输出里出现了跑完的请求，就从待完成集合里划掉。
                for o in msg.outputs:
                    if o.is_finished:
                        self._pending.discard(o.seq_id)
            first = False
        return outputs, metric

    # 关闭顺序：发 shutdown → 收进程 → 关 socket → 终止上下文。
    def destroy(self) -> None:
        # 先广播 shutdown：让副本自己把引擎拆干净（释放 CUDA 与 NCCL）。
        for sock in self._in:
            try:
                sock.send(self._enc.encode(CoreRequest(op="shutdown")))
            except zmq.ZMQError:
                pass
        # 等 15 秒；还活着就强杀。
        for p in self._procs:
            p.join(timeout=15)
            if p.is_alive():
                p.terminate()
        # 收尾：关掉所有 socket，终止 ZMQ 上下文，清空集合。
        for s in self._in:
            s.close(0)
        for s in self._out:
            s.close(0)
        self._ctx.term()
        self._procs.clear()
        self._pending.clear()
