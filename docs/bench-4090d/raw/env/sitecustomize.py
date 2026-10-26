"""torch 版本兼容垫片（不改项目源码）。

项目 v4 用了 torch.distributed.all_gather_single，这是新版 PyTorch 才有的
公开别名；其语义/签名与长期存在的 all_gather_into_tensor 完全一致
（官方 docstring 相同：Gather tensors from all ranks and put them in a
single output tensor）。这里在解释器启动时补上这个别名。

由 PYTHONPATH=/root/compat 生效。
"""
try:
    import torch.distributed as _d
    if not hasattr(_d, "all_gather_single"):
        _d.all_gather_single = _d.all_gather_into_tensor
except Exception:
    pass          # 没装 torch 的进程不受影响
