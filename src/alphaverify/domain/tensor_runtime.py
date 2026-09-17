"""One numerical runtime for Stages 1–3: PyTorch on CPU or CUDA."""
from __future__ import annotations

import numpy as np
import torch

_DEVICE = torch.device("cpu")
# Share of free CUDA memory a batched kernel may plan to use; a CPU run assumes this fixed allowance.
_CUDA_MEMORY_SHARE = 0.5
_CPU_MEMORY_ALLOWANCE = 4 * 2 ** 30


def configure(cuda: bool) -> torch.device:
    global _DEVICE
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("--cuda requested but no CUDA device is available")
    _DEVICE = torch.device("cuda" if cuda else "cpu")
    return _DEVICE


def device() -> torch.device:
    return _DEVICE


def memory_budget_bytes() -> int:
    """Memory a batched kernel may plan to use on the selected device."""
    if _DEVICE.type != "cuda":
        return _CPU_MEMORY_ALLOWANCE
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info(_DEVICE)
    return int(free * _CUDA_MEMORY_SHARE)


def tensor(value, *, dtype=torch.float64) -> torch.Tensor:
    """A kernel input on the selected device, in ``dtype``."""
    if isinstance(value, torch.Tensor):
        return value.to(device=_DEVICE, dtype=dtype)
    # pandas and safetensors may expose read-only NumPy views. Own the I/O-boundary
    # copy before entering kernels, which never mutate their source tensors.
    return torch.as_tensor(np.array(value, copy=True), dtype=dtype, device=_DEVICE)
