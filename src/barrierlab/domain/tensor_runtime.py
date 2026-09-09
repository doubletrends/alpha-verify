"""One numerical runtime for Stages 1–3: PyTorch on CPU or CUDA."""
from __future__ import annotations

import numpy as np
import torch

_DEVICE = torch.device("cpu")


def configure(cuda: bool) -> torch.device:
    global _DEVICE
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("--cuda requested but no CUDA device is available")
    _DEVICE = torch.device("cuda" if cuda else "cpu")
    return _DEVICE


def device() -> torch.device:
    return _DEVICE


def tensor(value, *, dtype=torch.float64) -> torch.Tensor:
    # pandas and safetensors may expose read-only NumPy views. Own the I/O-boundary
    # copy before entering kernels, which never mutate their source tensors.
    return torch.as_tensor(np.array(value, copy=True), dtype=dtype, device=_DEVICE)
