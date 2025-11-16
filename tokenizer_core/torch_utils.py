import os
from contextlib import contextmanager
from typing import Iterable, Optional, Sequence, Union

import torch

_DEFAULT_DEVICE: Optional[torch.device] = None


def default_device() -> torch.device:
    """Resolve the default torch device (CUDA if available, else CPU)."""
    global _DEFAULT_DEVICE
    if _DEFAULT_DEVICE is None:
        env_override = os.environ.get("TOKENIZER_DEVICE")
        if env_override:
            _DEFAULT_DEVICE = torch.device(env_override)
        else:
            _DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEFAULT_DEVICE


def set_default_device(device: Union[str, torch.device]) -> None:
    """Override the global default device."""
    global _DEFAULT_DEVICE
    _DEFAULT_DEVICE = torch.device(device)


def to_device(tensor: torch.Tensor, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    """Move a tensor to the requested (or default) device."""
    dev = torch.device(device) if device else default_device()
    return tensor.to(dev)


def ensure_tensor(
    data: Union[torch.Tensor, Sequence, float, int],
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    copy: bool = False,
) -> torch.Tensor:
    """Convert arbitrary data to a tensor on the configured device."""
    if isinstance(data, torch.Tensor):
        tensor = data.clone() if copy else data
        if dtype is not None:
            tensor = tensor.to(dtype)
    else:
        tensor = torch.as_tensor(data, dtype=dtype if dtype else None)
    return to_device(tensor, device)


def zeros(shape: Union[int, Sequence[int]], *, dtype: torch.dtype = torch.float32, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    return to_device(torch.zeros(shape, dtype=dtype), device)


def ones(shape: Union[int, Sequence[int]], *, dtype: torch.dtype = torch.float32, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    return to_device(torch.ones(shape, dtype=dtype), device)


def full(
    shape: Union[int, Sequence[int]],
    fill_value: Union[float, int],
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    return to_device(torch.full(shape, fill_value, dtype=dtype), device)


def randn(shape: Union[int, Sequence[int]], *, dtype: torch.dtype = torch.float32, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    return to_device(torch.randn(shape, dtype=dtype), device)


def arange(
    start: int,
    end: Optional[int] = None,
    *,
    step: int = 1,
    dtype: torch.dtype = torch.float32,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    if end is None:
        start, end = 0, start
    return to_device(torch.arange(start, end, step=step, dtype=dtype), device)


def eye(n: int, *, dtype: torch.dtype = torch.float32, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    return to_device(torch.eye(n, dtype=dtype), device)


def randperm(n: int, *, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    return to_device(torch.randperm(n), device)


def random_choice(
    population: Union[int, Sequence[int], torch.Tensor],
    size: int,
    *,
    replace: bool = False,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Torch-backed replacement for numpy.random.choice."""
    dev = torch.device(device) if device else default_device()
    if isinstance(population, int):
        if replace:
            return torch.randint(0, population, (size,), device=dev)
        perm = torch.randperm(population, device=dev)
        return perm[:size]
    source = torch.as_tensor(population, device=dev)
    if replace:
        idx = torch.randint(0, source.shape[0], (size,), device=dev)
    else:
        idx = torch.randperm(source.shape[0], device=dev)[:size]
    return source.index_select(0, idx)


@contextmanager
def use_device(device: Union[str, torch.device]):
    """Temporarily override the default device."""
    prev = default_device()
    set_default_device(device)
    try:
        yield
    finally:
        set_default_device(prev)
