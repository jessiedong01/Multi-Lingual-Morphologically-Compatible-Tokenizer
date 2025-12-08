import os
import socket
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence

import torch
import torch.multiprocessing as mp

try:
    import torch.distributed as dist
except ImportError:  # pragma: no cover - torch without distributed build
    dist = None


@dataclass
class DistributedContext:
    """Lightweight container describing the current distributed runtime."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = torch.device("cpu")
    initialized: bool = False
    backend: Optional[str] = None

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.initialized and self.world_size > 1

    def barrier(self) -> None:
        if self.is_distributed and dist is not None:
            dist.barrier()

    def log(self, message: str) -> None:
        prefix = f"[rank {self.rank}/{self.world_size}]"
        if self.is_primary or os.environ.get("TOKENIZER_DIST_DEBUG"):
            print(f"{prefix} {message}")

    @classmethod
    def standalone(cls, device: Optional[torch.device] = None) -> "DistributedContext":
        resolved = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls(device=resolved)


def _resolve_devices(raw: Optional[Sequence[str]]) -> List[str]:
    if not raw:
        default = "cuda" if torch.cuda.is_available() else "cpu"
        return [default]
    if len(raw) == 1 and raw[0].lower() == "auto":
        if torch.cuda.is_available():
            return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]
        return ["cpu"]
    return [entry.strip() for entry in raw if entry.strip()]


def shard_indices(total: int, ctx: DistributedContext) -> List[int]:
    indices = list(range(total))
    if not ctx.is_distributed:
        return indices
    return indices[ctx.rank :: ctx.world_size]


def all_gather_object(data: Any, ctx: DistributedContext) -> List[Any]:
    if not ctx.is_distributed or dist is None:
        return [data]
    gathered: List[Any] = [None for _ in range(ctx.world_size)]
    dist.all_gather_object(gathered, data)
    return gathered


def broadcast_object(data: Any, ctx: DistributedContext, src: int = 0) -> Any:
    if not ctx.is_distributed or dist is None:
        return data
    payload = [data if ctx.rank == src else None]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def merge_sets(local_set: Iterable[Any], ctx: DistributedContext) -> set:
    merged = set(local_set)
    if not ctx.is_distributed or dist is None:
        return merged
    gathered = all_gather_object(merged, ctx)
    union = set()
    for chunk in gathered:
        union |= set(chunk)
    return union


def _distributed_worker(
    local_rank: int,
    devices: Sequence[str],
    backend: str,
    target: Callable[[DistributedContext], Any],
    args: Sequence[Any],
    kwargs: dict,
    return_list,
) -> None:
    device = torch.device(devices[local_rank])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    global_rank = int(os.environ.get("RANK", local_rank))
    world_size = len(devices)
    dist.init_process_group(backend=backend, rank=global_rank, world_size=world_size)
    ctx = DistributedContext(
        rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        initialized=True,
        backend=backend,
    )
    try:
        result = target(ctx, *args, **kwargs)
        if return_list is not None and ctx.is_primary:
            return_list.append(result)
    finally:
        dist.destroy_process_group()


def launch_distributed(
    target: Callable[[DistributedContext], Any],
    devices: Optional[Sequence[str]] = None,
    *,
    backend: Optional[str] = None,
    args: Sequence[Any] = (),
    kwargs: Optional[dict] = None,
    return_result: bool = False,
) -> Any:
    """Launch ``target`` across the requested devices and return rank-0 output."""

    devices = _resolve_devices(devices)
    kwargs = kwargs or {}
    if len(devices) <= 1:
        ctx = DistributedContext.standalone(torch.device(devices[0]))
        return target(ctx, *args, **kwargs)

    if dist is None:
        raise RuntimeError("torch.distributed is not available but multiple devices were requested.")

    inferred_backend = backend
    if inferred_backend is None:
        inferred_backend = "nccl" if any("cuda" in dev for dev in devices) else "gloo"

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in os.environ:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            os.environ["MASTER_PORT"] = str(sock.getsockname()[1])

    manager = mp.Manager() if return_result else None
    return_list = manager.list() if manager else None

    mp.spawn(
        _distributed_worker,
        args=(devices, inferred_backend, target, args, kwargs, return_list),
        nprocs=len(devices),
        join=True,
    )

    if return_list:
        return return_list[0] if return_list else None
    return None
