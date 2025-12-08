"""
Utility script to verify the distributed runtime without running a full training job.
"""

import argparse

from tokenizer_core.distributed_utils import all_gather_object, launch_distributed


def _debug_worker(ctx, rounds: int) -> int:
    ctx.log(f"Connected to device {ctx.device} (backend={ctx.backend})")
    payload = ctx.rank + 1
    totals = 0
    for _ in range(rounds):
        gathered = all_gather_object(payload, ctx)
        totals = sum(gathered)
        ctx.barrier()
    if ctx.is_primary:
        ctx.log(f"Gathered payloads {gathered} -> checksum {totals}")
    return totals


def main():
    parser = argparse.ArgumentParser(description="Distributed sanity check.")
    parser.add_argument("--devices", nargs="+", help="Device list or 'auto' to detect GPUs.")
    parser.add_argument("--dist-backend", default=None, help="torch.distributed backend override.")
    parser.add_argument("--rounds", type=int, default=2, help="Number of gather rounds.")
    args = parser.parse_args()

    result = launch_distributed(
        _debug_worker,
        args.devices or ["auto"],
        backend=args.dist_backend,
        args=(args.rounds,),
        return_result=True,
    )
    if result is not None:
        print(f"[debug_distributed] checksum={result}")


if __name__ == "__main__":
    main()
