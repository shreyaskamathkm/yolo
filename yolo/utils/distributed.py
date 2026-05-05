import os
from contextlib import contextmanager

import torch.distributed as dist


def get_rank() -> int:
    """Returns the rank of the current process."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    """Returns True if the current process is the main process (rank 0)."""
    return get_rank() == 0


@contextmanager
def rank_zero_first():
    """
    Context manager to ensure rank 0 executes a block first.
    Other ranks wait at a barrier until rank 0 is done.
    """
    rank = get_rank()
    initialized = dist.is_available() and dist.is_initialized()

    if initialized and rank > 0:
        dist.barrier()

    yield

    if initialized and rank == 0:
        dist.barrier()
