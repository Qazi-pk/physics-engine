"""
Queue utilities for large benchmark sweeps.
"""

from __future__ import annotations

from queue import Queue
from typing import Iterable, TypeVar

T = TypeVar("T")


def build_experiment_queue(configs: Iterable[T]) -> Queue:
    """Build a FIFO queue from an iterable of experiment configs."""
    q: Queue = Queue()
    for config in configs:
        q.put(config)
    return q


def drain_queue(q: Queue) -> list[T]:
    """Drain all queue items to a list."""
    items: list[T] = []
    while not q.empty():
        items.append(q.get())
    return items
