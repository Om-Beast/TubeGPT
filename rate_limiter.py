"""Dependency-free per-key sliding-window rate limiter."""
from __future__ import annotations

from collections import OrderedDict
from math import ceil
from threading import Lock
from time import monotonic
from typing import Callable


class SlidingWindowRateLimiter:
    """Bounded in-memory limiter.

    Returns None when a request is allowed, or a retry-after number of seconds
    when the key has exhausted its budget. A single process owns the state.
    """

    def __init__(
        self,
        window_seconds: float = 60.0,
        max_keys: int = 5000,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if max_keys < 1:
            raise ValueError("max_keys must be at least one")
        self.window_seconds = float(window_seconds)
        self.max_keys = int(max_keys)
        self._clock = clock
        self._buckets: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = Lock()

    def check(self, key: str, limit: int) -> int | None:
        """Consume one request slot; otherwise return Retry-After seconds."""
        if limit < 1:
            raise ValueError("limit must be at least one")
        now = self._clock()
        retry_after = None

        with self._lock:
            timestamps = [
                stamp
                for stamp in self._buckets.get(key, [])
                if now - stamp < self.window_seconds
            ]
            if len(timestamps) >= limit:
                retry_after = max(
                    1,
                    ceil(self.window_seconds - (now - timestamps[0])),
                )
            else:
                timestamps.append(now)

            self._buckets[key] = timestamps
            self._buckets.move_to_end(key)
            while len(self._buckets) > self.max_keys:
                self._buckets.popitem(last=False)

        return retry_after

    @property
    def tracked_keys(self) -> int:
        """Expose a read-only count for observability and unit tests."""
        with self._lock:
            return len(self._buckets)
