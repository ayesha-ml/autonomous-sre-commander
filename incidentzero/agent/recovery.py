from __future__ import annotations
import time
from collections.abc import Callable
from typing import TypeVar
from incidentzero.model.errors import TransientModelError

T = TypeVar("T")

class RetryPolicy:
    def __init__(self, max_attempts: int = 3, sleeper: Callable[[float], None] = time.sleep) -> None:
        self.max_attempts = max_attempts
        self.sleeper = sleeper

    def call_model(self, fn: Callable[[], T]) -> T:
        attempt = 0
        while True:
            try:
                return fn()
            except TransientModelError:
                attempt += 1
                # re-raising error when reaching attempt limit
                if attempt >= self.max_attempts:
                    raise

                # calculating exponential delay and sleeping before retrying
                delay = 1.0 * (2 ** (attempt - 1))
                self.sleeper(delay)