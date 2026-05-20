import time
import random
from typing import Callable
from functools import wraps


# === Retry Decorator ===


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (Exception,),
):
    """Retry decorator with exponential backoff."""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2**attempt), max_delay)
                        # Add jitter
                        delay = delay * (0.5 + random.random())
                        print(
                            f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)

            raise last_exception

        return wrapper

    return decorator


@with_retry(max_retries=3, base_delay=1.0)
def unreliable_api_call(query: str) -> str:
    """Simulates an unreliable API."""
    if random.random() < 0.5:
        raise ConnectionError("Simulated API failure")
    return f"Success: {query}"

