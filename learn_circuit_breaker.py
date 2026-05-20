
import time
import random
from typing import Callable

# === Circuit Breaker ===


class CircuitBreaker:
    """Circuit breaker pattern for failing services."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""

        # Check if circuit should move from open to half-open
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)

            # Success - reset on half-open
            if self.state == "half-open":
                self.state = "closed"
                self.failures = 0

            return result

        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()

            if self.failures >= self.failure_threshold:
                self.state = "open"

            raise e


def demo_circuit_breaker():
    """Demonstrate circuit breaker pattern."""

    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=5.0)

    def flaky_service():
        if random.random() < 0.7:
            raise Exception("Service error")
        return "OK"

    print("\nCircuit Breaker Demo:\n")

    for i in range(15):
        try:
            result = breaker.call(flaky_service)
            print(f"Attempt {i+1}: ✅ {result} (state: {breaker.state})")
        except Exception as e:
            print(f"Attempt {i+1}: ❌ {e} (state: {breaker.state})")

        # After attempt 7, wait long enough for recovery
        if i == 6:
            print("  ⏳ Waiting 6 seconds for recovery timeout...")
            time.sleep(6)
        else:
            time.sleep(0.5)

