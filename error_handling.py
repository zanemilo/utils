"""
error_handling.py
=================

Author: Zane Milo Deso
Filename: error_handling.py
Created: 2025-03-02

Purpose
-------
Utilities for standardized error handling across your codebase:
- A custom base exception (UtilsError)
- A decorator to gracefully handle exceptions in functions (@handle_errors)
- A context manager to handle exceptions in code blocks (ErrorHandler)
- A retry decorator with backoff and optional jitter (@retry)
- A global exception hook to log unhandled exceptions (setup_global_exception_hook)

Design Notes
------------
- Integrates with your project logger via `logger.setup_logging()` from logger.py
- Uses Python's stdlib `logging` everywhere; no shadowing of names
- Fully type-annotated; safe defaults
- Small, composable utilities you can unit test easily

Usage
-----
    from error_handling import (
        UtilsError, handle_errors, ErrorHandler, retry, setup_global_exception_hook
    )

    @handle_errors(default_return="Error occurred")
    def risky_operation():
        return 1 / 0

    @retry(attempts=3, delay=1, backoff=2, exceptions=(TimeoutError,))
    def network_call():
        return get_data()

    with ErrorHandler(suppress=True):
        do_something_risky()

    setup_global_exception_hook()  # Call once at app startup
"""

from __future__ import annotations

import functools
import logging
import sys
import time
from typing import Callable, Iterable, Optional, Tuple, Type

# Initialize project logging once; safe no-op if already configured.
import logger as app_logger  # your logger.py
app_logger.setup_logging()

LOG = logging.getLogger(__name__)

__all__ = [
    "UtilsError",
    "handle_errors",
    "ErrorHandler",
    "retry",
    "setup_global_exception_hook",
]


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------
class UtilsError(Exception):
    """Base exception class for utilities-related errors."""
    pass


# -----------------------------------------------------------------------------
# Decorators
# -----------------------------------------------------------------------------
def handle_errors(
    default_return: object | None = None,
    log_exception: bool = True,
    use_logger: Optional[logging.Logger] = None,
) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """
    Decorator: catch any exception, optionally log it, and return a default value.

    Parameters
    ----------
    default_return : Any | None
        Value to return when an exception is caught (default: None).
    log_exception : bool
        If True, logs with traceback using the provided logger (default: True).
    use_logger : logging.Logger | None
        Logger to use; defaults to module logger if None.

    Returns
    -------
    Callable
        Wrapped function that handles exceptions.
    """
    log = use_logger or LOG

    def decorator(func: Callable[..., object]) -> Callable[..., object]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> object:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                if log_exception:
                    log.exception("Exception in %s: %s", func.__name__, exc)
                return default_return
        return wrapper

    return decorator


def retry(
    attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    jitter: Optional[Tuple[float, float]] = None,
    use_logger: Optional[logging.Logger] = None,
    on_giveup: Optional[Callable[[BaseException], None]] = None,
) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """
    Decorator: retry a function call on specified exceptions with backoff (and optional jitter).

    Parameters
    ----------
    attempts : int
        Max attempts before giving up (>=1).
    delay : float
        Initial delay between attempts in seconds.
    backoff : float
        Multiplier applied to delay after each failure (e.g., 2.0 doubles the delay).
    exceptions : tuple[type[BaseException], ...]
        Exception types that trigger a retry.
    jitter : tuple[float, float] | None
        Optional (min, max) seconds of random jitter added to the delay each attempt.
    use_logger : logging.Logger | None
        Logger for retry messages; defaults to module logger.
    on_giveup : Callable[[BaseException], None] | None
        Optional callback invoked with the last exception when all retries are exhausted.

    Returns
    -------
    Callable
        Wrapped function that retries on failure.
    """
    log = use_logger or LOG
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    if delay < 0:
        raise ValueError("delay must be >= 0")
    if backoff <= 0:
        raise ValueError("backoff must be > 0")
    if jitter is not None and (len(jitter) != 2 or jitter[0] > jitter[1]):
        raise ValueError("jitter must be a (min, max) tuple with min <= max")

    import random

    def decorator(func: Callable[..., object]) -> Callable[..., object]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> object:
            current_delay = delay
            attempt = 1
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # type: ignore[misc]
                    if attempt >= attempts:
                        log.exception(
                            "Function %s failed after %d attempt(s).",
                            func.__name__, attempt
                        )
                        if on_giveup:
                            on_giveup(exc)
                        raise
                    # compute sleep with optional jitter
                    sleep_for = current_delay
                    if jitter is not None:
                        sleep_for += random.uniform(jitter[0], jitter[1])
                    log.warning(
                        "Function %s failed on attempt %d/%d: %s. Retrying in %.2fs...",
                        func.__name__, attempt, attempts, exc, sleep_for
                    )
                    time.sleep(max(0.0, sleep_for))
                    current_delay *= backoff
                    attempt += 1
        return wrapper

    return decorator


# -----------------------------------------------------------------------------
# Context Manager
# -----------------------------------------------------------------------------
class ErrorHandler:
    """
    Context manager to log exceptions within a block and optionally suppress them.

    Parameters
    ----------
    suppress : bool
        If True, exceptions are suppressed after logging. If False, they propagate.
    use_logger : logging.Logger | None
        Logger to use for exception messages.
    """

    def __init__(self, suppress: bool = True, use_logger: Optional[logging.Logger] = None) -> None:
        self.suppress = suppress
        self.log = use_logger or LOG

    def __enter__(self) -> "ErrorHandler":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            # Use .error with exc_info to avoid double formatting
            self.log.error("Exception in context manager", exc_info=(exc_type, exc_val, exc_tb))
            return self.suppress
        return False


# -----------------------------------------------------------------------------
# Global exception hook
# -----------------------------------------------------------------------------
def setup_global_exception_hook(use_logger: Optional[logging.Logger] = None) -> None:
    """
    Install a global exception hook that logs uncaught exceptions.

    Parameters
    ----------
    use_logger : logging.Logger | None
        Logger to use for logging uncaught exceptions. Defaults to module logger.

    Notes
    -----
    - KeyboardInterrupt is delegated to the default excepthook to allow clean termination.
    """
    log = use_logger or LOG

    def _exception_hook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Preserve default behavior for Ctrl+C
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        log.error("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = _exception_hook


# -----------------------------------------------------------------------------
# Standalone demonstration
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Example: pretty basic logging level change for demo
    app_logger.setup_logging(log_level="DEBUG", force=True)

    @handle_errors(default_return="Default Value")
    def divide(a: float, b: float) -> float | str:
        """Divide two numbers; returns default on error."""
        return a / b

    print("Divide result (should be 'Default Value'):", divide(10, 0))

    # Retry demo
    attempt_counter = {"n": 0}

    @retry(attempts=3, delay=0.5, backoff=2.0, exceptions=(ZeroDivisionError,))
    def unreliable_divide(a: float, b: float) -> float:
        attempt_counter["n"] += 1
        if attempt_counter["n"] < 3:
            raise ZeroDivisionError("Intentional failure for testing retry mechanism.")
        return a / b

    try:
        print("Unreliable divide result (succeeds on 3rd):", unreliable_divide(10, 2))
    except ZeroDivisionError:
        print("unreliable_divide failed after retries.")

    # Context manager demo
    with ErrorHandler(suppress=True):
        _ = 1 / 0  # Will be logged and suppressed

    # Global hook demo (uncomment to see it in action)
    # setup_global_exception_hook()
    raise ValueError("This is an unhandled exception for testing.")
