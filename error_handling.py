"""
error_handling.py

Author: Zane Milo
Filename: error_handling.py
Created: 2025-03-02
Purpose: Provides utilities for standardized error handling in the utils toolbox.
         This module includes:
         - A custom base exception for utils-related errors.
         - A decorator to gracefully handle exceptions in functions.
         - A context manager to handle exceptions in code blocks.
         - A retry decorator to automatically retry functions on failure.
         - A global exception hook to log unhandled exceptions.

Features:
- Custom exception base class for unified error management.
- @handle_errors decorator to catch, log, and optionally return a default value when exceptions occur.
- ErrorHandler context manager to wrap code blocks, log exceptions, and optionally suppress them.
- @retry decorator to retry a function call upon encountering specified exceptions.
- Global exception hook to capture and log unhandled exceptions.
- Detailed logging for improved traceability and debugging.

Usage:
    from error_handling import handle_errors, ErrorHandler, retry, setup_global_exception_hook, UtilsError

    @handle_errors(default_return="Error occurred")
    def risky_operation():
         # Code that might throw an error.
         return result

    @retry(attempts=3, delay=1, backoff=2)
    def network_call():
         # Code that might fail due to network issues.
         return data

    with ErrorHandler(suppress=True):
         # Code that might throw an error
         do_something()

    # Set up the global exception hook early in your application.
    setup_global_exception_hook()

License: MIT License (or specify another if applicable)
"""

import logging
import sys
import time
import functools

class UtilsError(Exception):
    """
    Base exception class for all errors in the utils module.
    Use this as a parent class for custom exceptions in your utilities.
    """
    pass

def handle_errors(default_return=None, log_exception=True):
    """
    Decorator for handling errors in functions.

    Wraps a function to catch any exceptions, log the error details, and return a default value
    if an exception occurs.

    Parameters:
        default_return: The value to return if an exception is caught. Default is None.
        log_exception (bool): Whether to log the exception with traceback. Default is True.

    Returns:
        A wrapped function that handles exceptions and returns the default value on error.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Attempt to execute the function.
                return func(*args, **kwargs)
            except Exception as e:
                # Log the exception with traceback if logging is enabled.
                if log_exception:
                    logging.exception("Exception in function '%s': %s", func.__name__, e)
                # Return the specified default value when an error occurs.
                return default_return
        return wrapper
    return decorator

class ErrorHandler:
    """
    Context manager for handling errors in a block of code.

    This context manager logs any exception that occurs within the block. It can optionally
    suppress the exception (i.e., prevent it from propagating) or allow it to be re-raised.

    Usage:
        with ErrorHandler(suppress=True):
            # code that might throw an error
            risky_operation()
    """
    def __init__(self, suppress=True):
        """
        Initialize the ErrorHandler.

        Parameters:
            suppress (bool): If True, exceptions are suppressed after logging; if False, they are re-raised.
        """
        self.suppress = suppress

    def __enter__(self):
        # Nothing special needed when entering the context.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # If an exception occurred, log it.
        if exc_type is not None:
            logging.exception("Exception caught in context manager: %s", exc_val)
            # Return True if suppressing the exception, False otherwise.
            return self.suppress
        # No exception occurred.
        return False

def retry(attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, exceptions: tuple = (Exception,), logger: logging.Logger = None):
    """
    Decorator for retrying a function if an exception occurs.

    This decorator will attempt to call the decorated function up to a specified number of times
    if one of the specified exceptions is raised. After each failure, it waits for a given delay,
    which increases by the backoff factor with each subsequent attempt.

    Parameters:
        attempts (int): Maximum number of attempts before giving up.
        delay (float): Initial delay between attempts in seconds.
        backoff (float): Factor by which the delay increases after each failed attempt.
        exceptions (tuple): Tuple of exception classes that should trigger a retry.
        logger (logging.Logger): Logger instance for logging retry attempts. If None, uses the module logger.

    Returns:
        The result of the function if successful; otherwise, re-raises the exception after all attempts fail.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or logging.getLogger(func.__module__)
            attempt = 1
            current_delay = delay  # Use a local variable to track delay without modifying the parameter.
            while attempt <= attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == attempts:
                        _logger.exception("Function %s failed after %d attempts.", func.__name__, attempts)
                        raise
                    else:
                        _logger.warning(
                            "Function %s failed on attempt %d/%d: %s. Retrying in %.1f seconds...",
                            func.__name__, attempt, attempts, str(e), current_delay
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                        attempt += 1
        return wrapper
    return decorator

def setup_global_exception_hook(logger: logging.Logger = None):
    """
    Sets up a global exception hook to log unhandled exceptions.

    This function overrides the default sys.excepthook to log any unhandled exceptions using the specified logger.
    It is useful for capturing errors that are not caught by try/except blocks.

    Parameters:
        logger (logging.Logger, optional): Logger to use for logging exceptions.
            If not provided, uses logging.getLogger(__name__).
    """
    _logger = logger or logging.getLogger(__name__)

    def exception_hook(exc_type, exc_value, exc_traceback):
        # For KeyboardInterrupt, call the default exception hook to allow termination.
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        _logger.error("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    sys.excepthook = exception_hook

# Standalone testing block: runs when this script is executed directly.
if __name__ == "__main__":
    # Configure basic logging for demonstration.
    logging.basicConfig(level=logging.DEBUG)

    # Test the handle_errors decorator.
    @handle_errors(default_return="Default Value")
    def divide(a, b):
        """
        Divide two numbers and return the result.
        Returns a default value if a division error occurs.
        """
        return a / b

    print("Divide result:", divide(10, 0))  # Expected output: "Default Value"

    # Test the retry decorator.
    counter = 0
    
    @retry(attempts=3, delay=1, backoff=2, exceptions=(ZeroDivisionError,))
    def unreliable_divide(a, b):
        """
        Divides two numbers but raises ZeroDivisionError on purpose for testing.
        Succeeds only after a few failed attempts.
        """
        global counter
        counter += 1
        if counter < 3:
            raise ZeroDivisionError("Intentional failure for testing retry mechanism.")
        return a / b

    try:
        print("Unreliable divide result:", unreliable_divide(10, 2))  # Expected to succeed on the 3rd attempt.
    except ZeroDivisionError:
        print("unreliable_divide failed after retries.")

    # Test the ErrorHandler context manager.
    with ErrorHandler(suppress=True):
        # This block will catch and log the exception without stopping execution.
        _ = 1 / 0

    # Set up the global exception hook.
    setup_global_exception_hook()

    # Uncomment the following line to trigger an unhandled exception and see the global hook in action.
    #raise ValueError("This is an unhandled exception for testing.")
