"""
logger.py

Author: Zane Milo Deso
Filename: logger.py
Created: 2025-03-02
Purpose: Provides a configurable logging setup with both a rotating file handler and a console handler.
         Supports configurable log levels and file paths via environment variables.

Features:
- Log to both a file and the console.
- Automatic log rotation to prevent excessive file size.
- Configurable log level and file path.
- Exception handling with traceback logging.
- Separate formats for file and console logging.
- Prevents duplicate handlers when imported multiple times.
- Optional `force` parameter to reconfigure logging (Python 3.8+).

Usage:
    import logger
    logger.setup_logging()

License: MIT License
"""

import logging
import os
from logging.handlers import RotatingFileHandler

# Internal flag to prevent duplicate handler setup
_configured = False


def setup_logging(
    log_file=None,
    log_level=None,
    max_bytes=5 * 1024 * 1024,  # 5 MB default maximum log file size
    backup_count=5,
    force=False
) -> logging.Logger:
    """
    Set up logging configuration with a rotating file handler and a console handler.

    This function configures the logging system to output logs to both a file and the console.
    It supports log file rotation to prevent indefinite growth of the log file. Log file path and
    log level can be specified either via function parameters or through environment variables.

    Parameters:
        log_file (str, optional): Path to the log file. Defaults to env var 'LOG_FILE' or "logs/system.log".
        log_level (str, optional): Logging level as a string (e.g., "INFO", "DEBUG").
                                   Defaults to env var 'LOG_LEVEL' or "INFO".
        max_bytes (int, optional): Maximum size in bytes of a log file before it gets rotated. Defaults to 5 MB.
        backup_count (int, optional): Number of backup log files to keep. Defaults to 5.
        force (bool, optional): If True, forcibly reconfigures logging even if already configured.

    Returns:
        logging.Logger: Configured logger instance for the current module.
    """
    global _configured
    if _configured and not force:
        return logging.getLogger(__name__)

    # Use environment variables if parameters are not provided.
    log_file = log_file or os.getenv("LOG_FILE", "logs/system.log")
    level_str = (log_level or os.getenv("LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_str, None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid LOG_LEVEL: {level_str}")

    # Ensure that the log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Create a rotating file handler for logging to a file.
    file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))

    # Create a stream handler for logging to the console.
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    ))

    # Configure the root logger with both handlers.
    logging.basicConfig(level=level, handlers=[file_handler, console_handler], force=force)

    _configured = True
    logger = logging.getLogger(__name__)
    logger.debug("Logging configured", extra={"log_file": log_file, "level": level_str})
    return logger


if __name__ == "__main__":
    # Initialize logging for standalone testing.
    log = setup_logging(log_level="DEBUG", force=True)

    # Log messages at various levels to demonstrate the logging configuration.
    log.debug("This is a debug message.")
    log.info("This is an info message.")
    log.warning("This is a warning message.")
    log.error("This is an error message.")

    # Demonstrate logging of an exception with a traceback.
    try:
        1 / 0
    except ZeroDivisionError:
        log.exception("An exception occurred")
