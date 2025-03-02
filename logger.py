"""
logger.py

Author: Zane Milo
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

Usage:
    import logger
    logger.setup_logging()

License: MIT License
"""

import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging(
    log_file=None,
    log_level=None,
    max_bytes=5 * 1024 * 1024,  # 5 MB default maximum log file size
    backup_count=5,
):
    """
    Set up logging configuration with a rotating file handler and a console handler.

    This function configures the logging system to output logs to both a file and the console.
    It supports log file rotation to prevent indefinite growth of the log file. Log file path and
    log level can be specified either via function parameters or through environment variables.

    Parameters:
        log_file (str, optional): Path to the log file. Defaults to the value of the environment
                                  variable 'LOG_FILE' or "logs/system.log" if not set.
        log_level (str, optional): Logging level as a string (e.g., "INFO", "DEBUG"). Defaults to
                                   the value of the environment variable 'LOG_LEVEL' or "INFO".
        max_bytes (int, optional): Maximum size in bytes of a log file before it gets rotated.
                                   Defaults to 5 MB.
        backup_count (int, optional): Number of backup log files to keep. Defaults to 5.

    Returns:
        logging.Logger: Configured logger instance for the current module.
    """
    # Use environment variables if parameters are not provided.
    if log_file is None:
        log_file = os.getenv("LOG_FILE", "logs/system.log")
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Ensure that the log directory exists; create it if it doesn't.
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Create a rotating file handler for logging to a file.
    file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    # Define the format for file logs including timestamp, log level, module name, and message.
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(file_formatter)

    # Create a stream handler for logging to the console.
    console_handler = logging.StreamHandler()
    # Define a simpler format for console logs.
    console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)

    # Configure the logging system with the specified log level and both handlers.
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        handlers=[file_handler, console_handler]
    )

    # Create and return a module-level logger for further use.
    logger = logging.getLogger(__name__)
    logger.info("Logger is set up.")
    return logger

# Standalone testing block: runs when this script is executed directly.
if __name__ == "__main__":
    # Initialize the logging system.
    setup_logging()
    # Retrieve the logger instance for the current module.
    logger = logging.getLogger(__name__)

    # Log messages at various levels to demonstrate the logging configuration.
    logger.debug("This is a debug message.")   # Debug messages provide detailed information for diagnosing problems.
    logger.info("This is an info message.")      # Informational messages provide general runtime status.
    logger.warning("This is a warning message.") # Warnings indicate potential issues.
    logger.error("This is an error message.")    # Errors indicate a problem that has occurred.

    # Demonstrate logging of an exception with a traceback.
    try:
        # Deliberate error: division by zero to trigger an exception.
        1 / 0
    except ZeroDivisionError:
        # logger.exception() logs the exception traceback along with the message.
        logger.exception("An exception occurred")
