"""
Logging configuration for PIR Engine.

Provides a centralized logging system that users can configure as needed.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


# Create package-level logger
logger = logging.getLogger("pir")


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    output_file: Optional[str] = None,
) -> None:
    """
    Configure logging for PIR Engine.
    
    Args:
        level: Logging level (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        output_file: Optional file path to write logs to
    
    Examples:
        >>> from physics_engine import setup_logging
        >>> import logging
        >>> 
        >>> # Basic setup with INFO level
        >>> setup_logging(logging.INFO)
        >>> 
        >>> # Verbose debugging
        >>> setup_logging(logging.DEBUG)
        >>> 
        >>> # Log to file
        >>> setup_logging(logging.INFO, output_file="pir.log")
    """
    if format_string is None:
        format_string = "[%(levelname)s] %(name)s: %(message)s"
    
    # Configure the logger
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)
    
    # File handler if requested
    if output_file:
        file_handler = logging.FileHandler(output_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
        ))
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Name for the logger (typically __name__)
    
    Returns:
        Logger instance
    
    Examples:
        >>> from physics_engine.logging_config import get_logger
        >>> 
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting discovery process")
        >>> logger.debug("Detailed debug information")
    """
    return logging.getLogger(f"pir.{name}")


# Default setup with INFO level
setup_logging(logging.INFO)


__all__ = [
    "logger",
    "setup_logging",
    "get_logger",
]
