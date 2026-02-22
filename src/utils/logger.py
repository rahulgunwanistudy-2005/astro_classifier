"""
Structured logger: consistent, color-coded console output + file logging.

Usage:
    logger = get_logger(__name__)
    logger.info("Training started")
    logger.warning("Low GPU memory")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


_COLORS = {
    "DEBUG": "\033[36m",     
    "INFO": "\033[32m",      
    "WARNING": "\033[33m",  
    "ERROR": "\033[31m",     
    "CRITICAL": "\033[35m",  
    "RESET": "\033[0m",
}


class ColoredFormatter(logging.Formatter):
    """Adds color coding to log level names for terminal readability."""

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        color = _COLORS.get(levelname, _COLORS["RESET"])
        reset = _COLORS["RESET"]
        record.levelname = f"{color}{levelname:8s}{reset}"
        return super().format(record)


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str | Path] = None,
) -> logging.Logger:
    """
    Get (or create) a named logger with consistent formatting.

    Args:
        name:     Module name, typically __name__
        level:    Logging level (default: INFO)
        log_file: Optional path to write logs to disk

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if logger was already configured
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Console handler with color
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_fmt = ColoredFormatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # File handler (plain text, no colors)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def setup_root_logger(
    level: int = logging.INFO,
    log_file: Optional[str | Path] = None,
) -> None:
    """Configure the root logger — call once at program entry point."""
    root = logging.getLogger()
    root.setLevel(level)

    # Suppress noisy third-party loggers
    for noisy in ("PIL", "matplotlib", "urllib3", "wandb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if log_file:
        get_logger("root", level=level, log_file=log_file)
