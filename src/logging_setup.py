"""Logging configuration for Oracle Python Agent.

Call configure_logging() once at application startup (in main.py) before
any other imports that may emit log messages.

Usage in any module::

    import logging
    logger = logging.getLogger(__name__)
"""

import logging
import logging.handlers
import os


def configure_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    """Configure root logger with console and rotating file handlers.

    Args:
        log_dir: Directory for log files. Created if it does not exist.
        level: Logging level for both handlers.
    """
    os.makedirs(log_dir, exist_ok=True)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    # Daily rotating file handler — keeps 7 days of logs
    log_file = os.path.join(log_dir, "oracle.log")
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_file,
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
