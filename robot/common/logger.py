"""Centralized logging helpers for robot control."""

from __future__ import annotations

import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Optional, Tuple

_CONFIGURED = False
_FILE_HANDLERS: Dict[Tuple[str, str], logging.Handler] = {}
_DEFAULT_LOG_DIR = Path("logs")
_ROTATION_BYTES = 10 * 1024 * 1024  # 10MB
_BACKUP_COUNT = 5


def _configure_logger() -> None:
    """Configure the root logger once."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _CONFIGURED = True


def _default_log_filename() -> str:
    """Generate timestamp-based log filename in format YYYYMMDD_HHMMSS.log"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(_DEFAULT_LOG_DIR / f"{timestamp}.log")


def _ensure_file_handler(logger: logging.Logger, filename: str) -> None:
    """Attach a file handler to the logger for the given filename."""
    file_path = Path(filename).expanduser()
    key = (logger.name, str(file_path))

    if key in _FILE_HANDLERS:
        return

    if file_path.parent and not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        str(file_path),
        encoding="utf-8",
        maxBytes=_ROTATION_BYTES,
        backupCount=_BACKUP_COUNT,
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    _FILE_HANDLERS[key] = handler


def _remove_logger_file_handlers(logger: logging.Logger) -> None:
    """Detach and close all file handlers registered for the logger."""
    keys_to_remove = [key for key in _FILE_HANDLERS if key[0] == logger.name]
    for key in keys_to_remove:
        handler = _FILE_HANDLERS.pop(key)
        logger.removeHandler(handler)
        handler.close()


def get_logger(
    name: Optional[str] = None,
    filename: Optional[str] = None,
    is_save: bool = True,
) -> logging.Logger:
    """Return a module-level logger, ensuring configuration exists.

    Logs are persisted by default to `logs/YYYYMMDD_HHMMSS.log` unless `is_save=False`.
    Provide `filename` to override the destination.

    Args:
        name: Logger name. Defaults to "robot_control"
        filename: Custom log file path. If None, uses timestamp-based filename
        is_save: Whether to save logs to file. Default is True

    Returns:
        Configured logger instance
    """
    _configure_logger()
    logger_name = name if name else "robot_control"
    logger = logging.getLogger(logger_name)

    if is_save:
        target_filename = filename if filename else _default_log_filename()
        _ensure_file_handler(logger, target_filename)
    else:
        _remove_logger_file_handlers(logger)

    return logger
