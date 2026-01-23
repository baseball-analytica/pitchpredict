# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

from datetime import datetime
import logging
import os

from rich.logging import RichHandler


def init_logger(
    log_dir: str = ".pitchpredict_logs",
    log_level_console: str = "INFO",
    log_level_file: str = "INFO",
) -> None:
    """
    Initialize the logger for PitchPredict.
    """
    # validate log level params
    _validate_log_level(log_level_console)
    _validate_log_level(log_level_file)

    # ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # setup file handler
    file_handler = logging.FileHandler(
        f"{log_dir}/{datetime.now().strftime('%Y-%m-%d')}.log"
    )
    file_handler.setLevel(log_level_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] :: %(message)s")
    )

    # setup console handler
    console_handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=True,
        markup=True,
        tracebacks_show_locals=True,
    )
    console_handler.setLevel(log_level_console)

    # configure the PitchPredict logger
    pp_logger = logging.getLogger("pitchpredict")
    pp_logger.setLevel(logging.DEBUG)
    pp_logger.propagate = False  # prevent double logging
    pp_logger.handlers.clear()  # clear any existing handlers
    pp_logger.addHandler(file_handler)
    pp_logger.addHandler(console_handler)


def _validate_log_level(log_level: str) -> None:
    """
    Ensure the given log level is one of "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL".
    """
    LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level not in LEVELS:
        raise ValueError(f"unrecognized log level: {log_level}")
