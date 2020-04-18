import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

FORMATTER = logging.Formatter("[%(asctime)s - %(levelname)s - %(name)s] %(message)s")

handler = None
level = None
handler_std = None


def initialize(experiment=None, debug=False, std=False):
    """Initialize the logging module.

    It can be called before any logger is created to change the default arguments.
    """
    if experiment is not None:
        directory = os.path.join(
            "logging/" + experiment, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        os.makedirs(directory, exist_ok=True)
        file_name = os.path.join(directory, "experiment.log")
        _initialize_handler(file_name, std)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(FORMATTER)

    _initialize_level(debug)


def _initialize_handler(file_name, std):
    global handler
    global handler_std

    # Create a handler that rotate files each 512mb
    handler = RotatingFileHandler(
        file_name, mode="a", maxBytes=536_870_912, backupCount=4, encoding=None
    )
    handler.setFormatter(FORMATTER)

    if std:
        handler_std = logging.StreamHandler()
        handler_std.setFormatter(FORMATTER)


def _initialize_level(debug):
    global level
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO


def create_logger(name: str) -> logging.Logger:
    """Create a logger with default configuration and FORMATTER."""
    initialized = level is not None and handler is not None
    if not initialized:
        initialize()

    logger = logging.getLogger(name)
    logger.setLevel(level)  # type: ignore

    logger.addHandler(handler)  # type: ignore
    if handler_std is not None:
        logger.addHandler(handler_std)

    return logger
