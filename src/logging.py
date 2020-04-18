import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import List

FORMATTER = logging.Formatter("[%(asctime)s - %(levelname)s - %(name)s] %(message)s")

handlers: List[logging.Handler] = []
level = None


def initialize(experiment=None, debug=False, std=False):
    """Initialize the logging module.

    It can be called before any logger is created to change the default arguments.
    """
    if experiment is not None:
        _initialize_handler_file(experiment)
        if std:
            _initialize_handler_std()
    else:
        _initialize_handler_std()

    _initialize_level(debug)


def _initialize_handler_file(experiment):
    directory = os.path.join(
        "logging/" + experiment, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    os.makedirs(directory, exist_ok=True)
    file_name = os.path.join(directory, "experiment.log")

    # Create a handler_file that rotate files each 512mb
    handler_file = RotatingFileHandler(
        file_name, mode="a", maxBytes=536_870_912, backupCount=4, encoding=None
    )
    handler_file.setFormatter(FORMATTER)
    handlers.append(handler_file)


def _initialize_handler_std():
    handler_std = logging.StreamHandler()
    handler_std.setFormatter(FORMATTER)
    handlers.append(handler_std)


def _initialize_level(debug):
    global level
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO


def create_logger(name: str) -> logging.Logger:
    """Create a logger with default configuration and FORMATTER."""
    initialized = level is not None and len(handlers) > 0
    if not initialized:
        initialize()

    logger = logging.getLogger(name)
    logger.setLevel(level)  # type: ignore

    for handler in handlers:
        logger.addHandler(handler)

    return logger
