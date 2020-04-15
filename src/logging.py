import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

handler = None
level = None
handler_std = None


def initialize(experiment_name="experiment", debug=False, std=False):
    """Initialize the logging module.

    It can be called before any logger is created to change the default arguments.
    """
    directory = os.path.join(
        "logging/" + experiment_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    os.makedirs(directory, exist_ok=True)

    file_name = os.path.join(directory, "experiment.log")

    _initialize_level(debug)
    _initialize_handler(file_name, std)


def _initialize_handler(file_name, std):
    global handler
    global handler_std

    formatter = logging.Formatter(
        "[%(asctime)s - %(levelname)s - %(name)s] %(message)s"
    )
    # Create a handler that rotate files each 512mb
    handler = RotatingFileHandler(
        file_name, mode="a", maxBytes=536_870_912, backupCount=4, encoding=None
    )
    handler.setFormatter(formatter)

    if std:
        handler_std = logging.StreamHandler()
        handler_std.setFormatter(formatter)


def _initialize_level(debug):
    global level
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO


def create_logger(name: str) -> logging.Logger:
    """Create a logger with default configuration and formatter."""
    initialized = level is not None and handler is not None
    if not initialized:
        initialize()

    logger = logging.getLogger(name)
    logger.setLevel(level)  # type: ignore

    logger.addHandler(handler)  # type: ignore
    if handler_std is not None:
        logger.addHandler(handler_std)

    return logger
