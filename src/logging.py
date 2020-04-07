import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

handler = None
level = None


def initialize(experiment_name="experiment", debug=False):
    """Initialize the logging module.

    It can be called before any logger is created to change the default arguments.
    """
    directory = os.path.join(
        "logging/" + experiment_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    os.makedirs(directory, exist_ok=True)

    file_name = os.path.join(directory, "experiment.log")

    _initialize_level(debug)
    _initialize_handler(file_name)


def _initialize_handler(file_name):
    global handler
    formatter = logging.Formatter(
        "[%(asctime)s - %(levelname)s - %(name)s] %(message)s"
    )
    # Create a handler that rotate files each 512mb
    handler = RotatingFileHandler(
        file_name, mode="a", maxBytes=536_870_912, backupCount=4, encoding=None
    )
    handler.setFormatter(formatter)


def _initialize_level(debug):
    global level
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO


def create_logger(name: str) -> logging.Logger:
    """Create a logger with default configuration and formatter."""
    global level
    global debug

    initialized = level is not None and handler is not None
    if not initialized:
        initialize()

    logger = logging.getLogger(name)
    logger.addHandler(handler)  # type: ignore
    logger.setLevel(level)  # type: ignore
    return logger
