import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

handler = None
level = None


def initialize(args):
    """Initialize the logging module based on the user's arguments.

    It must be called before any logger is created.
    """
    directory = os.path.join(
        "logging/" + args.model, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    os.makedirs(directory, exist_ok=True)

    file_name = os.path.join(directory, "experiment.log")

    _initialize_handler(file_name)
    _initialize_level(args.debug)


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
    logger = logging.getLogger(name)
    logger.addHandler(handler)  # type: ignore
    logger.setLevel(level)  # type: ignore
    return logger
