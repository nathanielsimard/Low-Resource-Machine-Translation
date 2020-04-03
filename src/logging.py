import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

LEVEL = logging.INFO

handler = None

if "DEBUG" in os.environ:
    if os.environ["DEBUG"] == "1":
        LEVEL = logging.DEBUG


def initialise_logger(args):
    directory = os.path.join("logging/" + args.model, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    os.makedirs(directory, exist_ok=True)

    file_name = os.path.join(directory, "experiment.log")

    formatter = logging.Formatter("[%(asctime)s - %(levelname)s - %(name)s] %(message)s")

    global handler

    # Create a handler that rotate files each 512mb
    handler = RotatingFileHandler(
        file_name, mode="a", maxBytes=536_870_912, backupCount=4, encoding=None
    )
    handler.setFormatter(formatter)


def create_logger(name: str) -> logging.Logger:
    """Create a logger with default configuration and formatter."""
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(LEVEL)
    return logger
