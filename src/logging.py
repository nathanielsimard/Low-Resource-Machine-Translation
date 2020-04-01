import logging
import os
import sys
from datetime import datetime

LEVEL = logging.INFO

if "DEBUG" in os.environ:
    if os.environ["DEBUG"] == "1":
        LEVEL = logging.DEBUG


directory = os.path.join(
    "$HOME/project2-logs", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
)
os.makedirs(directory, exist_ok=True)
file_name = os.path.join(directory, "experiment.log")
file = open(file_name, "w")


def create_logger(name: str) -> logging.Logger:
    """Create a logger with default configuration and formatter."""
    logger = logging.getLogger(name)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler = logging.StreamHandler(stream=file)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(LEVEL)

    return logger
