import abc
import enum
import os
import pickle
import subprocess
from collections import defaultdict
from typing import Dict, List

import tensorflow as tf

from src import logging
from src.dataloader import AlignedDataloader
from src.model import base

logger = logging.create_logger(__name__)


class History(object):
    """Keeps track of the different losses."""

    def __init__(self):
        """Initialize dictionaries."""
        self.logs: Dict[str, List[float]] = defaultdict(History._new_log)

    def record(self, name, value):
        """Stores value in the corresponding log."""
        self.logs[name].append(value)

    def save(self, file_name):
        """Save file."""
        with open(file_name, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_name):
        """Load file."""
        with open(file_name, "rb") as file:
            return pickle.load(file)

    @staticmethod
    def _new_log():
        return []


class Training(abc.ABC):
    """Abstract class for training a model."""

    @abc.abstractmethod
    def run(
        self, batch_size: int, num_epoch: int, checkpoint=None,
    ):
        """Each training sub-class must implement their own run method."""
        pass


class Metrics(enum.Enum):
    """Supported metrics to calculate after each epoch.

    Each metric will be calculated for the train/valid datasets.
    """

    BLEU = "bleu"
    ABSOLUTE_ACC = "absolute accuracy"


def test(
    model: base.Model,
    loss_fn,
    dataloader: AlignedDataloader,
    batch_size: int,
    checkpoint: int,
):
    """Test a model at a specific checkpoint."""
    dataset = dataloader.create_dataset()
    dataset = model.preprocessing(dataset)
    model.load(str(checkpoint))

    predictions, _ = _generate_predictions(
        model, dataset, dataloader.encoder_target, batch_size, loss_fn
    )

    directory = os.path.join("results/test", model.title)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"test-{checkpoint}")
    write_text(predictions, path)

    bleu = compute_bleu(path, dataloader.file_name_target)
    logger.info(f"Bleu score {bleu}")


def _generate_predictions(model, dataset, encoder, batch_size, loss_fn):
    predictions = []
    for inputs, targets in dataset.padded_batch(
        batch_size, padded_shapes=model.padded_shapes
    ):
        outputs = model(inputs, training=False)
        predictions += model.predictions(outputs, encoder)

        loss = loss_fn(targets, outputs)

    return predictions, loss


def write_text(sentences, output_file):
    """Write text from sentences."""
    with open(output_file, "w+") as out_stream:
        for sentence in reversed(sentences):
            out_stream.write(sentence + "\n")


def compute_bleu(pred_file_path: str, target_file_path: str):
    """Compute bleu score.

    Args:
        pred_file_path: the file path that contains the predictions.
        target_file_path: the file path that contains the targets (also called references).

    Returns: Bleu score

    """
    out = subprocess.run(
        [
            "sacrebleu",
            "--input",
            pred_file_path,
            target_file_path,
            "--tokenize",
            "none",
            "--sentence-level",
            "--score-only",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines = out.stdout.split("\n")
    scores = [float(x) for x in lines[:-1]]
    return sum(scores) / len(scores)
