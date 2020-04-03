import abc
import os
import pickle
import subprocess
from datetime import datetime
from typing import List

import tensorflow as tf

from src import logging
from src.dataloader import AlignedDataloader
from src.model import base
from typing import Dict, Any

logger = logging.create_logger(__name__)


class History(object):
    """Keeps track of the different losses."""

    def __init__(self, *args: str):
        """Initialize dictionaries."""
        tmp_logs: Dict[str, Any] = {"train_loss": [], "valid_loss": []}
        for i in range(len(args)):
            tmp_logs.update({args[i]: []})
        self.logs = tmp_logs

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


class Training(abc.ABC):
    """Abstract class for training a model."""

    @abc.abstractmethod
    def run(
        self,
        loss_fn: tf.keras.losses,
        optimizer: tf.keras.optimizers,
        batch_size: int,
        num_epoch: int,
        checkpoint=None,
    ):
        """Each training sub-class must implement their own run method."""
        pass


class BasicMachineTranslationTraining(Training):
    """Train a machine translation model with only aligned datasets."""

    def __init__(
        self,
        model,
        train_dataloader: AlignedDataloader,
        valid_dataloader: AlignedDataloader,
    ):
        """Create BasicMachineTranslationTraining."""
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.metrics = {
            "train": tf.keras.metrics.Mean("train_loss", tf.float32),
            "valid": tf.keras.metrics.Mean("valid_loss", tf.float32),
        }

        self.history = History("train_bleu", "valid_bleu")

    def run(
        self,
        loss_fn: tf.keras.losses,
        optimizer: tf.keras.optimizers,
        batch_size: int,
        num_epoch: int,
        checkpoint=None,
    ):
        """Training session."""
        logger.info("Creating datasets...")
        train_dataset = self.train_dataloader.create_dataset()
        valid_dataset = self.valid_dataloader.create_dataset()

        train_dataset = self.model.preprocessing(train_dataset)
        valid_dataset = self.model.preprocessing(valid_dataset)

        logger.info("Creating results directory...")

        directory = os.path.join(
            "results/" + self.model.title, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        os.makedirs(directory, exist_ok=True)

        if checkpoint is not None:
            logger.info(f"Loading model {checkpoint}")
            self.model.load(str(checkpoint))
        else:
            checkpoint = 0

        for epoch in range(checkpoint + 1, num_epoch + 1):
            train_predictions: List[str] = []

            for i, (inputs, targets) in enumerate(
                train_dataset.padded_batch(
                    batch_size, padded_shapes=self.model.padded_shapes
                )
            ):

                train_predictions += self._train_step(
                    inputs, targets, i, optimizer, loss_fn
                )

            valid_predictions = self._valid_step(valid_dataset, loss_fn, batch_size)

            self._record_bleu(epoch, train_predictions, valid_predictions, directory)
            self.model.save(epoch)

            self._update_progress(epoch)
            self.history.save(directory + f"/history-{epoch}")

    def _train_step(
        self,
        inputs,
        targets,
        batch: int,
        optimizer: tf.keras.optimizers,
        loss_fn: tf.keras.losses,
    ):
        with tf.GradientTape() as tape:
            outputs = self.model(inputs, training=True)
            # Calculate the training prediction tokens
            predictions = self.model.predictions(
                outputs, self.train_dataloader.encoder_target
            )
            # Calculate the loss and update the parameters
            loss = loss_fn(targets, outputs)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        metric = self.metrics["train"]
        metric(loss)

        logger.info(f"Batch #{batch} : training loss {metric.result()}")

        return predictions

    def _valid_step(self, dataset, loss_fn, batch_size):
        valid_predictions: List[str] = []
        for i, (inputs, targets) in enumerate(
            dataset.padded_batch(batch_size, padded_shapes=self.model.padded_shapes)
        ):
            outputs = self.model(inputs, training=False)
            valid_predictions += self.model.predictions(
                outputs, self.valid_dataloader.encoder_target
            )

            loss = loss_fn(targets, outputs)
            metric = self.metrics["valid"]
            metric(loss)
            logger.info(f"Batch #{i} : validation loss {metric.result()}")

        return valid_predictions

    def _update_progress(self, epoch):
        train_metric = self.metrics["train"]
        valid_metric = self.metrics["valid"]

        logger.info(
            f"Epoch: {epoch}, Train loss: {train_metric.result()}, Valid loss: {valid_metric.result()} "
        )

        # Reset the cumulative metrics after each epoch
        self.history.record("train_loss", train_metric.result())
        self.history.record("valid_loss", valid_metric.result())
        train_metric.reset_states()
        valid_metric.reset_states()

    def _record_bleu(self, epoch, train_pred, valid_pred, directory):
        train_path = os.path.join(directory, f"train-{epoch}")
        valid_path = os.path.join(directory, f"valid-{epoch}")

        write_text(train_pred, train_path)
        write_text(valid_pred, valid_path)
        train_bleu = compute_bleu(train_path, self.train_dataloader.file_name_target)
        valid_bleu = compute_bleu(valid_path, self.valid_dataloader.file_name_target)

        logger.info(
            f"Epoch {epoch}: train bleu score: {train_bleu} valid bleu score: {valid_bleu}"
        )

        self.history.record("train_bleu", train_bleu)
        self.history.record("valid_bleu", valid_bleu)


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
