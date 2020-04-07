import os
from datetime import datetime
from typing import List

import tensorflow as tf
import numpy as np

from src import logging
from src.dataloader import AlignedDataloader
from src.training import base

logger = logging.create_logger(__name__)
"""
train_step_signature = [
    Tuple[tf.TensorSpec(shape=(None, None), dtype=tf.int64)],
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]
"""


class Training(base.Training):
    """Train a machine translation model with only aligned datasets."""

    def __init__(
        self,
        model,
        train_dataloader: AlignedDataloader,
        valid_dataloader: AlignedDataloader,
        metrics: List[base.Metrics],
        optim: tf.keras.optimizers,
        loss_fn: tf.keras.losses,
    ):
        """Create BasicMachineTranslationTraining."""
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.metrics = metrics
        self.recorded_losses = {
            "train": tf.keras.metrics.Mean("train_loss", tf.float32),
            "valid": tf.keras.metrics.Mean("valid_loss", tf.float32),
        }
        if base.Metrics.ABSOLUTE_ACC in self.metrics:
            self.accuracies = {
                "train": tf.keras.metrics.Mean("train_accuracy", tf.float32),
                "valid": tf.keras.metrics.Mean("valid_accuracy", tf.float32),
            }

        self.optimizer = optim
        self.loss_fn = loss_fn

        self.history = base.History()

    def run(
        self, batch_size: int, num_epoch: int, checkpoint=None,
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
                outputs, loss = self._train_step(inputs, targets)
                train_predictions += self.model.predictions(
                    outputs, self.train_dataloader.encoder_target
                )
                metric = self.recorded_losses["train"]
                metric(loss)
                logger.info(f"Batch #{i} : training loss: {metric.result()}")

                if base.Metrics.ABSOLUTE_ACC in self.metrics:
                    other_metric = self.accuracies["train"]
                    acc = self._record_abs_acc(outputs, targets, i, batch_size, "train")
                    other_metric(acc)

            valid_predictions: List[str] = []
            for i, (inputs, targets) in enumerate(
                valid_dataset.padded_batch(
                    batch_size, padded_shapes=self.model.padded_shapes
                )
            ):

                outputs, loss = self._valid_step(inputs, targets)
                valid_predictions += self.model.predictions(
                    outputs, self.valid_dataloader.encoder_target
                )
                metric = self.recorded_losses["valid"]
                metric(loss)
                logger.info(f"Batch #{i} : validation loss {metric.result()}")
                if base.Metrics.ABSOLUTE_ACC in self.metrics:
                    other_metric = self.accuracies["valid"]
                    acc = self._record_abs_acc(outputs, targets, i, batch_size, "valid")
                    other_metric(acc)

            train_path = os.path.join(directory, f"train-{epoch}")
            valid_path = os.path.join(directory, f"valid-{epoch}")

            base.write_text(train_predictions, train_path)
            base.write_text(valid_predictions, valid_path)

            if base.Metrics.BLEU in self.metrics:
                self._record_bleu(epoch, train_path, valid_path)

            self._update_progress(epoch)

            self.model.save(epoch)
            self.history.save(directory + f"/history-{epoch}")

    @tf.function
    def _train_step(
        self, inputs, targets,
    ):
        target_inputs = targets[:-1]
        targets_true = targets[1:]
        with tf.GradientTape() as tape:
            outputs = self.model(inputs, target_inputs, training=True)
            # Calculate the loss and update the parameters
            loss = self.loss_fn(targets_true, outputs)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return outputs, loss

    @tf.function
    def _valid_step(self, inputs, targets):
        target_inputs = targets[:-1]
        targets_true = targets[1:]
        outputs = self.model(inputs, target_inputs, training=False)
        loss = self.loss_fn(targets_true, outputs)

        return outputs, loss

    def _update_progress(self, epoch):
        train_metric = self.recorded_losses["train"]
        valid_metric = self.recorded_losses["valid"]

        logger.info(
            f"Epoch: {epoch}, Train loss: {train_metric.result()}, Valid loss: {valid_metric.result()} "
        )

        # Reset the cumulative recorded_losses after each epoch
        self.history.record("train_loss", train_metric.result())
        self.history.record("valid_loss", valid_metric.result())
        train_metric.reset_states()
        valid_metric.reset_states()

    def _record_bleu(self, epoch, train_path, valid_path):
        train_bleu = base.compute_bleu(
            train_path, self.train_dataloader.file_name_target
        )
        valid_bleu = base.compute_bleu(
            valid_path, self.valid_dataloader.file_name_target
        )

        logger.info(
            f"Epoch {epoch}: train bleu score: {train_bleu} valid bleu score: {valid_bleu}"
        )

        self.history.record("train_bleu", train_bleu)
        self.history.record("valid_bleu", valid_bleu)

    def _record_abs_acc(self, outputs, targets, batch, batch_size, name):
        sentences = np.argmax(outputs.numpy(), axis=2)
        absolute_accuracy = (
            tf.math.reduce_mean(tf.cast(sentences == targets, dtype=tf.int64), axis=1)
            .numpy()
            .sum()
            / batch_size
        )

        logger.info(f"Batch #{batch} : {name} accuracy {absolute_accuracy*100} %")

        return absolute_accuracy
