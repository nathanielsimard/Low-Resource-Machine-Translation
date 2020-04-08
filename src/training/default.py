import os
from datetime import datetime
from typing import List

import numpy as np
import tensorflow as tf

from src import logging
from src.dataloader import AlignedDataloader
from src.training import base

logger = logging.create_logger(__name__)


class Training(base.Training):
    """Train a machine translation model with only aligned datasets."""

    def __init__(
        self,
        model,
        train_dataloader: AlignedDataloader,
        valid_dataloader: AlignedDataloader,
        metrics: List[base.Metrics],
    ):
        """Create BasicMachineTranslationTraining."""
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.metrics = metrics
        self.recorded_losses = {
            "train": tf.keras.metrics.Mean("train_loss", tf.float32),
        }

        self.history = base.History()

        self.max_seq_length = self.valid_dataloader.max_seq_length
        if self.max_seq_length is None:
            self.max_seq_length = 250

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
                    inputs, targets, i, batch_size, optimizer, loss_fn
                )

            valid_predictions = self._valid_step(
                valid_dataset,
                loss_fn,
                batch_size,
                self.valid_dataloader.encoder_target,
                self.max_seq_length,
            )

            train_path = os.path.join(directory, f"train-{epoch}")
            valid_path = os.path.join(directory, f"valid-{epoch}")

            base.write_text(train_predictions, train_path)
            base.write_text(valid_predictions, valid_path)

            if base.Metrics.BLEU in self.metrics:
                self._record_bleu(epoch, train_path, valid_path)

            self._update_progress(epoch)

            self.model.save(epoch)
            self.history.save(directory + f"/history-{epoch}")

    def _train_step(
        self,
        inputs,
        targets,
        batch: int,
        batch_size: int,
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

        metric = self.recorded_losses["train"]
        metric(loss)

        if base.Metrics.ABSOLUTE_ACC in self.metrics:
            self._record_abs_acc(outputs, targets, batch, batch_size, "train")

        logger.info(f"Batch #{batch} : training loss {metric.result()}")

        return predictions

    def _valid_step(self, dataset, loss_fn, batch_size, encoder, max_seq_length):
        valid_predictions: List[str] = []
        for i, (inputs, targets) in enumerate(
            dataset.padded_batch(batch_size, padded_shapes=([None], [None]))
        ):
            logger.info(f"Batch #{i} : validation")
            outputs = self.model.translate(inputs, encoder, max_seq_length)
            valid_predictions += self.model.predictions(
                outputs, self.valid_dataloader.encoder_target, logit=False
            )

        return valid_predictions

    def _update_progress(self, epoch):
        train_metric = self.recorded_losses["train"]

        logger.info(
            f"Epoch: {epoch}, Train loss: {train_metric.result()}"
        )

        # Reset the cumulative recorded_losses after each epoch
        self.history.record("train_loss", train_metric.result())
        train_metric.reset_states()

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
        self.history.record(name + "_abs_acc", absolute_accuracy)
        logger.info(f"Batch #{batch} : {name} accuracy {absolute_accuracy*100} %")
