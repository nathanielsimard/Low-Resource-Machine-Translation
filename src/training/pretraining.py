import os
from datetime import datetime

import tensorflow as tf

from src.dataloader import UnalignedDataloader
from src.logging import create_logger
from src.training import base

logger = create_logger(__name__)


class Pretraining(base.Training):
    """Pretraining using a BERT-like Masked Language Model."""

    def __init__(
        self,
        model,
        train_monolingual_dataloader: UnalignedDataloader,
        valid_monolingual_dataloader: UnalignedDataloader,
        optim: tf.keras.optimizers,
        loss_fn: tf.keras.losses,
    ):
        """Initialize the pretraining session."""
        self.model = model
        self.train_dataloader = train_monolingual_dataloader
        self.valid_dataloader = valid_monolingual_dataloader
        self.losses = {
            "train": tf.keras.metrics.Mean("train_loss", tf.float32),
            "valid": tf.keras.metrics.Mean("valid_loss", tf.float32),
        }
        self.optimizer = optim
        self.loss_fn = loss_fn

        self.history = base.History()

    def run(
        self, batch_size: int, num_epoch: int, checkpoint=None,
    ):
        """Pretraining session."""
        logger.info("Creating datasets...")
        train_dataset = self.train_dataloader.create_dataset()
        valid_dataset = self.valid_dataloader.create_dataset()

        train_dataset = self.model.preprocessing(train_dataset)
        valid_dataset = self.model.preprocessing(valid_dataset)

        logger.info("Creating results directory...")

        directory = os.path.join(
            "results/" + self.model.title + "_mono",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        os.makedirs(directory, exist_ok=True)

        if checkpoint is not None:
            logger.info(f"Loading model {checkpoint}")
            self.model.load(str(checkpoint))
        else:
            checkpoint = 0

        logger.info("Beginning pretraining session...")

        for epoch in range(checkpoint + 1, num_epoch + 1):

            for i, minibatch in enumerate(
                train_dataset.padded_batch(
                    batch_size, padded_shapes=self.model.padded_shapes
                )
            ):
                with tf.GradientTape() as tape:
                    loss = self._step(minibatch, i, "train")
                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    self.optimizer.apply_gradients(
                        zip(gradients, self.model.trainable_variables)
                    )
            logger.debug("Saving training loss")
            self.history.record("train_loss", self.losses["train"].result())

            for i, minibatch in enumerate(
                valid_dataset.padded_batch(
                    batch_size, padded_shapes=self.model.padded_shapes
                )
            ):
                loss = self._step(minibatch, i, "valid")

            logger.debug("Saving validation loss")
            self.history.record("valid_loss", self.losses["valid"].result())

            self.model.save(epoch)

            self.losses["train"].reset_states()
            self.losses["valid"].reset_states()

    def _step(self, inputs, batch, name):
        masked_inputs, mask = self.create_and_apply_masks(inputs)
        outputs = self.model(masked_inputs, training=True)

        def mlm_loss(real, pred):
            mask_targets = tf.cast(mask, dtype=tf.int32)
            loss_ = self.loss_fn(real, pred)

            mask_targets = tf.cast(mask_targets, dtype=loss_.dtype)
            loss_ *= mask_targets

            logger.debug(f"loss : {loss_}")

            logger.debug(f"mask target: {mask_targets}")
            logger.debug(f"preds : {pred}")

            return tf.reduce_mean(loss_)

        loss = mlm_loss(inputs, outputs)
        self.losses[name](loss)
        logger.info(f"Batch {batch} : {name} loss: {self.losses[name].result()}")

        return loss

    def create_and_apply_masks(self, inputs):
        """Create masks, then apply them to the inputs."""
        mask = (
            tf.random.uniform(inputs.shape, minval=0, maxval=1, dtype=tf.dtypes.float32)
            < 0.15
        )

        random_tensor = tf.random.uniform(
            inputs.shape, minval=0, maxval=1, dtype=tf.dtypes.float32
        )
        mask_index = random_tensor < 0.8
        unchanged_index = tf.math.logical_and(random_tensor >= 0.8, random_tensor < 0.9)

        random_words = tf.random.uniform(
            inputs.shape,
            minval=5,
            maxval=self.train_dataloader.encoder.vocab_size,
            dtype=tf.int32,
        )
        random_index_mask = random_tensor >= 0.9

        return (
            self.apply_masks(
                mask,
                mask_index,
                random_index_mask,
                unchanged_index,
                inputs,
                random_words,
            ),
            mask,
        )

    def apply_masks(
        self,
        mask,
        masked_word_mask,
        random_word_mask,
        unchanged_mask,
        inputs,
        random_words,
    ):
        """Apply masks."""
        mask_ = tf.cast(mask, dtype=tf.int32)

        mask_index = (
            tf.cast(masked_word_mask, dtype=tf.int32)
            * self.train_dataloader.encoder.mask_token_index
            * mask_
        )
        unchanged_index = tf.cast(unchanged_mask, dtype=tf.int32) * inputs * mask_
        random_index = tf.cast(random_word_mask, dtype=tf.int32) * random_words * mask_

        inputs = inputs * tf.cast(tf.math.logical_not(mask), dtype=tf.int32)

        return inputs + unchanged_index + random_index + mask_index
