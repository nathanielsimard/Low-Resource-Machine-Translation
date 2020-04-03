from src.training import base
from src.dataloader import UnalignedDataloader
import tensorflow as tf
from datetime import datetime
from src.training import base
from src.logging import create_logger

logger = create_logger(__name__)


class Pretraining(base.Training):
    def __init__(
        self,
        model,
        train_monolingual_dataloader: UnalignedDataloader,
        valid_monolingual_dataloader: UnalignedDataloader,
    ):
        self.model = model
        self.train_dataloader = train_monolingual_dataloader
        self.valid_dataloader = valid_monolingual_dataloader
        self.metrics = {
            "train": tf.keras.metrics.Mean("train_loss", tf.float32),
            "valid": tf.keras.metrics.Mean("valid_loss", tf.float32),
        }

        self.history = base.History()

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

        for epoch in range(checkpoint + 1, num_epoch + 1):

            for i, minibatch in enumerate(
                train_dataset.padded_batch(batch_size, padded_shapes=([None]))
            ):
                with tf.GradientTape() as tape:
                    mask = self._create_mask(minibatch, 0.15)
                    inputs = self._apply_mask_to_inputs(minibatch, mask)
                    outputs = self.model(inputs, training=True)

                    def mlm_loss(real, pred):
                        mask_targets = tf.cast(mask, dtype=tf.int64)
                        loss_ = loss_fn(real, pred)

                        mask_targets = tf.cast(mask_targets, dtype=loss_.dtype)
                        loss_ *= mask_targets

                        return tf.reduce_mean(loss_)

                    loss = mlm_loss(minibatch, outputs)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(
                    zip(gradients, self.model.trainable_variables)
                )

    def _create_mask(self, inputs, prob):
        random_tensor = tf.random.uniform(
            inputs.shape, minval=0, maxval=1, dtype=tf.dtypes.float32
        )  # tensor of 0 and 1
        mask = random_tensor < prob
        return mask

    def _apply_mask_to_inputs(self, inputs, mask):
        random_tensor = tf.random.uniform(
            inputs.shape, minval=0, maxval=1, dtype=tf.dtypes.float32
        )
        mask_ = tf.cast(mask, dtype=tf.int64)

        mask_index = (
            tf.cast(random_tensor < 0.8, dtype=tf.int64)
            * self.train_dataloader.encoder.mask_token_index
            * mask_
        )
        unchanged_index = (
            tf.cast(
                tf.math.logical_and(random_tensor >= 0.8, random_tensor < 0.9),
                dtype=tf.int64,
            )
            * inputs
            * mask_
        )

        random_words = tf.random.uniform(
            inputs.shape,
            minval=5,
            maxval=self.train_dataloader.encoder.vocab_size,
            dtype=tf.int64,
        )
        random_index = (
            tf.cast(random_tensor >= 0.9, dtype=tf.int64) * random_words * mask_
        )

        mask_inputs = tf.cast(tf.math.logical_not(mask), dtype=tf.int64)

        inputs = inputs * mask_inputs

        return inputs + unchanged_index + random_index + mask_index
