from src.training import base
from src.dataloader import UnalignedDataloader
import tensorflow as tf

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

        self.history = base.History("train_ppl", "valid_ppl")

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

        train_dataset = self.model.preprocessing(train_dataset,)
        valid_dataset = self.model.preprocessing(valid_dataset)
