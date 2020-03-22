import abc

import tensorflow as tf

from src.data import dataloader

MODEL_BASE_DIR = "models"


class Model(tf.keras.Model, abc.ABC):
    """All models will inherit from this class.

    Each model must supplie their configuration with what features they need.
    Each model has full control over the preprocessing apply on the data.
    """

    def __init__(self, title: str):
        """Name of the model."""
        super().__init__()
        self.title = title

    def save(self, instance: str):
        """Save the model weights."""
        file_name = f"{MODEL_BASE_DIR}/{self.title}/{instance}"
        super().save_weights(
            file_name, save_format="tf", overwrite=True,
        )

    def load(self, instance: str):
        """Loading the model weights."""
        file_name = f"{MODEL_BASE_DIR}/{self.title}/{instance}"
        super().load_weights(file_name)

    @abc.abstractmethod
    def call(self, x: tf.Tensor, training=False) -> tf.Tensor:
        """Call the foward past."""
        pass
