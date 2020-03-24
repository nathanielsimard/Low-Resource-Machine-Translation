import abc

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import src.dataloader as dataloader

UNKNOWN_TOKEN = dataloader.UNKNOWN_TOKEN
END_OF_SAMPLE_TOKEN = dataloader.END_OF_SAMPLE_TOKEN

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

    @property
    def padded_shapes(self):
        """Padded shapes used to add padding when batching multiple sequences."""
        return ([None], [None])

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

    def preprocessing(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Can apply some preprocessing specific to the model."""
        return dataset

    def predictions(self, outputs: tf.Tensor, encoder: tfds.features.text.TextEncoder):
        """Generate prediction tokens from the outputs from the last layer."""
        # Index must start at 1.
        sentences = [
            encoder.decode(sentence + 1)
            for sentence in np.argmax(outputs.numpy(), axis=2)
        ]
        return _remove_unk(sentences)


def _remove_unk(sentences):
    result = []
    for sentence in sentences:
        new_sentence = []
        for word in sentence.split():
            if not (END_OF_SAMPLE_TOKEN in word or UNKNOWN_TOKEN in word):
                new_sentence.append(word)
        result.append(' '.join(new_sentence))

    return result
