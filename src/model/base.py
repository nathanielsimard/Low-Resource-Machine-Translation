import abc
from typing import List

import numpy as np
import tensorflow as tf

from src.preprocessing import END_OF_SAMPLE_TOKEN, START_OF_SAMPLE_TOKEN
from src.text_encoder import TextEncoder

MODEL_BASE_DIR = "models"


class Model(tf.keras.Model):
    """All models will inherit from this class.

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


class MachineTranslationModel(Model, abc.ABC):
    """Model used to do machine translation.

    Each model must implement the translate method.
    """

    def predictions(
        self, outputs: tf.Tensor, encoder: TextEncoder, logit=True
    ) -> List[str]:
        """Generate prediction tokens from the outputs from the last layer."""
        sentences = outputs

        if logit:
            sentences = tf.math.argmax(sentences, axis=2)

        sentences = [encoder.decode(sentence) for sentence in sentences]

        return _clean_tokens(sentences)

    @abc.abstractmethod
    def translate(self, x: tf.Tensor, encoder: TextEncoder) -> tf.Tensor:
        """Translate a sentence from input.

        Example::
            >>> translated = model.translate(x, target_encoder)
            >>> predictions = model.predictions(translated, target_encoder, logit=False)

        Returns the indexes corresponding to each vocabulary word.
        """
        pass


def _clean_tokens(sentences):
    result = []
    for sentence in sentences:
        new_sentence = []
        for word in sentence.split():
            if not (START_OF_SAMPLE_TOKEN in word or END_OF_SAMPLE_TOKEN in word):
                new_sentence.append(word)
        result.append(" ".join(new_sentence))

    return result
