import abc
from typing import List

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

    @abc.abstractproperty
    def embedding_size(self):
        """Size of the embedding layer."""
        pass

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
            sentences = tf.argmax(sentences, axis=-1)

        sentences = [encoder.decode(sentence) for sentence in sentences.numpy()]

        return clean_sentences(sentences)

    @abc.abstractmethod
    def translate(
        self, x: tf.Tensor, encoder: TextEncoder, max_seq_length: int
    ) -> tf.Tensor:
        """Translate a sentence from input.

        Example::
            >>> translated = model.translate(x, target_encoder, max_seq_length)
            >>> predictions = model.predictions(translated, target_encoder, logit=False)

        Returns the indexes corresponding to each vocabulary word.
        """
        pass


def clean_sentences(sentences: List[str]) -> List[str]:
    """Clean sentences from start en end token."""
    result = []
    for sentence in sentences:
        result.append(" ".join(_clean_tokens(sentence)))

    return result


def _clean_tokens(sentence: str) -> List[str]:
    cleaned_sentence: List[str] = []
    for word in sentence.split():
        if END_OF_SAMPLE_TOKEN == word:
            return cleaned_sentence

        if START_OF_SAMPLE_TOKEN == word:
            continue

        cleaned_sentence.append(word)

    return cleaned_sentence
