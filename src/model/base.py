import abc
from typing import List

import numpy as np
import tensorflow as tf

from src.preprocessing import END_OF_SAMPLE_TOKEN, START_OF_SAMPLE_TOKEN
from src.text_encoder import TextEncoder

MODEL_BASE_DIR = "models"
EN_TO_FR_FACTOR = 1.25


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
        self, x: tf.Tensor, encoder_inputs: TextEncoder, encoder_targets: TextEncoder
    ) -> tf.Tensor:
        """Translate a sentence from input.

        Example::
            >>> translated = model.translate(x, encoder_inputs, encoder_targets)
            >>> predictions = model.predictions(translated, target_encoder, logit=False)

        Returns the indexes corresponding to each vocabulary word.
        """
        pass


def clean_sentences(sentences: List[str]) -> List[str]:
    """Clean sentences from start en end token."""
    result = []
    for sentence in sentences:
        cleaned_sentence = " ".join(_clean_tokens(sentence))
        cleaned_sentence = _remove_early_dash(cleaned_sentence)
        result.append(cleaned_sentence)

    return result


def _remove_early_dash(sentence: str, num_char=4) -> str:
    """Check the first characters to remove unnecessary dashes."""
    if "-" in sentence[0:num_char]:
        sentence = sentence.replace("-", "", sentence.count("-", 0, num_char))
        sentence = _remove_first_empty_char(sentence)

    return sentence


def _remove_first_empty_char(sentence: str) -> str:
    while True:
        if len(sentence) > 0 and sentence[0] == " ":
            sentence = sentence[1:]
        else:
            return sentence


def _clean_tokens(sentence: str) -> List[str]:
    cleaned_sentence: List[str] = []
    for word in sentence.split():
        if END_OF_SAMPLE_TOKEN in word:
            return cleaned_sentence

        if START_OF_SAMPLE_TOKEN in word:
            continue

        cleaned_sentence.append(word)

    return cleaned_sentence


def translation_max_seq_lenght(
    inputs: tf.Tensor, encoder: TextEncoder, factor=EN_TO_FR_FACTOR
):
    """Calculate the max index for each sentence in the input tensor."""
    bool_tensor = inputs.numpy() == encoder.end_of_sample_index
    max_index_tensor = np.ceil(np.argmax(bool_tensor, axis=-1) * factor)

    return tf.convert_to_tensor(max_index_tensor, dtype=tf.int32,)
