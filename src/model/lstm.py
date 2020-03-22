from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras import layers

from src.model import base

NAME = "lstm"


class Encoder(base.Model):
    """Encoder of the lstm basic model."""

    def __init__(self, vocab_size: int):
        """Create the encoder.

        Args:
            vocab_size: Size of the vocabulary of the input language.
        """
        super().__init__(f"{NAME}-Encoder")
        self.embed = layers.Embedding(vocab_size, 200)
        self.lstm = layers.LSTM(200, return_state=True)

    def call(self, x: tf.Tensor, training=False) -> List[tf.Tensor]:
        """Call the foward past."""
        x = self.embed(x)
        _, hidden_state, carry_state = self.lstm(x)

        return [hidden_state, carry_state]


class Decoder(base.Model):
    """Decoder of the lstm basic model."""

    def __init__(self, vocab_size: int):
        """Create the encoder.

        Args:
            vocab_size: Size of the vocabulary of the output language.
        """
        super().__init__(f"{NAME}-Decoder")
        self.embed = layers.Embedding(vocab_size, 200)
        self.dense = layers.Dense(vocab_size, activation="softmax")
        self.lstm = layers.LSTM(200, return_sequences=True, return_state=True)

    def call(self, x: tf.Tensor, states: List[tf.Tensor], training=False) -> tf.Tensor:
        """Call the foward past."""
        x = self.embed(x)
        x, _, _ = self.lstm(x, initial_state=states)
        x = self.dense(x)

        return x


class Lstm(base.Model):
    """Basic sequence-to-sequence lstm model to perform machine translation."""

    def __init__(self, input_vocab_size: int, output_vocab_size: int):
        """Create the machine translation model."""
        super().__init__(NAME)
        self.encoder = Encoder(input_vocab_size)
        self.decoder = Decoder(output_vocab_size)

    def call(self, x: Tuple[tf.Tensor, tf.Tensor], training=False) -> tf.Tensor:
        """Call the foward past.

        Args:
            x: Inputs of the model (encoder_input, decoder_input).
            training: If the model is training.
        """
        states = self.encoder(x[0])
        x = self.decoder(x[1], states)

        return x

    @property
    def padded_shapes(self):
        """Padded shapes used to add padding when batching multiple sequences."""
        return (([None], [None]), [None])

    def preprocessing(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Proprocess dataset to have ((encoder_input, decoder_input), target)."""

        def preprocess(input_sentence, output_sentence):
            return ((input_sentence, output_sentence[:-1]), output_sentence[1:])

        return dataset.map(preprocess)
