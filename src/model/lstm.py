from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src.dataloader import END_OF_SAMPLE_TOKEN_INDEX, UNKNOWN_TOKEN_INDEX
from src.model import base

NAME = "lstm"
MAX_SEQ_LENGHT = 100


class Encoder(base.Model):
    """Encoder of the lstm basic model."""

    def __init__(self, vocab_size: int):
        """Create the encoder.

        Args:
            vocab_size: Size of the vocabulary of the input language.
        """
        super().__init__(f"{NAME}-Encoder")
        self.embed = layers.Embedding(vocab_size, 256)

        self.lstm1 = layers.LSTM(256, return_sequences=True, return_state=True)
        self.dense1 = layers.TimeDistributed(layers.Dense(1024, activation="relu"))
        self.lstm2 = layers.LSTM(1024, return_state=True)

    def call(self, x: tf.Tensor, training=False) -> List[List[tf.Tensor]]:
        """Call the foward past."""
        x = self.embed(x)

        x, hidden_state_1, carry_state_1 = self.lstm1(x)
        x = self.dense1(x)

        _, hidden_state_2, carry_state_2 = self.lstm2(x)

        return [[hidden_state_1, carry_state_1], [hidden_state_2, carry_state_2]]


class Decoder(base.Model):
    """Decoder of the lstm basic model."""

    def __init__(self, vocab_size: int):
        """Create the encoder.

        Args:
            vocab_size: Size of the vocabulary of the output language.
        """
        super().__init__(f"{NAME}-Decoder")
        self.embed = layers.Embedding(vocab_size, 256)
        self.lstm1 = layers.LSTM(256, return_sequences=True, return_state=True)
        self.dense1 = layers.TimeDistributed(layers.Dense(1024, activation="relu"))

        self.lstm2 = layers.LSTM(1024, return_sequences=True, return_state=True)
        self.dense2 = layers.TimeDistributed(
            layers.Dense(vocab_size, activation="softmax")
        )

    def call(
        self, x: tf.Tensor, states: List[List[tf.Tensor]], training=False
    ) -> tf.Tensor:
        """Call the foward past."""
        x = self.embed(x)

        x, hidden_state_1, carry_state_1 = self.lstm1(x, initial_state=states[0])
        x = self.dense1(x)

        x, hidden_state_2, carry_state_2 = self.lstm2(x, initial_state=states[1])
        x = self.dense2(x)

        return x, [[hidden_state_1, carry_state_1], [hidden_state_2, carry_state_2]]


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
        x, _ = self.decoder(x[1], states)

        return x

    @property
    def padded_shapes(self):
        """Padded shapes used to add padding when batching multiple sequences."""
        return (([None], [None]), [None])

    def translate(self, x: tf.Tensor):
        """Translate on input tensor."""
        batch_size = x.shape[0]
        states = self.encoder(x)

        words = tf.ones([batch_size, 1], dtype=tf.int64)

        last_words, states = self.decoder(words, states)
        last_words = tf.math.argmax(last_words, axis=2)
        words = tf.concat([words, last_words], 1)

        # print(f"Words : {words}")
        # print(f"Last Words : {last_words}")
        end_of_sample = (
            np.zeros([batch_size, 1], dtype=np.int64) + END_OF_SAMPLE_TOKEN_INDEX
        )
        while not (np.array_equal(last_words.numpy(), end_of_sample) or words.shape[1] > MAX_SEQ_LENGHT):
            # print("Inside")
            last_words, states = self.decoder(last_words, states)
            last_words = tf.math.argmax(last_words, axis=2)
            words = tf.concat([words, last_words], 1)

            # print(last_words)
            # print(words.shape)
            # print(f"Last Words : {last_words}")

        return words

    def preprocessing(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Proprocess dataset to have ((encoder_input, decoder_input), target)."""

        def preprocess(input_sentence, output_sentence):
            return ((input_sentence, output_sentence[:-1]), output_sentence[1:])

        return dataset.map(preprocess)
