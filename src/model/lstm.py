from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src.model import base
from src.text_encoder import TextEncoder

NAME = "lstm"


class Encoder(base.Model):
    """Encoder of the lstm basic model."""

    def __init__(self, vocab_size: int):
        """Create the encoder.

        Args:
            vocab_size: Size of the vocabulary of the input language.
        """
        super().__init__(f"{NAME}-Encoder")
        self.embed = layers.Embedding(vocab_size, 256)

        self.lstm1 = layers.LSTM(512, return_sequences=True, return_state=True)
        self.dense1 = layers.TimeDistributed(layers.Dense(512, activation="relu"))
        self.lstm2 = layers.LSTM(512, return_state=True)

    def call(self, x: tf.Tensor, training=False) -> List[List[tf.Tensor]]:
        """Call the foward past."""
        x = self.embed(x)

        x, hidden_state_1, carry_state_1 = self.lstm1(x)
        x = self.dense1(x)

        _, hidden_state_2, carry_state_2 = self.lstm2(x)

        return [[hidden_state_1, carry_state_1], [hidden_state_2, carry_state_2]]

    @property
    def embedding_size(self):
        """Embedding size."""
        return 256


class Decoder(base.Model):
    """Decoder of the lstm basic model."""

    def __init__(self, vocab_size: int):
        """Create the encoder.

        Args:
            vocab_size: Size of the vocabulary of the output language.
        """
        super().__init__(f"{NAME}-Decoder")
        self.embed = layers.Embedding(vocab_size, 256)
        self.lstm1 = layers.LSTM(512, return_sequences=True, return_state=True)
        self.dense1 = layers.TimeDistributed(layers.Dense(512, activation="relu"))

        self.lstm2 = layers.LSTM(512, return_sequences=True, return_state=True)
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

    @property
    def embedding_size(self):
        """Embedding size."""
        return 256


class Lstm(base.MachineTranslationModel):
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

    @property
    def embedding_size(self):
        """Embedding size."""
        return self.encoder.embedding_size

    def translate(
        self, x: tf.Tensor, encoder_inputs: TextEncoder, encoder_targets: TextEncoder
    ) -> tf.Tensor:
        """Translate on input tensor."""
        batch_size = x.shape[0]
        max_seq_length = tf.reduce_max(
            base.translation_max_seq_lenght(x, encoder_inputs)
        )
        states = self.encoder(x)

        # The first words of each sentence in the batch is the start of sample token.
        words = (
            tf.zeros([batch_size, 1], dtype=tf.int64)
            + encoder_targets.start_of_sample_index
        )
        last_words = words

        has_finish_predicting = False
        reach_max_seq_lenght = False

        while not (has_finish_predicting or reach_max_seq_lenght):
            # Predicting the next words from the last words using
            # the last state while updating the last words and states.
            last_words, states = self.decoder(last_words, states)
            last_words = tf.math.argmax(last_words, axis=2)

            # Append the newly predicted words into words.
            words = tf.concat([words, last_words], 1)

            # Compute the end condition of the while loop.
            end_of_sample = (
                np.zeros([batch_size, 1], dtype=np.int64)
                + encoder_targets.end_of_sample_index
            )
            has_finish_predicting = np.array_equal(last_words.numpy(), end_of_sample)
            reach_max_seq_lenght = words.shape[1] >= max_seq_length

        return words

    def preprocessing(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Proprocess dataset to have ((encoder_input, decoder_input), target)."""

        def preprocess(input_sentence, output_sentence):
            return ((input_sentence, output_sentence[:-1]), output_sentence[1:])

        return dataset.map(preprocess)
