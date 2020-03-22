import tensorflow as tf
from tensorflow.keras import layers
from typing import List, Tuple

from src.model import base

NAME = "lsmt"


class Encoder(base.Model):
    def __init__(self, vocab_size: int):
        super().__init__(f"{NAME}-Encoder")
        self.embed = layers.Embedding(vocab_size, 200)

        # self.lstm_1 = layers.LSTM(200, return_sequences=True)
        self.lstm_2 = layers.LSTM(200, return_state=True)

    def call(self, x: tf.Tensor, training=False) -> List[tf.Tensor]:
        """Call the foward past."""
        x = self.embed(x)
        # x = self.lstm_1(x)
        _, hidden_state, carry_state = self.lstm_2(x)

        return [hidden_state, carry_state]


class Decoder(base.Model):
    def __init__(self, vocab_size: int):
        super().__init__(f"{NAME}-Decoder")
        self.embed = layers.Embedding(vocab_size, 200)
        self.dense = layers.Dense(vocab_size, activation="softmax")
        self.lstm_1 = layers.LSTM(200, return_sequences=True, return_state=True)
        # self.lstm_2 = layers.LSTM(200, return_sequences=True)

    def call(self, x: tf.Tensor, states: List[tf.Tensor], training=False) -> tf.Tensor:
        """Call the foward past."""
        x = self.embed(x)
        x, _, _ = self.lstm_1(x, initial_state=states)
        # x = self.lstm_2(x)
        x = self.dense(x)

        return x


class Lstm(base.Model):
    def __init__(self, input_vocab_size: int, output_vocab_size: int):
        super().__init__(NAME)
        self.encoder = Encoder(input_vocab_size)
        self.decoder = Decoder(output_vocab_size)

    def call(self, x: Tuple[tf.Tensor, tf.Tensor], training=False) -> tf.Tensor:
        """Call the foward past."""
        states = self.encoder(x[0])
        x = self.decoder(x[1], states)

        return x
