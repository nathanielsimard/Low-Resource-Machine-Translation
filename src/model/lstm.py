import tensorflow as tf
from tensorflow.keras import layers

from src.model import base

NAME = "lsmt"


class Encoder(base.Model):
    def __init__(self, vocab_size: int):
        super().__init__(f"{NAME}-Encoder")
        self.embed = layers.Embedding(vocab_size, 200)

        self.lstm_1 = layers.LSTM(200, return_sequences=True)
        self.lstm_2 = layers.LSTM(200, return_state=True)

    def call(self, x: tf.Tensor, training=False) -> tf.Tensor:
        """Call the foward past."""
        x = self.embed(x)
        x = self.lstm_1(x)
        x = self.lstm_2(x)

        return x


class Decoder(base.Model):
    def __init__(self, vocab_size: int):
        super().__init__(f"{NAME}-Decoder")
        self.embed = layers.Embedding(vocab_size, 200)
        self.lstm_1 = layers.LSTM(200, return_sequences=True)
        self.lstm_2 = layers.LSTM(200, return_sequences=True)

    def call(self, x: tf.Tensor, training=False) -> tf.Tensor:
        """Call the foward past."""
        x = self.lstm_1(x)
        x = self.lstm_2(x)
        x = self.embed(x)

        return x


class Lstm(base.Model):
    def __init__(self):
        super().__init__(NAME)
        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, x: tf.Tensor, training=False) -> tf.Tensor:
        """Call the foward past."""
        x = self.encoder(x)
        x = self.decoder(x)

        return x
