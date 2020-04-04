import tensorflow as tf
from src.model.transformer import Encoder
from src.model import base
from tensorflow.keras import layers

NAME = "demi-bert"


class DemiBERT(base.Model):
    def __init__(
        self, num_layers, embedding_size, num_heads, dff, vocab_size, max_pe, dropout
    ):
        super().__init__(f"{NAME}")

        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=embedding_size,
            num_heads=num_heads,
            dff=dff,
            input_vocab_size=vocab_size,
            maximum_positional_encoding=max_pe,
            rate=dropout,
        )
        self.dense = layers.Dense(embedding_size)
        self.embeddings = self.encoder.embedding.embeddings

    def call(self, x: tf.Tensor, training=False):
        x = self.encoder(x, training)
        x = self.dense(x)
        x = tf.linalg.matmul(x, self.embeddings, transpose_b=True)
        x = tf.keras.activations.softmax(x)

        return x

    def padded_shapes(self):
        return [None]
