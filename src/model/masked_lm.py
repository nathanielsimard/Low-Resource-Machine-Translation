import tensorflow as tf
from tensorflow.keras import layers

from src.model import base
from src.model.transformer import Encoder, _create_padding_mask

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
            maximum_position_encoding=max_pe,
            rate=dropout,
        )
        self.dense = layers.Dense(embedding_size)

    def call(self, x: tf.Tensor, training=False):
        padding_mask = _create_padding_mask(x)
        x = self.encoder(x, training, padding_mask)
        x = self.dense(x)
        x = tf.linalg.matmul(x, self.encoder.embedding.embeddings, transpose_b=True)
        x = tf.keras.activations.softmax(x)

        return x

    @property
    def padded_shapes(self):
        return [None]
