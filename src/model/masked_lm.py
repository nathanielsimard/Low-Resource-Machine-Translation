import tensorflow as tf
from tensorflow.keras import layers

from src.logging import create_logger
from src.model import base
from src.model.transformer import Encoder, _create_padding_mask

NAME = "demi-bert"
logger = create_logger(__name__)


class DemiBERT(base.Model):
    """D E M I B E R T."""

    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, max_pe, rate):
        """Initialize D E M I B E R T."""
        super().__init__(f"{NAME}")

        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            input_vocab_size=vocab_size,
            maximum_position_encoding=max_pe,
            rate=rate,
        )
        self.dense = layers.Dense(d_model)
        self._embedding_size = d_model

    def call(self, x: tf.Tensor, training=False):
        """Performs a forward pass."""
        logger.debug("Forward pass.")
        padding_mask = _create_padding_mask(x)
        logger.debug("Before encoder.")
        x = self.encoder(x, training, padding_mask)
        logger.debug("After encoder.")
        x = self.dense(x)
        logger.debug("After dense.")
        x = tf.linalg.matmul(x, self.encoder.embedding.embeddings, transpose_b=True)
        logger.debug("Done.")

        return x

    @property
    def padded_shapes(self):
        """Padded shapes for the minibatch."""
        return [None]

    @property
    def embedding_size(self):
        """Embedding size."""
        return self._embedding_size
