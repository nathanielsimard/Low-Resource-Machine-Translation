from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src.model import base
from src.text_encoder import TextEncoder

# DISCLAIMER: This module is taken as is from the tensorflow documentation, under the Creative Commons 4.0 License.
# It is thus available for free adaptation and sharing.
# The more detailed implementation can be found @ https://www.tensorflow.org/tutorials/text/transformer

NAME = "transformer-2"
MAX_SEQ_LENGHT = 250


class Transformer(base.MachineTranslationModel):
    """Transformer model."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        input_vocab_size: int,
        target_vocab_size: int,
        pe_input: int,
        pe_target: int,
    ):
        """Initialize the Encoder module.

        Args:
            d_model: dimensional space of the embeddings.
            num_layers: number of layers of the Encoder.
            num_heads: number of heads for the MultiHeadAttention module.
            dff: dimension of the feed forward layer.
            input_vocab_size: size of the vocabulary of the input language.
            target_vocab_size: size of the vocabulary of the target language.
            maximum_position_encoding: cap on the positional encoding values.
            pe_input: positional encodings of the input.decoder
        """
        super(Transformer, self).__init__(f"{NAME}")
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size

        self.encoder = Encoder(num_layers, d_model, num_heads, input_vocab_size, pe_input)
        self.encoder_output, _ = self.encoder(tf.constant([[1, 2, 3, 0, 0]]))

        self.decoder = Decoder(num_layers, d_model, num_heads, target_vocab_size, pe_target)
        self.decoder_output, _, _ = self.decoder(tf.constant([[14, 24, 36, 0, 0]]), self.encoder_output)
       # self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, x: Tuple[tf.Tensor, tf.Tensor], training=False):
        """Forward pass of the Transformer."""
        encoder_mask = 1 - tf.cast(tf.equal(x[0], 0), dtype=tf.float32)
        encoder_mask = tf.expand_dims(encoder_mask, axis=1)
        encoder_mask = tf.expand_dims(encoder_mask, axis=1)
        encoder_output, _ = self.encoder(x[0], encoder_mask=encoder_mask)

        decoder_output, _, _ = self.decoder(
            x[1], encoder_output, encoder_mask=encoder_mask)

        return decoder_output

    def translate(self, x: tf.Tensor, encoder: TextEncoder):
        """Translation function for the test set."""
        batch_size = x.shape[0]
        # The first words of each sentence in the batch is the start of sample token.
        words = (
            tf.zeros([batch_size, 1], dtype=tf.int64) + encoder.start_of_sample_index
        )
        last_words = words

        has_finish_predicting = False
        reach_max_seq_lenght = False

        # Always use the same mask because the decoder alway decode one word at a time.
        en_output, en_alignments = encoder(tf.constant(x), training=False)
        de_input = tf.constant([[encoder.start_of_sample_index]], dtype=tf.int64)
        while not (has_finish_predicting or reach_max_seq_lenght):
            dec_output, de_bot_alignments, de_mid_alignments = self.decoder(
                de_input, en_output, training=False
            )
            last_words = self.final_layer(dec_output)
            last_words = tf.math.argmax(last_words, axis=2)
            words = tf.concat([words, last_words], 1)

            # Compute the end condition of the while loop.
            end_of_sample = (
                np.zeros([batch_size, 1], dtype=np.int64) + encoder.end_of_sample_index
            )
            has_finish_predicting = np.array_equal(last_words.numpy(), end_of_sample)
            reach_max_seq_lenght = words.shape[1] >= MAX_SEQ_LENGHT

        return words

    @property
    def padded_shapes(self):
        """Padded shapes used to add padding when batching multiple sequences."""
        return (([None], [None]), [None])

    def preprocessing(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Proprocess dataset to have ((encoder_input, decoder_input), target)."""

        def preprocess(input_sentence, output_sentence):
            return ((input_sentence, output_sentence[:-1]), output_sentence[1:])

        return dataset.map(preprocess)


class Decoder(layers.Layer):
    """Decoder of the Transformer.

    Stacks Decoder Layers.
    """

    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        target_vocab_size,
        maximum_position_encoding
    ):
        """Initialize the Encoder module.

        Args:
            d_model: dimensional space of the embeddings.
            num_layers: number of layers of the Encoder.
            num_heads: number of heads for the MultiHeadAttention module.
            dff: dimension of the feed forward layer.
            target_vocab_size: size of the vocabulary of the target language.
            maximum_position_encoding: cap on the positional encoding values.
            rate: dropout rate.
        """
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.embedding_dropout = tf.keras.layers.Dropout(0.1)
        self.attention_bot = [MultiHeadAttention(d_model, num_heads) for _ in range(num_layers)]
        self.attention_bot_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]
        self.attention_bot_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]
        self.attention_mid = [MultiHeadAttention(d_model, num_heads) for _ in range(num_layers)]
        self.attention_mid_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]
        self.attention_mid_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]

        self.dense_1 = [tf.keras.layers.Dense(
            d_model * 4, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(
            d_model) for _ in range(num_layers)]
        self.ffn_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]

        self.dense = tf.keras.layers.Dense(target_vocab_size)

        self.pes = _positional_encoding(maximum_position_encoding, d_model)

    def call(self, x, enc_output, training=True, encoder_mask=None):
        """Forward pass in the Decoder module."""
        embed_out = self.embedding(x)

        embed_out *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        embed_out += self.pes[:x.shape[1], :]
        embed_out = self.embedding_dropout(embed_out)

        bot_sub_in = embed_out
        bot_alignments = []
        mid_alignments = []

        for i in range(self.num_layers):
            # BOTTOM MULTIHEAD SUB LAYER
            seq_len = bot_sub_in.shape[1]

            if training:
                mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
            else:
                mask = None

            bot_sub_out, bot_alignment = self.attention_bot[i](bot_sub_in, bot_sub_in, mask)
            bot_sub_out = self.attention_bot_dropout[i](bot_sub_out, training=training)
            bot_sub_out = bot_sub_in + bot_sub_out
            bot_sub_out = self.attention_bot_norm[i](bot_sub_out)

            bot_alignments.append(bot_alignment)

            # MIDDLE MULTIHEAD SUB LAYER
            mid_sub_in = bot_sub_out
            mid_sub_out, mid_alignment = self.attention_mid[i](
                mid_sub_in, enc_output, encoder_mask)
            mid_sub_out = self.attention_mid_dropout[i](mid_sub_out, training=training)
            mid_sub_out = mid_sub_out + mid_sub_in
            mid_sub_out = self.attention_mid_norm[i](mid_sub_out)

            mid_alignments.append(mid_alignment)

            # FFN
            ffn_in = mid_sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = self.ffn_dropout[i](ffn_out, training=training)
            ffn_out = ffn_out + ffn_in
            ffn_out = self.ffn_norm[i](ffn_out)

            bot_sub_in = ffn_out

        logits = self.dense(ffn_out)

        return logits, bot_alignments, mid_alignments


class Encoder(layers.Layer):
    """Encoder of the Transformer model.

    Stacks many EncoderLayer.
    """

    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        input_vocab_size,
        maximum_position_encoding
    ):
        """Initialize the Encoder module.

        Args:
            d_model: dimensional space of the embeddings.
            num_layers: number of layers of the Encoder.
            num_heads: number of heads for the MultiHeadAttention module.
            dff: dimension of the feed forward layer.
            input_vocab_size: size of the vocabulary of the input language.
            maximum_position_encoding: cap on the positional encoding values.
            rate: dropout rate.
        """
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.embedding_dropout = tf.keras.layers.Dropout(0.1)
        self.attention = [MultiHeadAttention(d_model, num_heads) for _ in range(num_layers)]
        self.attention_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]

        self.attention_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]

        self.dense_1 = [tf.keras.layers.Dense(
            d_model * 4, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(
            d_model) for _ in range(num_layers)]
        self.ffn_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]

        self.pes = _positional_encoding(maximum_position_encoding, d_model)

    def call(self, x, training=True, encoder_mask=None):
        """Forward pass in the Encoder."""
        embed_out = self.embedding(x)

        embed_out *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        embed_out += self.pes[:x.shape[1], :]
        embed_out = self.embedding_dropout(embed_out)

        sub_in = embed_out
        alignments = []

        for i in range(self.num_layers):
            sub_out, alignment = self.attention[i](sub_in, sub_in, encoder_mask)
            sub_out = self.attention_dropout[i](sub_out, training=training)
            sub_out = sub_in + sub_out
            sub_out = self.attention_norm[i](sub_out)

            alignments.append(alignment)
            ffn_in = sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = self.ffn_dropout[i](ffn_out, training=training)
            ffn_out = ffn_in + ffn_out
            ffn_out = self.ffn_norm[i](ffn_out)

            sub_in = ffn_out

        return ffn_out, alignments


class MultiHeadAttention(layers.Layer):
    """Multi head attention module."""

    def __init__(self, d_model, num_heads):
        """Initialization."""
        super(MultiHeadAttention, self).__init__()
        self.key_size = d_model // num_heads
        self.num_heads = num_heads
        self.wq = tf.keras.layers.Dense(d_model)  # [tf.keras.layers.Dense(key_size) for _ in range(h)]
        self.wk = tf.keras.layers.Dense(d_model)  # [tf.keras.layers.Dense(key_size) for _ in range(h)]
        self.wv = tf.keras.layers.Dense(d_model)  # [tf.keras.layers.Dense(value_size) for _ in range(h)]
        self.wo = tf.keras.layers.Dense(d_model)

    def call(self, q, v, mask):
        """Forward pass in the attention module."""
        # query has shape (batch, query_len, model_size)
        # value has shape (batch, value_len, model_size)
        query = self.wq(q)
        key = self.wk(v)
        value = self.wv(v)

        # Split matrices for multi-heads attention
        batch_size = query.shape[0]

        # Originally, query has shape (batch, query_len, model_size)
        # We need to reshape to (batch, query_len, h, key_size)
        query = tf.reshape(query, [batch_size, -1, self.num_heads, self.key_size])
        # In order to compute matmul, the dimensions must be transposed to (batch, h, query_len, key_size)
        query = tf.transpose(query, [0, 2, 1, 3])

        # Do the same for key and value
        key = tf.reshape(key, [batch_size, -1, self.num_heads, self.key_size])
        key = tf.transpose(key, [0, 2, 1, 3])
        value = tf.reshape(value, [batch_size, -1, self.num_heads, self.key_size])
        value = tf.transpose(value, [0, 2, 1, 3])

        # Compute the dot score
        # and divide the score by square root of key_size (as stated in paper)
        # (must convert key_size to float32 otherwise an error would occur)
        score = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.dtypes.cast(self.key_size, dtype=tf.float32))
        # score will have shape of (batch, h, query_len, value_len)

        # Mask out the score if a mask is provided
        # There are two types of mask:
        # - Padding mask (batch, 1, 1, value_len): to prevent attention being drawn to padded token (i.e. 0)
        # - Look-left mask (batch, 1, query_len, value_len): to prevent decoder to draw attention to tokens to the right
        if mask is not None:
            score *= mask

            # We want the masked out values to be zeros when applying softmax
            # One way to accomplish that is assign them to a very large negative value
            score = tf.where(tf.equal(score, 0), tf.ones_like(score) * -1e9, score)

        # Alignment vector: (batch, h, query_len, value_len)
        alignment = tf.nn.softmax(score, axis=-1)

        # Context vector: (batch, h, query_len, key_size)
        context = tf.matmul(alignment, value)

        # Finally, do the opposite to have a tensor of shape (batch, query_len, model_size)
        context = tf.transpose(context, [0, 2, 1, 3])
        context = tf.reshape(context, [batch_size, -1, self.key_size * self.num_heads])

        # Apply one last full connected layer (WO)
        heads = self.wo(context)

        return heads, alignment


def _positional_encoding(pos, model_size):
    """ Compute positional encoding for a particular position
    Args:
        pos: position of a token in the sequence
        model_size: depth size of the model

    Returns:
        The positional encoding for the given token
    """
    PE = np.zeros((1, model_size))
    for i in range(model_size):
        if i % 2 == 0:
            PE[:, i] = np.sin(pos / 10000 ** (i / model_size))
        else:
            PE[:, i] = np.cos(pos / 10000 ** ((i - 1) / model_size))
    return PE
