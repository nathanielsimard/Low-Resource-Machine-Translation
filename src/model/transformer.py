from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src.model import base
from src.text_encoder import TextEncoder

# DISCLAIMER: This module is taken as is from the tensorflow documentation, under the Creative Commons 4.0 License.
# It is thus available for free adaptation and sharing.
# The more detailed implementation can be found @ https://www.tensorflow.org/tutorials/text/transformer

NAME = "transformer"


class Transformer(base.MachineTranslationModel):
    """Transformer model."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dff: int,
        input_vocab_size: int,
        target_vocab_size: int,
        pe_input: int,
        pe_target: int,
        rate: float,
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
            pe_input: positional encodings of the input.
            pe_target: positional encodings of the target.
            rate: dropout rate.
        """
        super(Transformer, self).__init__(f"{NAME}")
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size

        self.encoder = Encoder(
            num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate
        )

        self.decoder = Decoder(
            num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate
        )

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, x: Tuple[tf.Tensor, tf.Tensor], training=False):
        """Forward pass of the Transformer."""
        enc_padding_mask, look_ahead_mask, dec_padding_mask = _create_masks(x[0], x[1])
        enc_output = self.encoder(
            x[0], training, enc_padding_mask
        )  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            x[1], enc_output, training, look_ahead_mask, dec_padding_mask
        )

        final_output = self.final_layer(
            dec_output
        )  # (batch_size, tar_seq_len, target_vocab_size)
        return final_output

    def translate(self, x: tf.Tensor, encoder: TextEncoder, max_seq_length: int):
        batch_size = x.shape[0]
        max_seq_length = x.shape[1]
        output = (
            tf.zeros([batch_size, 1], dtype=tf.int64) + encoder.start_of_sample_index
        )

        for i in range(max_seq_length):
            enc_padding_mask, combined_mask, dec_padding_mask = _create_masks(x, output)
            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions = self.call((x, output), training=False)
            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            pred = tf.argmax(predictions, axis=-1)

            output = tf.concat([output, pred], axis=1)
        return output

    def allo(self, x: tf.Tensor, encoder: TextEncoder, max_seq_length: int):
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
        enc_padding_mask, look_ahead_mask, dec_padding_mask = _create_masks(
            x, last_words
        )
        enc_output = self.encoder(x, False, enc_padding_mask)
        while not (has_finish_predicting or reach_max_seq_lenght):
            dec_output, attention_weights = self.decoder(
                last_words, enc_output, False, look_ahead_mask, dec_padding_mask
            )
            last_words = self.final_layer(dec_output)
            last_words = tf.math.argmax(last_words, axis=2)
            words = tf.concat([words, last_words], 1)

            # Compute the end condition of the while loop.
            end_of_sample = (
                np.zeros([batch_size, 1], dtype=np.int64) + encoder.end_of_sample_index
            )
            has_finish_predicting = np.array_equal(last_words.numpy(), end_of_sample)
            reach_max_seq_lenght = words.shape[1] >= max_seq_length

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
        dff,
        target_vocab_size,
        maximum_position_encoding,
        rate=0.1,
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

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = _positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """Forward pass in the Decoder module."""
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                x, enc_output, training, look_ahead_mask, padding_mask
            )

            attention_weights["decoder_layer{}_block1".format(i + 1)] = block1
            attention_weights["decoder_layer{}_block2".format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Encoder(layers.Layer):
    """Encoder of the Transformer model.

    Stacks many EncoderLayer.
    """

    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        maximum_position_encoding,
        rate=0.1,
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

        self.embedding = layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = _positional_encoding(
            maximum_position_encoding, self.d_model
        )
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        """Forward pass in the Encoder."""
        seq_len = tf.shape(x)[1]
        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class EncoderLayer(layers.Layer):
    """Encoder layer of the Transformer model."""

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """Initialization."""
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = _point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training, mask):
        """Forward pass in this custom layer."""
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(
            out1 + ffn_output
        )  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(layers.Layer):
    """Decoder layer of the Transformer model."""

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """Initialization of the custom decoder layer."""
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = _point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """Forward pass in this custom layer."""
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(
            x, x, x, look_ahead_mask
        )  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask
        )  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(
            ffn_output + out2
        )  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class MultiHeadAttention(layers.Layer):
    """Multi head attention module."""

    def __init__(self, d_model, num_heads):
        """Initialization."""
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).

        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        """Forward pass in the attention module."""
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = _scaled_dot_product_attention(
            q, k, v, mask
        )

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model)
        )  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def _get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def _positional_encoding(position, d_model):
    angle_rads = _get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def _create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def _create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def _scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.

    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += mask * -1e9

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1
    )  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def _point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential(
        [
            layers.Dense(dff, activation="relu"),  # (batch_size, seq_len, dff)
            layers.Dense(d_model),  # (batch_size, seq_len, d_model)
        ]
    )


def _create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = _create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = _create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = _create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = _create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask
