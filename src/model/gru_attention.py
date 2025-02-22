from typing import Tuple

import numpy as np
import tensorflow as tf

from src import logging
from src.model import base
from src.text_encoder import TextEncoder

NAME = "gru-attention"


logger = logging.create_logger(__name__)


class Encoder(tf.keras.Model):
    """Encoder of the gru with attention model."""

    def __init__(self, vocab_size, embedding_dim, enc_units, dropout):
        """Create the encoder."""
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
            recurrent_dropout=dropout,
        )

    def call(self, x, hidden, training):
        """Call the foward past."""
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden, training=training)
        return output, state

    def initialize_hidden_state(self, batch_size):
        """Initialize the hidden state with zeros."""
        return tf.zeros((batch_size, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    """Attention layer used with the gru model."""

    def __init__(self, units):
        """Create the attention layer."""
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        """Call of the attention layer.

        Note that the call must be for one caracter/word at a time.
        """
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    """Decoder of the gru with attention model."""

    def __init__(self, vocab_size, embedding_dim, dec_units, dropout):
        """Create the decoder."""
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
            recurrent_dropout=dropout,
        )
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output, training):
        """Call the foward past.

        Note that the call must be for one caracter/word at a time.
        """
        # enc_output shape == (batch_size, seq_lenght, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, seq_lenght, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, seq_lenght, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x, training=training)

        # output shape == (batch_size * seq_lenght, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


class GRU(base.MachineTranslationModel):
    """Gru with Bahdanau attention."""

    def __init__(
        self,
        input_vocab_size: int,
        output_vocab_size: int,
        embedding_size: int,
        layers_size: int,
        dropout: float,
        attention_size: int,
    ):
        """Create the gru model."""
        super().__init__(NAME)
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size

        self.encoder = Encoder(input_vocab_size, embedding_size, layers_size, dropout)
        self.attention_layer = BahdanauAttention(attention_size)
        self.decoder = Decoder(output_vocab_size, embedding_size, layers_size, dropout)

        self._embedding_size = embedding_size

    def call(self, x: Tuple[tf.Tensor, tf.Tensor], training=False):
        """Call the foward past."""
        batch_size = x[0].shape[0]

        encoder_hidden = self.encoder.initialize_hidden_state(batch_size)
        encoder_output, encoder_hidden = self.encoder(x[0], encoder_hidden, training)

        decoder_hidden = encoder_hidden
        predictions = None

        seq_lenght = x[1].shape[1]
        for t in range(seq_lenght):
            # The decoder input is the word at timestep t
            previous_target_word = x[1][:, t]
            decoder_input = tf.expand_dims(previous_target_word, 1)

            # Call the decoder and update the decoder hidden state
            decoder_output, decoder_hidden, _ = self.decoder(
                decoder_input, decoder_hidden, encoder_output, training
            )

            # The predictions are concatenated on the time axis
            # The shape is (batch_size, seq_lenght, output_vocab_size)
            if predictions is None:
                predictions = tf.expand_dims(decoder_output, 1)
            else:
                decoder_output = tf.expand_dims(decoder_output, 1)
                predictions = tf.concat([predictions, decoder_output], axis=1)

        return predictions

    @property
    def embedding_size(self):
        """Embedding size."""
        return self._embedding_size

    def translate(
        self, x: tf.Tensor, encoder_inputs: TextEncoder, encoder_targets: TextEncoder
    ) -> tf.Tensor:
        """Translate a sentence from input."""
        batch_size = x.shape[0]
        max_seq_length = tf.reduce_max(
            base.translation_max_seq_lenght(x, encoder_inputs)
        )

        encoder_hidden = self.encoder.initialize_hidden_state(batch_size)
        encoder_output, encoder_hidden = self.encoder(x, encoder_hidden, False)
        decoder_hidden = encoder_hidden

        # The first words of each sentence in the batch is the start of sample token.
        words = (
            tf.zeros([batch_size, 1], dtype=tf.int64)
            + encoder_targets.start_of_sample_index
        )
        last_words = words

        has_finish_predicting = False
        reach_max_seq_lenght = False

        while not (has_finish_predicting or reach_max_seq_lenght):
            # Call the decoder and update the decoder hidden state
            decoder_output, decoder_hidden, _ = self.decoder(
                last_words, decoder_hidden, encoder_output, False
            )
            last_words = tf.expand_dims(decoder_output, 1)
            last_words = tf.math.argmax(last_words, axis=2)

            logger.debug(f"New word {last_words}.")

            # Append the newly predicted words into words.
            words = tf.concat([words, last_words], 1)

            # Compute the end condition of the while loop.
            end_of_sample = (
                np.zeros([batch_size, 1], dtype=np.int64)
                + encoder_targets.end_of_sample_index
            )
            has_finish_predicting = np.array_equal(last_words.numpy(), end_of_sample)
            reach_max_seq_lenght = words.shape[1] >= max_seq_length

            logger.debug(f"Has finish predicting {has_finish_predicting}.")
            logger.debug(f"Has reach max sequence length {reach_max_seq_lenght}.")

        return words

    @property
    def padded_shapes(self):
        """Padded shapes used to add padding when batching multiple sequences."""
        return (([None], [None]), [None])

    def preprocessing(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Proprocess dataset to have ((encoder_input, decoder_input), target)."""
        # Teacher forcing.
        def preprocess(input_sentence, output_sentence):
            return ((input_sentence, output_sentence[:-1]), output_sentence[1:])

        return dataset.map(preprocess)
