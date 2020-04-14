from typing import Tuple

import numpy as np
import tensorflow as tf

from src import logging
from src.model import base
from src.text_encoder import TextEncoder

NAME = "lstm_luong_attention"

logger = logging.create_logger(__name__)


class Encoder(tf.keras.Model):
    """Encoder of the seq2seq with attention model."""

    def __init__(self, vocab_size, embedding_dim, lstm_size):
        """Create the encoder."""
        super(Encoder, self).__init__()
        self.lstm_size = lstm_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(
            self.lstm_size, return_sequences=True, return_state=True
        )

    def call(self, x, hidden):
        """Call the foward past."""
        x = self.embedding(x)
        output, state_h, state_c = self.lstm(x, initial_state=hidden)
        return output, state_h, state_c

    def initialize_hidden_state(self, batch_size):
        """Initialize the hidden state with zeros."""
        return (
            tf.zeros([batch_size, self.lstm_size]),
            tf.zeros([batch_size, self.lstm_size]),
        )


class LuongAttention(tf.keras.Model):
    """Attention layer used with the gru model."""

    def __init__(self, rnn_size, attention_func="dot"):
        """Create the attention layer."""
        super(LuongAttention, self).__init__()
        self.attention_func = attention_func

        if attention_func not in ["dot", "general", "concat"]:
            raise ValueError(
                "Unknown attention score function! Must be either dot, general or concat."
            )

        if attention_func == "general":
            self.wa = tf.keras.layers.Dense(rnn_size)

        elif attention_func == "concat":
            self.wa = tf.keras.layers.Dense(rnn_size, activation="tanh")
            self.va = tf.keras.layers.Dense(1)

    def call(self, decoder_output, encoder_output):
        """Call of the attention layer.

        Note that the call must be for one caracter/word at a time.
        """
        if self.attention_func == "dot":
            score = tf.matmul(decoder_output, encoder_output, transpose_b=True)

        elif self.attention_func == "general":
            score = tf.matmul(decoder_output, self.wa(encoder_output), transpose_b=True)

        elif self.attention_func == "concat":
            decoder_output = tf.tile(decoder_output, [1, encoder_output.shape[1], 1])
            score = self.va(
                self.wa(tf.concat((decoder_output, encoder_output), axis=-1))
            )
            score = tf.transpose(score, [0, 2, 1])

        alignment = tf.nn.softmax(score, axis=2)
        context = tf.matmul(alignment, encoder_output)

        return context, alignment


class Decoder(tf.keras.Model):
    """Decoder of the gru with attention model."""

    def __init__(self, vocab_size, embedding_dim, rnn_size, attention_func="dot"):
        """Create the decoder."""
        super(Decoder, self).__init__()
        self.attention = LuongAttention(rnn_size, attention_func)
        self.rnn_size = rnn_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(
            rnn_size, return_sequences=True, return_state=True
        )
        self.wc = tf.keras.layers.Dense(rnn_size, activation="tanh")
        self.ws = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden, enc_output):
        """Call the foward past.

        Note that the call must be for one caracter/word at a time.
        """
        # x shape after passing through embedding == (batch_size, seq_lenght, embedding_dim)
        x = self.embedding(x)

        # passing the concatenated vector to the lstm
        output, state_h, state_c = self.lstm(x, initial_state=hidden)

        # enc_output shape == (batch_size, seq_lenght, hidden_size)
        context_vector, alignment = self.attention(output, enc_output)

        output = tf.concat([tf.squeeze(context_vector, 1), tf.squeeze(output, 1)], 1)

        # lstm_out now has shape (batch_size, rnn_size)
        output = self.wc(output)

        # Finally, it is converted back to vocabulary space: (batch_size, vocab_size)
        logits = self.ws(output)

        return logits, state_h, state_c, alignment


class LSTM_ATTENTION(base.MachineTranslationModel):
    """LSTM with luong attention."""

    def __init__(self, input_vocab_size: int, output_vocab_size: int):
        """Create the lstm model."""
        super().__init__(NAME)
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size

        self.encoder = Encoder(input_vocab_size, 256, 512)
        self.attention_layer = LuongAttention(10)
        self.decoder = Decoder(output_vocab_size, 256, 512)

    def call(self, x: Tuple[tf.Tensor, tf.Tensor], training=False):
        """Call the foward past."""
        batch_size = x[0].shape[0]

        encoder_hidden = self.encoder.initialize_hidden_state(batch_size)
        encoder_output, encoder_state_h, encoder_state_c = self.encoder(
            x[0], encoder_hidden
        )

        decoder_state_h, decoder_state_c = encoder_state_h, encoder_state_c
        predictions = None

        seq_lenght = x[1].shape[1]
        for t in range(seq_lenght):
            # The decoder input is the word at timestep t
            previous_target_word = x[1][:, t]
            decoder_input = tf.expand_dims(previous_target_word, 1)

            # Call the decoder and update the decoder hidden state
            decoder_output, decoder_state_h, decoder_state_c, _ = self.decoder(
                decoder_input, (decoder_state_h, decoder_state_c), encoder_output
            )

            # The predictions are concatenated on the time axis
            # The shape is (batch_size, seq_lenght, output_vocab_size)
            if predictions is None:
                predictions = tf.expand_dims(decoder_output, 1)
            else:
                decoder_output = tf.expand_dims(decoder_output, 1)
                predictions = tf.concat([predictions, decoder_output], axis=1)

        return predictions

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
    def embedding_size(self):
        """Embedding size."""
        return 256

    @property
    def padded_shapes(self):
        """Padded shapes used to add padding when batching multiple sequences."""
        return (([None], [None]), [None])

    def preprocessing(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Proprocess dataset to have ((encoder_input, decoder_input), target)."""

        def preprocess(input_sentence, output_sentence):
            return ((input_sentence, output_sentence[:-1]), output_sentence[1:])

        return dataset.map(preprocess)
