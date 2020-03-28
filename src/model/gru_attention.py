from typing import Tuple

import tensorflow as tf

from src.model import base

NAME = "GRU-Attention"


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self, batch_size):
        return tf.zeros((batch_size, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
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
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, seq_lenght, hidden_size)
        seq_lenght = x.shape[1]
        print(f"Input shape {x.shape}")
        context_vector, attention_weights = self.attention(hidden, enc_output)
        print(f"Context shape {context_vector.shape}")
        print(f"Attention shape {attention_weights.shape}")

        # x shape after passing through embedding == (batch_size, seq_lenght, embedding_dim)
        x = self.embedding(x)
        print(f"Embedding shape {x.shape}")

        # x shape after concatenation == (batch_size, seq_lenght, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, seq_lenght), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * seq_lenght, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


class GRU(base.MachineTranslationModel):
    def __init__(self, input_vocab_size: int, output_vocab_size: int):
        super().__init__(NAME)
        self.encoder = Encoder(input_vocab_size, 256, 1024)
        self.attention_layer = BahdanauAttention(10)
        self.decoder = Decoder(output_vocab_size, 256, 1024)

    def call(self, x: Tuple[tf.Tensor, tf.Tensor], training=False):
        batch_size = x[0].shape[0]

        encoder_hidden = self.encoder.initialize_hidden_state(batch_size)
        encoder_output, encoder_hidden = self.encoder(x[0], encoder_hidden)

        decoder_hidden = encoder_hidden
        decoder_output, decoder_hidden, _ = self.decoder(
            x[1], decoder_hidden, encoder_output
        )

        return decoder_output

    def translate(self, x: tf.Tensor) -> tf.Tensor:
        """Translate a sentence from input."""
        pass

    @property
    def padded_shapes(self):
        """Padded shapes used to add padding when batching multiple sequences."""
        return (([None], [None]), [None])

    def preprocessing(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Proprocess dataset to have ((encoder_input, decoder_input), target)."""

        def preprocess(input_sentence, output_sentence):
            return ((input_sentence, output_sentence[:-1]), output_sentence[1:])

        return dataset.map(preprocess)
