import unittest
from unittest import mock

import numpy as np
import tensorflow as tf

from src.model import gru_attention

BATCH_SIZE = 2
VOCAB_SIZE = 20
SEQ_LENGHT = 10


class ModelTest(unittest.TestCase):
    def setUp(self):
        self.encoder = mock.Mock()
        self.encoder.start_of_sample_index = 1
        self.encoder.end_of_sample_index = 2

    def test_translate(self):
        sentences = np.random.randint(0, VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LENGHT))
        model = gru_attention.GRU(VOCAB_SIZE + 1, VOCAB_SIZE + 1)

        translated = model.translate(sentences, self.encoder)

        self.assertIsNotNone(translated)

    def test_foward(self):
        model = gru_attention.GRU(VOCAB_SIZE + 1, VOCAB_SIZE + 1)
        sentences = np.random.randint(0, VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LENGHT))

        def gen():
            for i in range(BATCH_SIZE):
                yield (sentences[i], sentences[i])

        dataset = tf.data.Dataset.from_generator(gen, (tf.int64, tf.int64))
        dataset = model.preprocessing(dataset)

        for inputs, targets in dataset.padded_batch(
            BATCH_SIZE, padded_shapes=model.padded_shapes
        ):
            output = model(inputs)
            self.assertIsNotNone(output)
