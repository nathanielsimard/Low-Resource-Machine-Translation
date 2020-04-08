import abc
from unittest import mock

import numpy as np
import tensorflow as tf

from src.model import base

BATCH_SIZE = 2
VOCAB_SIZE = 20
SEQ_LENGHT = 10

MAX_SEQ_LENGTH = 100


class MachineTranslationModelTest(abc.ABC):
    def setUp(self):
        self.encoder = mock.Mock()
        self.encoder.start_of_sample_index = 1
        self.encoder.end_of_sample_index = 2
        self.model = self.create_model()

    def test_translate(self):
        sentences = np.random.randint(0, VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LENGHT))

        translated = self.model.translate(sentences, self.encoder, MAX_SEQ_LENGTH)

        self.assertIsNotNone(translated)  # type: ignore

    def test_foward(self):
        sentences = np.random.randint(0, VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LENGHT))

        def gen():
            for i in range(BATCH_SIZE):
                yield (sentences[i], sentences[i])

        dataset = tf.data.Dataset.from_generator(gen, (tf.int64, tf.int64))
        dataset = self.model.preprocessing(dataset)

        for inputs, targets in dataset.padded_batch(
            BATCH_SIZE, padded_shapes=self.model.padded_shapes
        ):
            output = self.model(inputs)
            self.assertIsNotNone(output)  # type: ignore

    @abc.abstractmethod
    def create_model(self) -> base.MachineTranslationModel:
        pass
