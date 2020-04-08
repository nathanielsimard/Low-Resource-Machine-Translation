import unittest
from unittest import mock
import numpy as np

import tensorflow as tf

from src.training.pretraining import Pretraining

MASKED_TOKEN_INDEX = 3
UNHANDLE_TOKEN_INDEX = 0
INPUTS = tf.convert_to_tensor([[8, 15, 8, 10, 1, 28, 11], [1, 5, 2, 8, 9, 9, 20]])
RANDOM_TOKENS = tf.convert_to_tensor(
    [[16, 10, 4, 1, 15, 89, 20], [8, 4, 3, 9, 5, 3, 2]]
)


class PretrainingTest(unittest.TestCase):
    def setUp(self):
        fake_model = mock.Mock()
        fake_train_dl = mock.Mock()
        fake_valid_dl = mock.Mock()
        fake_optimizer = mock.Mock()
        fake_loss = mock.Mock()
        encoder = mock.Mock()
        encoder.vocab_size = 200
        encoder.mask_token_index = MASKED_TOKEN_INDEX
        fake_train_dl.encoder = encoder

        self.session = Pretraining(
            fake_model, fake_train_dl, fake_valid_dl, fake_loss, fake_optimizer
        )

    def test_apply_masked_token_index(self):
        mask = create_mask(indexes=[(0, 0), (0, 1), (1, 2), (1, 5)])
        masked_token_mask = create_mask(indexes=[(0, 0), (0, 4), (1, 1), (1, 2)])

        masked_inputs = self.session.apply_masks(
            mask,
            masked_token_mask,
            create_mask(),
            create_mask(),
            INPUTS,
            RANDOM_TOKENS,
        )

        expected_inputs = INPUTS.numpy()
        expected_inputs[0, 0] = MASKED_TOKEN_INDEX
        expected_inputs[0, 1] = UNHANDLE_TOKEN_INDEX
        expected_inputs[1, 2] = MASKED_TOKEN_INDEX
        expected_inputs[1, 5] = UNHANDLE_TOKEN_INDEX
        expected_inputs = tf.constant(expected_inputs)

        self.assertTrue(tf.reduce_all(expected_inputs == masked_inputs))

    def test_apply_mask_unchanged_mask(self):
        mask = create_mask(indexes=[(0, 0), (0, 1), (1, 2), (1, 5)])
        unchanged_mask = create_mask(indexes=[(0, 0), (0, 4), (1, 1), (1, 2)])

        masked_inputs = self.session.apply_masks(
            mask, create_mask(), create_mask(), unchanged_mask, INPUTS, RANDOM_TOKENS,
        )

        expected_inputs = INPUTS.numpy()
        expected_inputs[0, 1] = UNHANDLE_TOKEN_INDEX
        expected_inputs[1, 5] = UNHANDLE_TOKEN_INDEX
        expected_inputs = tf.constant(expected_inputs)

        self.assertTrue(tf.reduce_all(expected_inputs == masked_inputs))

    def test_apply_mask_random_word(self):
        mask = create_mask(indexes=[(0, 0), (0, 1), (1, 2), (1, 5)])
        random_token_mask = create_mask(indexes=[(0, 0), (0, 4), (1, 1), (1, 2)])

        masked_inputs = self.session.apply_masks(
            mask,
            create_mask(),
            random_token_mask,
            create_mask(),
            INPUTS,
            RANDOM_TOKENS,
        )

        expected_inputs = INPUTS.numpy()
        expected_inputs[0, 0] = RANDOM_TOKENS[0, 0]
        expected_inputs[0, 1] = UNHANDLE_TOKEN_INDEX
        expected_inputs[1, 2] = RANDOM_TOKENS[1, 2]
        expected_inputs[1, 5] = UNHANDLE_TOKEN_INDEX
        expected_inputs = tf.constant(expected_inputs)

        self.assertTrue(tf.reduce_all(expected_inputs == masked_inputs))

    def test_create_and_apply_mask(self):
        masked_inputs, mask = self.session.create_and_apply_masks(INPUTS)
        self.assertIsNotNone(mask)
        self.assertFalse(tf.reduce_all(masked_inputs == INPUTS))


def create_mask(shape=INPUTS.shape, indexes=[]):
    """Returns a tensor full of False."""
    mask = np.ones(shape) < 0
    for i, j in indexes:
        mask[i, j] = True
    return tf.convert_to_tensor(mask)
