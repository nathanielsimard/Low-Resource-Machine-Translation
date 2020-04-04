import unittest
import tensorflow as tf
from src.training.pretraining import Pretraining
from unittest import mock

MASKED_TOKEN_INDEX = 3


class PretrainingTest(unittest.TestCase):
    def test_apply_masked_token_index(self):
        fake_model = mock.Mock()
        fake_train_dl = mock.Mock()
        fake_valid_dl = mock.Mock()
        encoder = mock.Mock()
        encoder.mask_token_index = MASKED_TOKEN_INDEX
        fake_train_dl.encoder = encoder

        session = Pretraining(fake_model, fake_train_dl, fake_valid_dl)
        inputs = tf.constant([[8, 15, 8, 10, 1, 28, 11], [1, 5, 2, 8, 9, 9, 20]])

        mask = tf.constant(
            [
                [True, True, False, False, False, False, False],
                [False, False, True, False, False, True, False],
            ]
        )
        masked_token_mask = tf.constant(
            [
                [True, False, False, False, True, False, True],
                [False, True, True, False, False, False, True],
            ]
        )
        other_mask = tf.ones(mask.shape) < 0
        masked_inputs = session.apply_masks(
            mask, masked_token_mask, other_mask, other_mask, inputs, inputs
        )

        expected_inputs = inputs
        expected_inputs[0, 0] = MASKED_TOKEN_INDEX
        expected_inputs[1, 2] = MASKED_TOKEN_INDEX

        self.assertTrue(tf.reduce_all(expected_inputs == masked_inputs))
