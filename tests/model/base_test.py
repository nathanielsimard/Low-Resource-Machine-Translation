import unittest

import tensorflow as tf

from src.dataloader import UnalignedDataloader
from src.model.base import clean_sentences, translation_max_seq_lenght
from src.preprocessing import END_OF_SAMPLE_TOKEN, START_OF_SAMPLE_TOKEN
from src.text_encoder import TextEncoderType


class CleanSentences(unittest.TestCase):
    def test_clean_sentences(self):
        inputs = [
            f"{START_OF_SAMPLE_TOKEN} It starts like that {END_OF_SAMPLE_TOKEN} this must be cleared",
            f"{START_OF_SAMPLE_TOKEN} Another sentence {END_OF_SAMPLE_TOKEN} another garbage",
            f"{START_OF_SAMPLE_TOKEN} Another amazing sentence {END_OF_SAMPLE_TOKEN}another garbage, but this time without space.",
        ]
        actual = clean_sentences(inputs)
        self.assertEqual("It starts like that", actual[0])
        self.assertEqual("Another sentence", actual[1])
        self.assertEqual("Another amazing sentence", actual[2])


class MaxSeqLenght(unittest.TestCase):
    def test_(self):
        inputs = [
            "It starts like that very long sequence",
            "Another sentence",
        ]
        dataloader = UnalignedDataloader(
            "nothing", 10000, TextEncoderType.WORD, corpus=inputs
        )
        dataset = dataloader.create_dataset()

        for x in dataset.padded_batch(2, padded_shapes=[None]):
            max_seq_length = translation_max_seq_lenght(x, dataloader.encoder)

            self.assertEqual((2,), max_seq_length.shape)
            self.assertEqual(4, max_seq_length[0])
            self.assertEqual(10, max_seq_length[1])
            self.assertEqual(tf.int32, max_seq_length.dtype)
