import unittest
from src import dataloader

ANY_TEXT_FILE = "data/train.lang1"


class DataloaderTest(unittest.TestCase):
    def test_read_token_from_file(self):
        output = dataloader.read_file(ANY_TEXT_FILE)

        self.assertTrue(len(output) >= 1)
        self.assertTrue(len(output[0]) >= 1)

    def test_create_encoder_from_sentences(self):
        corpus = ["a against", "battle", "pandemy"]

        sample = "a battle"

        encoder = dataloader.create_encoder(corpus, 258)
        ids = encoder.encode(sample)
        out = encoder.decode(ids)
        self.assertEqual(out, sample)
