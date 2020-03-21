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

    def test_create_dataset(self):
        dl = dataloader.Dataloader("tests/sample1.txt", "tests/sample2.txt", 300)
        dataset = dl.create_dataset()

        samples_num = 0
        for sample in dataset.padded_batch(1, padded_shapes=([None], [None])):
            samples_num += 1
            self.assertTrue(sample is not None)

        self.assertEqual(25, samples_num)
