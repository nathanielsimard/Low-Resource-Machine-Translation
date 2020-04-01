import unittest

from src import dataloader, text_encoder

ANY_TEXT_FILE = "data/train.lang1"

NUM_SAMPLES = 25
VOCAB_SIZE = 300


class DataloaderTest(unittest.TestCase):
    def test_read_token_from_file(self):
        output = dataloader.read_file(ANY_TEXT_FILE)

        self.assertTrue(len(output) >= 1)
        self.assertTrue(len(output[0]) >= 1)

    def test_create_unaligned_dataset(self):
        dl = dataloader.UnalignedDataloader(
            "tests/sample1.txt", VOCAB_SIZE, text_encoder.TextEncoderType.SUBWORD,
        )

        dataset = dl.create_dataset()

        samples_num = self._count_entries(dataset, ([None]))
        self.assertEqual(NUM_SAMPLES, samples_num)

    def test_create_aligned_dataset(self):
        dl = dataloader.AlignedDataloader(
            "tests/sample1.txt",
            "tests/sample2.txt",
            VOCAB_SIZE,
            text_encoder.TextEncoderType.SUBWORD,
        )

        dataset = dl.create_dataset()

        samples_num = self._count_entries(dataset, ([None], [None]))
        self.assertEqual(NUM_SAMPLES, samples_num)

    def _count_entries(self, dataset, padded_shapes):
        samples_num = 0
        for sample in dataset.padded_batch(1, padded_shapes):
            samples_num += 1
            self.assertTrue(sample is not None)

        return samples_num
