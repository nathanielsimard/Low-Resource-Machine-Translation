import unittest

from src import dataloader

ANY_TEXT_FILE = "data/train.lang1"
CORPUS = ["a against", "battle", "pandemy"]


class DataloaderTest(unittest.TestCase):
    def test_read_token_from_file(self):
        output = dataloader.read_file(ANY_TEXT_FILE)

        self.assertTrue(len(output) >= 1)
        self.assertTrue(len(output[0]) >= 1)

    def test_create_encoder_from_sentences(self):
        sample = "a battle"
        encoder = dataloader.create_subword_encoder(CORPUS, 258)

        ids = encoder.encode(sample)
        out = encoder.decode(ids)

        self.assertEqual(out, sample)

    def test_create_encoder_from_sentences_with_cache(self):
        sample = "a battle"
        encoder = dataloader.create_subword_encoder(
            CORPUS, 258, cache_file="/tmp/cachetest"
        )

        ids = encoder.encode(sample)
        out = encoder.decode(ids)

        self.assertEqual(out, sample)

    def test_encoder_start_of_sentence_token(self):
        encoder = dataloader.create_subword_encoder(
            CORPUS, 258, cache_file="/tmp/cachetest"
        )
        ids = encoder.encode(dataloader.START_OF_SAMPLE_TOKEN + " battle")
        self.assertEqual(ids[0], dataloader.START_OF_SAMPLE_TOKEN_INDEX)

    def test_encoder_end_of_sentence_token(self):
        encoder = dataloader.create_subword_encoder(
            CORPUS, 258, cache_file="/tmp/cachetest"
        )
        ids = encoder.encode(dataloader.END_OF_SAMPLE_TOKEN + " battle")
        self.assertEqual(ids[0], dataloader.END_OF_SAMPLE_TOKEN_INDEX)

    def test_create_dataset(self):
        dl = dataloader.AlignedDataloader(
            "tests/sample1.txt",
            "tests/sample2.txt",
            300,
            dataloader.TextEncoderType.SUBWORD,
        )
        dataset = dl.create_dataset()

        samples_num = 0
        for sample in dataset.padded_batch(1, padded_shapes=([None], [None])):
            samples_num += 1
            self.assertTrue(sample is not None)

        self.assertEqual(25, samples_num)

    def test_word_encoder(self):
        sample = "a battle"
        text_encoder = dataloader.WordEncoder(10, CORPUS)
        ids = text_encoder.encode(sample)
        out = text_encoder.decode(ids)

        self.assertEqual(out, sample)

    def test_words_encoder_start_of_sentence_token(self):
        encoder = dataloader.WordEncoder(4, CORPUS)
        ids = encoder.encode(dataloader.START_OF_SAMPLE_TOKEN + " battle")
        self.assertEqual(ids[0], dataloader.START_OF_SAMPLE_TOKEN_INDEX)

    def test_words_encoder_end_of_sentence_token(self):
        encoder = dataloader.WordEncoder(4, CORPUS)
        ids = encoder.encode(dataloader.END_OF_SAMPLE_TOKEN + " battle")
        self.assertEqual(ids[0], dataloader.END_OF_SAMPLE_TOKEN_INDEX)
