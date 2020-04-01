# type: ignore
import abc
import unittest

from src import preprocessing
from src.text_encoder import SubWordTextEncoder, TextEncoder, WordTextEncoder

CORPUS = preprocessing.add_start_end_token(
    ["a against", "battle", "pandemy covid-19 not easy"]
)
A_VOCAB_SIZE = 258
ANY_TEXT_FILE = "data/train.lang1"
SAMPLE = "a pandemy"


class TextEncoderTest(abc.ABC):
    def test_can_save_load(self):
        encoder = self.create_encoder()
        file_name = "text_ecoder"

        encoder.save_to_file(file_name)
        loaded_encoder = self.load_encoder(file_name)

        self.assertEqual(loaded_encoder.vocab_size, encoder.vocab_size)

    def test_can_encode_decode(self):
        encoder = self.create_encoder()

        ids = encoder.encode(SAMPLE)
        decoded = encoder.decode(ids)

        self.assertEqual(decoded, SAMPLE)

    def test_start_of_sample_index(self):
        encoder = self.create_encoder()

        ids = encoder.encode(preprocessing.START_OF_SAMPLE_TOKEN)

        self.assertEqual(1, len(ids))
        self.assertEqual(encoder.start_of_sample_index, ids[0])

    def test_end_of_sample_index(self):
        encoder = self.create_encoder()

        ids = encoder.encode(preprocessing.END_OF_SAMPLE_TOKEN)

        self.assertEqual(1, len(ids))
        self.assertEqual(encoder.end_of_sample_index, ids[0])

    @abc.abstractmethod
    def create_encoder(self) -> TextEncoder:
        pass

    @abc.abstractmethod
    def load_encoder(self, file_name) -> TextEncoder:
        pass


class WordTextEncoderTest(TextEncoderTest, unittest.TestCase):
    def create_encoder(self) -> TextEncoder:
        return WordTextEncoder(A_VOCAB_SIZE, CORPUS)

    def load_encoder(self, file_name) -> TextEncoder:
        return WordTextEncoder.load_from_file(file_name)


class SubwordTextEncoderTest(TextEncoderTest, unittest.TestCase):
    def create_encoder(self) -> TextEncoder:
        return SubWordTextEncoder(A_VOCAB_SIZE, CORPUS)

    def load_encoder(self, file_name) -> TextEncoder:
        return SubWordTextEncoder.load_from_file(file_name)
