import unittest

from src import preprocessing
from src.text_encoder import WordTextEncoder

CORPUS = preprocessing.add_start_end_token(["a against", "battle", "pandemy"])
A_VOCAB_SIZE = 25
ANY_TEXT_FILE = "data/train.lang1"


class WordTextEncoderTest(unittest.TestCase):
    def test_can_save(self):
        encoder = WordTextEncoder(A_VOCAB_SIZE, CORPUS)
        file_name = "text_ecoder"
        encoder.save_to_file(file_name)
        loaded_encoder = WordTextEncoder.load_from_file(file_name)

        self.assertEqual(encoder, loaded_encoder)
