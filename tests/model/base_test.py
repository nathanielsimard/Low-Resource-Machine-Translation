import unittest

from src.model.base import clean_sentences
from src.preprocessing import END_OF_SAMPLE_TOKEN, START_OF_SAMPLE_TOKEN


class CleanSentences(unittest.TestCase):
    def test_clean_sentences(self):
        inputs = [
            f"{START_OF_SAMPLE_TOKEN} It starts like that {END_OF_SAMPLE_TOKEN} this must be cleared",
            f"{START_OF_SAMPLE_TOKEN} Another sentence {END_OF_SAMPLE_TOKEN} another garbage",
        ]
        actual = clean_sentences(inputs)
        self.assertEqual("It starts like that", actual[0])
        self.assertEqual("Another sentence", actual[1])
