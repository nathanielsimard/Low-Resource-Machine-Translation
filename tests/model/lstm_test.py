import unittest

from src.model import lstm
from tests.model.base import VOCAB_SIZE, MachineTranslationModelTest


class LstmTest(MachineTranslationModelTest, unittest.TestCase):
    def create_model(self):
        return lstm.Lstm(VOCAB_SIZE + 1, VOCAB_SIZE + 1)
