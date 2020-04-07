import unittest

from src.model import gru_attention
from tests.model.base import VOCAB_SIZE, MachineTranslationModelTest


class GruTest(MachineTranslationModelTest, unittest.TestCase):
    def create_model(self):
        return gru_attention.GRU(VOCAB_SIZE + 1, VOCAB_SIZE + 1, 128, 128, 0.3, 10)
