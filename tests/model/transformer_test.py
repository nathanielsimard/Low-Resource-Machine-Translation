import unittest

from src.model import transformer
from tests.model.base import VOCAB_SIZE, MachineTranslationModelTest


class TransformerTest(MachineTranslationModelTest, unittest.TestCase):
    def create_model(self):
        return transformer.Transformer(
            num_layers=4,
            num_heads=8,
            dff=512,
            d_model=256,
            input_vocab_size=VOCAB_SIZE + 1,
            target_vocab_size=VOCAB_SIZE + 1,
            pe_input=VOCAB_SIZE + 1,
            pe_target=VOCAB_SIZE + 1,
            rate=0.1,
        )
