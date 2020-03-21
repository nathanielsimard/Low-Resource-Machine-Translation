import tensorflow as tf
import tensorflow_datasets as tfds
from typing import List


def read_file(file_name: str) -> List[str]:
    """Read file and returns paragraphs."""
    output = []
    with open(file_name, "r") as stream:
        for line in stream:
            tokens = line.strip()
            output.append(tokens)
    return output


def create_encoder(
    sentences: List[str], max_vocab_size: int
) -> tfds.features.text.TextEncoder:
    """Create the encoder from sentences."""
    return tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (sentence for sentence in sentences), target_vocab_size=max_vocab_size
    )


def create_dataset(
    encoder: tfds.features.text.TextEncoder, data: List[str]
) -> tf.data.Dataset:
    """Create the Tensorflow dataset."""
