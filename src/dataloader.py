from typing import List

import tensorflow as tf
import tensorflow_datasets as tfds


class Dataloader:
    def __init__(self, file_name_input: str, file_name_output: str, vocab_size: int):
        self.file_name_input = file_name_input
        self.file_name_output = file_name_output
        self.vocab_size = vocab_size

    def create_dataset(self) -> tf.data.Dataset:
        """Create the Tensorflow dataset."""
        corpus_input = read_file(self.file_name_input)
        corpus_output = read_file(self.file_name_output)

        encoder_input = create_encoder(corpus_input, self.vocab_size)
        encoder_ouput = create_encoder(corpus_output, self.vocab_size)

        def gen():
            for i, o in zip(corpus_input, corpus_output):
                yield (encoder_input.encode(i), encoder_ouput.encode(o))

        return tf.data.Dataset.from_generator(gen, (tf.int64, tf.int64))


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
