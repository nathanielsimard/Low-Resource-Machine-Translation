import os
from typing import List

import tensorflow as tf
import tensorflow_datasets as tfds


class Dataloader:
    def __init__(
        self,
        file_name_input: str,
        file_name_output: str,
        vocab_size: int,
        cache_dir=".cache",
    ):
        self.file_name_input = file_name_input
        self.file_name_output = file_name_output
        self.vocab_size = vocab_size
        self.cache_dir = cache_dir

    def create_dataset(self) -> tf.data.Dataset:
        """Create the Tensorflow dataset."""
        corpus_input = read_file(self.file_name_input)
        corpus_output = read_file(self.file_name_output)

        encoder_input = self._create_cached_encoder(self.file_name_input, corpus_input)
        encoder_ouput = self._create_cached_encoder(
            self.file_name_output, corpus_output
        )

        def gen():
            for i, o in zip(corpus_input, corpus_output):
                yield (encoder_input.encode(i), encoder_ouput.encode(o))

        return tf.data.Dataset.from_generator(gen, (tf.int64, tf.int64))

    def _create_cached_encoder(self, file_name, corpus):
        directory = f"{self.cache_dir}/{file_name}"
        os.makedirs(directory, exist_ok=True)
        return create_encoder(
            corpus,
            self.vocab_size,
            cache_file=os.path.join(directory, str(self.vocab_size)).format(),
        )


def read_file(file_name: str) -> List[str]:
    """Read file and returns paragraphs."""
    output = []
    with open(file_name, "r") as stream:
        for line in stream:
            tokens = line.strip()
            output.append(tokens)
    return output


def create_encoder(
    sentences: List[str], max_vocab_size: int, cache_file=None
) -> tfds.features.text.TextEncoder:
    """Create the encoder from sentences."""
    if cache_file is not None and os.path.isfile(cache_file):
        return tfds.features.text.SubwordTextEncoder.load_from_file(cache_file)

    encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (sentence for sentence in sentences), target_vocab_size=max_vocab_size
    )

    if cache_file is not None:
        encoder.save_to_file(cache_file)

    return encoder
