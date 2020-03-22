import os
from typing import List

import tensorflow as tf
import tensorflow_datasets as tfds

UNKNOWN_TOKEN = "<unk>"
END_OF_SAMPLE_TOKEN = "<eos>"


class Dataloader:
    """TranslationDataloader class used for translation."""

    def __init__(
        self,
        file_name_input: str,
        file_name_target: str,
        vocab_size: int,
        cache_dir=".cache",
        encoder_input=None,
        encoder_target=None,
    ):
        """Create dataset for translation.

        Args:
            file_name_input: File name to the input data.
            file_name_target: File name to the target data.
            vocab_size: maximum vocabulary size.
            cache_dir: Cache directory for the encoders.
            encoder_input: English tokenizer.
            encoder_target: French tokenizer.
        """
        self.file_name_input = file_name_input
        self.file_name_target = file_name_target
        self.vocab_size = vocab_size
        self.cache_dir = cache_dir
        self.encoder_input = encoder_input
        self.encoder_target = encoder_target

        self.corpus_input = read_file(file_name_input)
        self.corpus_target = read_file(file_name_target)

        if self.encoder_input is None:
            self.encoder_input = self._create_cached_encoder(
                file_name_input, self.corpus_input
            )

        if self.encoder_target is None:
            self.encoder_target = self._create_cached_encoder(
                file_name_target, self.corpus_target
            )

    def create_dataset(self) -> tf.data.Dataset:
        """Create a Tensorflow dataset."""

        def gen():
            for i, o in zip(self.corpus_input, self.corpus_target):
                encoder_input = self.encoder_input.encode(i + " " + END_OF_SAMPLE_TOKEN)
                encoder_target = self.encoder_target.encode(
                    o + " " + END_OF_SAMPLE_TOKEN
                )

                yield (encoder_input, encoder_target)

        return tf.data.Dataset.from_generator(gen, (tf.int64, tf.int64))

    def _create_cached_encoder(self, file_name, corpus):
        directory = os.path.join(self.cache_dir, file_name)
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

    # The unknown token must be at first because the padded batch
    # add zero padding, which will be understood by the network as
    # unknown words.
    encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (sentence for sentence in sentences),
        target_vocab_size=max_vocab_size,
        reserved_tokens=[UNKNOWN_TOKEN, END_OF_SAMPLE_TOKEN],
    )

    if cache_file is not None:
        encoder.save_to_file(cache_file)

    return encoder
