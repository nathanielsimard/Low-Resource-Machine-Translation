import os
from typing import List

import tensorflow as tf
import tensorflow_datasets as tfds

EMPTY_TOKEN = "<empty>"
EMPTY_TOKEN_INDEX = 1

START_OF_SAMPLE_TOKEN = "<start>"
START_OF_SAMPLE_TOKEN_INDEX = 2

END_OF_SAMPLE_TOKEN = "<end>"
END_OF_SAMPLE_TOKEN_INDEX = 3


class UnalignedDataloader:
    """Dataloader used for a single unaligned dataset."""

    def __init__(
        self,
        file_name: str,
        vocab_size: int,
        cache_dir=".cache",
        encoder=None,
        corpus=None,
        max_seq_lenght=None,
    ):
        """Create the UnalignedDataloader.

        If no corpus of encoder are passed, new ones are created.
        """
        self.file_name = file_name
        self.vocab_size = vocab_size
        self.cache_dir = cache_dir
        self.encoder = encoder
        self.corpus = corpus
        self.max_seq_lenght = max_seq_lenght

        if self.corpus is None:
            self.corpus = read_file(file_name)

        if self.encoder is None:
            self.encoder = _create_cached_encoder(
                file_name, self.corpus, self.cache_dir, self.vocab_size
            )
        self.corpus = reversed(corpus)

    def create_dataset(self) -> tf.data.Dataset:
        """Create a Tensorflow dataset."""

        def gen():
            for i in self.corpus:
                if self.max_seq_lenght is not None:
                    i = i[: self.max_seq_lenght]

                yield self.encoder.encode(
                    START_OF_SAMPLE_TOKEN + " " + i + " " + END_OF_SAMPLE_TOKEN
                )

        return tf.data.Dataset.from_generator(gen, tf.int64)


class AlignedDataloader:
    """AlignedDataloader class used for translation."""

    def __init__(
        self,
        file_name_input: str,
        file_name_target: str,
        vocab_size: int,
        cache_dir=".cache",
        encoder_input=None,
        encoder_target=None,
        corpus_input=None,
        corpus_target=None,
        max_seq_lenght=None,
    ):
        """Create dataset for translation.

        Args:
            file_name_input: File name to the input data.
            file_name_target: File name to the target data.
            vocab_size: maximum vocabulary size.
            cache_dir: Cache directory for the encoders.
            encoder_input: English tokenizer.
            encoder_target: French tokenizer.
            corpus_input: The corpus lang1,
            corpus_target: the corpus lang2,
        """
        self.file_name_input = file_name_input
        self.file_name_target = file_name_target
        self.vocab_size = vocab_size
        self.cache_dir = cache_dir
        self.encoder_input = encoder_input
        self.encoder_target = encoder_target
        self.corpus_input = corpus_input
        self.corpus_target = corpus_target
        self.max_seq_lenght = max_seq_lenght

        if self.corpus_input is None:
            self.corpus_input = read_file(file_name_input)

        if self.corpus_target is None:
            self.corpus_target = read_file(file_name_target)

        if self.encoder_input is None:
            self.encoder_input = _create_cached_encoder(
                file_name_input, self.corpus_input, self.cache_dir, self.vocab_size
            )

        if self.encoder_target is None:
            self.encoder_target = _create_cached_encoder(
                file_name_target, self.corpus_target, self.cache_dir, self.vocab_size
            )

        self.corpus_input = reversed(corpus_input)
        self.corpus_target = reversed(corpus_target)

    def create_dataset(self) -> tf.data.Dataset:
        """Create a Tensorflow dataset."""

        def gen():
            for i, o in zip(self.corpus_input, self.corpus_target):
                if self.max_seq_lenght is not None:
                    i = i[: self.max_seq_lenght]
                    o = o[: self.max_seq_lenght]

                encoder_input = self.encoder_input.encode(
                    START_OF_SAMPLE_TOKEN + " " + i + " " + END_OF_SAMPLE_TOKEN
                )
                encoder_target = self.encoder_target.encode(
                    START_OF_SAMPLE_TOKEN + " " + o + " " + END_OF_SAMPLE_TOKEN
                )

                yield (encoder_input, encoder_target)

        return tf.data.Dataset.from_generator(gen, (tf.int64, tf.int64))


def _create_cached_encoder(file_name, corpus, cache_dir, vocab_size):
    directory = os.path.join(cache_dir, file_name)
    os.makedirs(directory, exist_ok=True)

    return create_encoder(
        corpus,
        vocab_size,
        cache_file=os.path.join(directory, str(vocab_size)).format(),
    )


def read_file(file_name: str) -> List[str]:
    """Read file and returns paragraphs."""
    print(f"Reading file {file_name}")
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
    if cache_file is not None and os.path.isfile(cache_file + ".subwords"):
        print(f"Loading cache encoder {cache_file}")
        return tfds.features.text.SubwordTextEncoder.load_from_file(cache_file)

    print("Creating new encoder")
    # The empty token must be at first because the padded batch
    # add zero padding, which will be understood by the network as
    # empty words.
    encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (sentence for sentence in sentences),
        target_vocab_size=max_vocab_size,
        reserved_tokens=[EMPTY_TOKEN, START_OF_SAMPLE_TOKEN, END_OF_SAMPLE_TOKEN],
    )

    if cache_file is not None:
        print(f"Saving encoder {cache_file}")
        encoder.save_to_file(cache_file)

    return encoder
