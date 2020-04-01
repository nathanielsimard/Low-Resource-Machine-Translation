from typing import List

import tensorflow as tf

from src import logging, preprocessing, text_encoder

logger = logging.create_logger(__name__)

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
        text_encoder_type: text_encoder.TextEncoderType,
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

        self.corpus = preprocessing.add_start_end_token(reversed(self.corpus))

        if self.encoder is None:
            self.encoder = text_encoder.create_encoder(
                file_name,
                self.corpus,
                self.vocab_size,
                text_encoder_type,
                cache_dir=self.cache_dir,
            )

    def create_dataset(self) -> tf.data.Dataset:
        """Create a Tensorflow dataset."""

        def gen():
            for i in self.corpus:
                if self.max_seq_lenght is not None:
                    drop_char_len = len(i) - self.max_seq_lenght
                    if drop_char_len > 0:
                        i = (
                            i[: self.max_seq_lenght]
                            + " "
                            + preprocessing.END_OF_SAMPLE_TOKEN
                        )
                        logger.info(
                            f"{drop_char_len} characters were cut from the line."
                        )

                yield self.encoder.encode(i)

        return tf.data.Dataset.from_generator(gen, tf.int64)


class AlignedDataloader:
    """AlignedDataloader class used for translation."""

    def __init__(
        self,
        file_name_input: str,
        file_name_target: str,
        vocab_size: int,
        text_encoder_type: text_encoder.TextEncoderType,
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
            text_encoder_type: Type of text encoder to use.
            cache_dir: Cache directory for the encoders.
            encoder_input: English tokenizer.
            encoder_target: French tokenizer.
            corpus_input: The corpus lang1,
            corpus_target: the corpus lang2,
            max_seq_lenght: The maximum seuqnce lenght of a sample in both corpuses.
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

        self.corpus_input = preprocessing.add_start_end_token(
            reversed(self.corpus_input)
        )
        self.corpus_target = preprocessing.add_start_end_token(
            reversed(self.corpus_target)
        )

        if self.encoder_input is None:
            self.encoder_input = text_encoder.create_encoder(
                file_name_input,
                self.corpus_input,
                self.vocab_size,
                text_encoder_type,
                cache_dir=self.cache_dir,
            )

        if self.encoder_target is None:
            self.encoder_target = text_encoder.create_encoder(
                file_name_target,
                self.corpus_target,
                self.vocab_size,
                text_encoder_type,
                cache_dir=self.cache_dir,
            )

    def create_dataset(self) -> tf.data.Dataset:
        """Create a Tensorflow dataset."""

        def gen():
            for i, o in zip(self.corpus_input, self.corpus_target):
                if self.max_seq_lenght is not None:
                    i_drop_char_len = len(i) - self.max_seq_lenght
                    o_drop_char_len = len(o) - self.max_seq_lenght

                    if i_drop_char_len > 0:
                        i = (
                            i[: self.max_seq_lenght]
                            + " "
                            + preprocessing.END_OF_SAMPLE_TOKEN
                        )
                        logger.info(
                            f"{i_drop_char_len} characters were cut from the input line."
                        )

                    if o_drop_char_len > 0:
                        o = (
                            o[: self.max_seq_lenght]
                            + " "
                            + preprocessing.END_OF_SAMPLE_TOKEN
                        )
                        logger.info(
                            f"{o_drop_char_len} characters were cut from the output line."
                        )

                encoder_input = self.encoder_input.encode(i)
                encoder_target = self.encoder_target.encode(o)

                yield (encoder_input, encoder_target)

        return tf.data.Dataset.from_generator(gen, (tf.int64, tf.int64))


def read_file(file_name: str) -> List[str]:
    """Read file and returns paragraphs."""
    logger.info(f"Reading file {file_name}")
    output = []
    with open(file_name, "r") as stream:
        for line in stream:
            tokens = line.strip()
            output.append(tokens)
    return output
