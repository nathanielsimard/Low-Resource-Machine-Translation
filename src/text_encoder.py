import abc
import enum
import os
import pickle
from typing import Any, List

import tensorflow as tf
import tensorflow_datasets as tfds

from src import preprocessing


class TextEncoderType(enum.Enum):
    """Supported Text Encoder."""

    SUBWORD = "subword"
    WORD = "word"


class TextEncoder(abc.ABC):
    """Abstract Text Enoder to encoder and decode text into and from numbers."""

    @abc.abstractmethod
    def encode(self, texts: str) -> List[int]:
        """Encode a text into numbers."""
        pass

    def decode(self, sequences: List[int]) -> str:
        """Decode numbers into text."""
        pass

    @abc.abstractproperty
    def vocab_size(self) -> int:
        """The vocabulary size handled by the encoder."""
        pass

    @abc.abstractproperty
    def start_of_sample_index(self) -> int:
        """The index representing the start of sample token."""
        pass

    @abc.abstractproperty
    def end_of_sample_index(self) -> int:
        """The index representing the end of sample token."""
        pass

    @abc.abstractclassmethod
    def type(cls):
        pass

    def save_to_file(self, file_name: str):
        """Save the encoder to disk."""
        file_name = os.path.join(self.type().value, file_name)
        print(f"Saving text encoder to file {file_name}.")

        with open(file_name, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load_from_file(cls, file_name: str):
        """Load the encoder from disk."""
        file_name = os.path.join(cls.type(), file_name)
        print(f"Loading text encoder of type {cls} from file {file_name}.")
        with open(file_name, "rb") as file:
            return pickle.load(file)


class WordTextEncoder(TextEncoder):
    """Text Encoder where most popular words become tokens."""

    def __init__(self, vocab_size: int, corpus: List[str]):
        """Create the encoder using the keras tokenizer."""
        print("Creating new word text encoder.")
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=vocab_size, oov_token=preprocessing.OUT_OF_SAMPLE_TOKEN
        )
        self.tokenizer.fit_on_texts(corpus)
        self._vocab_size = vocab_size

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.texts_to_sequences([text])[0]

    def decode(self, sequences: List[int]) -> str:
        return self.tokenizer.sequences_to_texts([sequences])[0]

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def start_of_sample_index(self) -> int:
        """The index representing the start of sample token."""
        return self.tokenizer.word_index[preprocessing.START_OF_SAMPLE_TOKEN]

    @property
    def end_of_sample_index(self) -> int:
        """The index representing the end of sample token."""
        return self.tokenizer.word_index[preprocessing.END_OF_SAMPLE_TOKEN]


class SubWordTextEncoder(TextEncoder):
    """Text Encoder where most popular subwords become tokens."""

    def __init__(self, vocab_size: int, corpus: List[str]):
        """Create the encoder using the tensorflow dataset corpus utilities."""
        print("Creating new subword text encoder.")
        self._encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (sentence for sentence in corpus),
            target_vocab_size=vocab_size,
            reserved_tokens=[preprocessing.OUT_OF_SAMPLE_TOKEN],
        )

    def encode(self, text: str) -> List[int]:
        return self._encoder.encode(text)

    def decode(self, sequences: List[int]) -> str:
        return self._encoder.decode(sequences)

    @property
    def vocab_size(self):
        return self._encoder.vocab_size

    @property
    def start_of_sample_index(self) -> int:
        """The index representing the start of sample token."""
        return self._encoder.encode(preprocessing.START_OF_SAMPLE_TOKEN)[0]

    @property
    def end_of_sample_index(self) -> int:
        """The index representing the end of sample token."""
        return self._encoder.encode(preprocessing.END_OF_SAMPLE_TOKEN)[0]


def create_encoder(
    file_name, corpus, vocab_size, text_encoder_type: TextEncoderType, cache_dir=None,
):
    """Create a text encoder of the given type to encode and decode text from and into tensor."""
    directory = os.path.join(cache_dir, file_name)
    os.makedirs(directory, exist_ok=True)

    clazz: Any = None
    if text_encoder_type == TextEncoderType.WORD:
        clazz = WordTextEncoder
    elif text_encoder_type == TextEncoderType.SUBWORD:
        clazz = SubWordTextEncoder
    else:
        raise Exception(f"Text Encoder Type {text_encoder_type} is not supported.")

    return _create_text_encoder(
        corpus,
        vocab_size,
        clazz,
        cache_file=os.path.join(directory, str(vocab_size)).format(),
    )


def _create_text_encoder(text: List[str], vocab_size: int, clazz, cache_file=None):
    if cache_file is not None and os.path.isfile(cache_file):
        return clazz.load_from_file(cache_file)

    encoder = clazz(vocab_size, text)

    if cache_file is not None:
        encoder.save_to_file(cache_file)

    return encoder
