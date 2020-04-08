import abc
import enum
import os
import pickle
from typing import Any, List

import tensorflow as tf
import tensorflow_datasets as tfds

from src import logging, preprocessing

logger = logging.create_logger(__name__)


class TextEncoderType(enum.Enum):
    """Supported Text Encoder."""

    SUBWORD = "subword"
    WORD = "word"
    WORD_NO_FILTER = "word-no-filter"


class TextEncoder(abc.ABC):
    """Abstract Text Enoder to encoder and decode text into and from numbers.

    Indexes start at 1 and end at vocab_size included [1, vocab_size].
    """

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
    def type(cls) -> TextEncoderType:
        """Type of the text encoder."""
        pass

    def save_to_file(self, file_name: str):
        """Save the encoder to disk."""
        suffix = self.cls.type().value  # type: ignore
        file_name = f"{file_name}.{suffix}"

        logger.info(f"Saving text encoder to file {file_name}.")

        with open(file_name, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load_from_file(cls, file_name: str):
        """Load the encoder from disk."""
        suffix = cls.type().value  # type: ignore
        file_name = f"{file_name}.{suffix}"

        logger.info(f"Loading text encoder of type {cls} from file {file_name}.")
        with open(file_name, "rb") as file:
            return pickle.load(file)

    @abc.abstractmethod
    def vocabulary(self) -> List[str]:
        """Return all the word tokens in the vocabulary."""
        pass


class WordTextEncoder(TextEncoder):
    """Text Encoder where most popular words become tokens."""

    def __init__(
        self,
        vocab_size: int,
        corpus: List[str],
        filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n',
    ):
        """Create the encoder using the keras tokenizer."""
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=vocab_size,
            oov_token=preprocessing.OUT_OF_SAMPLE_TOKEN,
            filters=filters,
        )
        self.tokenizer.fit_on_texts(corpus)

        self._vocab_size = vocab_size
        max_vocab_size = len(self.tokenizer.index_word.keys())
        if self._vocab_size > max_vocab_size:
            self._vocab_size = max_vocab_size

        self.cls = WordTextEncoder

    def encode(self, text: str) -> List[int]:
        """Encode a text into numbers."""
        return self.tokenizer.texts_to_sequences([text])[0]

    def decode(self, sequences: List[int]) -> str:
        """Decode numbers into text."""
        return self.tokenizer.sequences_to_texts([sequences])[0]

    @classmethod
    def type(cls):
        """Type of the text encoder."""
        return TextEncoderType.WORD

    @property
    def vocab_size(self):
        """The vocabulary size handled by the encoder."""
        return self._vocab_size

    @property
    def start_of_sample_index(self) -> int:
        """The index representing the start of sample token."""
        return self.tokenizer.word_index[preprocessing.START_OF_SAMPLE_TOKEN]

    @property
    def end_of_sample_index(self) -> int:
        """The index representing the end of sample token."""
        return self.tokenizer.word_index[preprocessing.END_OF_SAMPLE_TOKEN]

    def vocabulary(self) -> List[str]:
        """Return all the word tokens in the vocabulary."""
        word_tokens = []
        for i in range(1, self.vocab_size + 1):
            word_tokens.append(self.tokenizer.index_word[i])
        return word_tokens


class WordNoFilterTextEncoder(WordTextEncoder):
    """Text Encoder where most popular words become tokens.

    The corpus is expected to be already tokenized, so no
    filters need to be applied.
    """

    def __init__(self, vocab_size: int, corpus: List[str]):
        """Create the encoder like word but without filters."""
        super().__init__(vocab_size, corpus, filters="")
        self.cls = WordNoFilterTextEncoder

    @classmethod
    def type(cls):
        """Type of the text encoder."""
        return TextEncoderType.WORD_NO_FILTER


class SubWordTextEncoder(TextEncoder):
    """Text Encoder where most popular subwords become tokens."""

    def __init__(self, vocab_size: int, corpus: List[str]):
        """Create the encoder using the tensorflow dataset corpus utilities."""
        self._encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (sentence for sentence in corpus),
            target_vocab_size=vocab_size,
            reserved_tokens=[
                preprocessing.OUT_OF_SAMPLE_TOKEN,
                preprocessing.START_OF_SAMPLE_TOKEN,
                preprocessing.END_OF_SAMPLE_TOKEN,
                preprocessing.MASK_TOKEN,
            ],
        )
        self.cls = SubWordTextEncoder

    def encode(self, text: str) -> List[int]:
        """Encode a text into numbers."""
        return self._encoder.encode(text)

    def decode(self, sequences: List[int]) -> str:
        """Decode numbers into text."""
        # Handle 0 index as 1 for <out>
        sequences = [1 if i == 0 else i for i in sequences]
        return self._encoder.decode(sequences)

    @classmethod
    def type(cls):
        """Type of the text encoder."""
        return TextEncoderType.SUBWORD

    @property
    def vocab_size(self):
        """The vocabulary size handled by the encoder."""
        return self._encoder.vocab_size

    @property
    def start_of_sample_index(self) -> int:
        """The index representing the start of sample token."""
        return self._encoder.encode(preprocessing.START_OF_SAMPLE_TOKEN)[0]

    @property
    def end_of_sample_index(self) -> int:
        """The index representing the end of sample token."""
        return self._encoder.encode(preprocessing.END_OF_SAMPLE_TOKEN)[0]

    @property
    def mask_token_index(self) -> int:
        """The index representing the mask token."""
        return self._encoder.encode(preprocessing.MASK_TOKEN)[0]

    def vocabulary(self) -> List[str]:
        """Return all the word tokens in the vocabulary."""
        return self._encoder._subwords


def create_encoder(
    file_name, corpus, vocab_size, text_encoder_type: TextEncoderType, cache_dir=None,
):
    """Create a text encoder of the given type to encode and decode text from and into tensor."""
    cache_file = None

    if cache_dir is not None:
        directory = os.path.join(cache_dir, file_name)
        os.makedirs(directory, exist_ok=True)
        cache_file = os.path.join(directory, str(vocab_size)).format()

    clazz: Any = None
    if text_encoder_type == TextEncoderType.WORD:
        clazz = WordTextEncoder
    elif text_encoder_type == TextEncoderType.SUBWORD:
        clazz = SubWordTextEncoder
    elif text_encoder_type == TextEncoderType.WORD_NO_FILTER:
        clazz = WordNoFilterTextEncoder
    else:
        raise Exception(f"Text Encoder Type {text_encoder_type} is not supported.")

    return _create_text_encoder(corpus, vocab_size, clazz, cache_file=cache_file)


def _create_text_encoder(text: List[str], vocab_size: int, clazz, cache_file=None):
    if cache_file is None:
        encoder = clazz(vocab_size, text)
        logger.info(f"Created new text encoder of type {encoder.type()}.")
        return encoder

    try:
        return clazz.load_from_file(cache_file)
    except FileNotFoundError:
        return _create_text_encoder(text, vocab_size, clazz)
