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

        print(f"Saving text encoder to file {file_name}.")

        with open(file_name, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load_from_file(cls, file_name: str):
        """Load the encoder from disk."""
        suffix = cls.type().value  # type: ignore
        file_name = f"{file_name}.{suffix}"

        print(f"Loading text encoder of type {cls} from file {file_name}.")
        with open(file_name, "rb") as file:
            return pickle.load(file)

    @abc.abstractmethod
    def vocabulary(self) -> List[str]:
        """Return all the word tokens in the vocabulary."""
        pass


class WordTextEncoder(TextEncoder):
    """Text Encoder where most popular words become tokens."""

    def __init__(self, vocab_size: int, corpus: List[str]):
        """Create the encoder using the keras tokenizer."""
        print("Creating new word text encoder.")
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=vocab_size, oov_token=preprocessing.OUT_OF_SAMPLE_TOKEN
        )
        self.tokenizer.fit_on_texts(corpus)
        self._update_word(
            preprocessing.START_OF_SAMPLE_TOKEN[1:-1],
            preprocessing.START_OF_SAMPLE_TOKEN,
        )

        self._update_word(
            preprocessing.END_OF_SAMPLE_TOKEN[1:-1], preprocessing.END_OF_SAMPLE_TOKEN,
        )

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

    def _update_word(self, old_word, new_word):
        index = self.tokenizer.word_index[old_word]
        self.tokenizer.word_index[new_word] = index
        self.tokenizer.index_word[index] = new_word


class SubWordTextEncoder(TextEncoder):
    """Text Encoder where most popular subwords become tokens."""

    def __init__(self, vocab_size: int, corpus: List[str]):
        """Create the encoder using the tensorflow dataset corpus utilities."""
        print("Creating new subword text encoder.")
        self._encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (sentence for sentence in corpus),
            target_vocab_size=vocab_size,
            reserved_tokens=[
                preprocessing.OUT_OF_SAMPLE_TOKEN,
                preprocessing.START_OF_SAMPLE_TOKEN,
                preprocessing.END_OF_SAMPLE_TOKEN,
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
        print(sequences)
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

    def vocabulary(self) -> List[str]:
        """Return all the word tokens in the vocabulary."""
        return self._encoder._subwords


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
    if cache_file is None:
        return clazz(vocab_size, text)

    try:
        return clazz.load_from_file(cache_file)
    except FileNotFoundError:
        encoder = clazz(vocab_size, text)
        encoder.save_to_file(cache_file)

        return encoder
