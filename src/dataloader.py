import os
from typing import List
import numpy as np


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

        all_eng_words = set()
        for eng in self.corpus_input:
            for word in eng:
                if word not in all_eng_words:
                    all_eng_words.add(word)

        all_french_words = set()
        for fr in self.corpus_target:
            for word in fr:
                if word not in all_french_words:
                    all_french_words.add(word)

        self.input_words = sorted(list(all_eng_words))
        self.target_words = sorted(list(all_french_words))
        self.num_encoder_tokens = len(all_eng_words)
        self.num_decoder_tokens = len(all_french_words)
        max_len_input = _max_len(self.corpus_input)
        max_len_target = _max_len(self.corpus_target)
        self.max_len = max(max_len_input, max_len_target) + 1
        # del all_eng_words, all_french_words

    def create_dataset(self) -> tf.data.Dataset:
        """Create a Tensorflow dataset."""

        def gen():
            # lines.fr = lines.fr.apply(lambda x : 'START_ '+ x + ' _END')# Create vocabulary of words

            max_len_input = _max_len(self.corpus_input)
            max_len_target = _max_len(self.corpus_target)
            max_len = max(max_len_input, max_len_target) + 1

            encoder_input_data = np.zeros((len(self.corpus_input), max_len), dtype='float32')
            decoder_input_data = np.zeros((len(self.corpus_target), max_len), dtype='float32')
            # generate datafor i, (input_text, target_text) in enumerate(zip(lines.eng, lines.fr)):

            for i, (input_text, target_text) in enumerate(zip(self.corpus_input, self.corpus_target)):
                for t, word in enumerate(input_text):
                    encoder_input_data[i, t] = self.input_words.index(word)

                decoder_input_data[i, 0] = len(self.target_words)
                for t, word in enumerate(target_text):
                    # decoder_target_data is ahead of decoder_input_data by one timestep
                    decoder_input_data[i, t + 1] = self.target_words.index(word)

                yield (encoder_input_data[i], decoder_input_data[i])

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
            output.append(tokens.split())
    return output


def _max_len(sentences):
    max_len = 0
    for sentence in sentences:
        if len(sentence) > max_len:
            max_len = len(sentence)
    return max_len


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
