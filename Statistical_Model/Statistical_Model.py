import collections
import math

import numpy as np
from nltk import ngrams

from utility import read_file_np_lower, write_text

TRAIN_EN_PATH = "data/splitted_data/train/train_token10000.en"
TRAIN_FR_PATH = "data/splitted_data/train/train_token10000.fr"
VAL_EN_PATH = "data/splitted_data/valid/val_token10000.en"
VAL_FR_PATH = "data/splitted_data/valid/val_token10000.fr"
TEST_EN_PATH = "data/splitted_data/test/test_token10000.en"
TEST_FR_PATH = "data/splitted_data/test/test_token10000.fr"
UNALIGNED_EN_PATH = "data/unaligned.en"
UNALIGNED_FR_PATH = "data/unaligned.fr"


class Dictionnary_monolingual:
    def __init__(self, sentences):
        self.sentences = sentences
        self.counter = self._get_counter()
        self.len_dict = len(self.counter)
        self.dict_word_number, self.dict_number_word = self._get_word_to_number()

    def _get_counter(self):
        return collections.Counter(
            [word for sentence in self.sentences for word in sentence]
        )

    def _get_word_to_number(self):
        dict_word_number = {}
        dict_number_word = {}
        dict_word_number["<NULL>"] = 0
        dict_number_word[0] = "<NULL>"
        i = 1

        for word in self.counter:
            dict_word_number[word] = i
            dict_number_word[i] = word
            i += 1
        return dict_word_number, dict_number_word

    def get_count_word(self, word):
        return self.counter[word]

    def get_number_of_word(self):
        return self.len_dict

    def get_max_len(self):
        return np.max([len(sentence) for sentence in self.sentences])

    def sequence_to_words(self, sequence):
        # [[0], [3], [1]] to [[<NULL], ['allo'], ['oui']]
        return [self.dict_number_word[seq] for seq in sequence]

    def words_to_sequence(self, words):
        # [[<NULL], ['allo'], ['oui']] to [[0], [3], [1]]
        return [self.dict_word_number[word] for word in words]


class Unaligned_dictionnary:
    def __init__(self, sentences):
        self.sentences = sentences
        self.counter = self._get_counter()
        self.bigram_counter = self._get_bigram_counter()
        self.number_words = sum(self.counter.values())

    def _get_counter(self):
        return collections.Counter(
            [word for sentence in self.sentences for word in sentence]
        )

    def _get_bigram_counter(self):
        bigram_words = []
        for sentence in self.sentences:
            sentence = ["<start>"] + sentence
            bigram = ngrams(sentence, 2)
            for bi in bigram:
                bigram_words.append(" ".join(bi))
        return collections.Counter(bigram_words)

    def get_proba(self, prev, actual, l):
        if prev == "<start>":
            get_p = self.bigram_counter[prev + " " + actual] / len(self.sentences)
            return l * (self.counter[actual] / self.number_words) + (1 - l) * get_p
        if prev not in self.counter:
            get_p = self.bigram_counter[prev + " " + actual] / self.counter[prev]
            return l * (self.counter[actual] / self.number_words)
        return l * (self.counter[actual] / self.number_words) + (1 - l) * get_p


class Proba_wordEN_knowing_wordFR:
    def __init__(self, dictionnary_en, dictionnary_fr):
        self.dictionnary_en = dictionnary_en
        self.dictionnary_fr = dictionnary_fr
        self.correspondance_table = self._get_correspondance_table()

    def _get_correspondance_table(self):
        # initialising 2-d array for each english/french words and NULL
        nb_word_en = self.dictionnary_en.get_number_of_word() + 1
        nb_word_fr = self.dictionnary_fr.get_number_of_word() + 1
        return np.ones((nb_word_en, nb_word_fr)) / nb_word_fr

    def get_prob_wordEN_knowing_wordFR(self, word_en, word_fr):
        return self.correspondance_table[word_en][word_fr]

    def update_proba_table(self, aligned_en, aligned_fr):
        nb_word_en = self.dictionnary_en.get_number_of_word() + 1
        nb_word_fr = self.dictionnary_fr.get_number_of_word() + 1
        new_table = np.zeros((nb_word_en, nb_word_fr))

        for (sentence_en, sentence_fr) in zip(aligned_en, aligned_fr):
            for (word_en, word_fr) in zip(sentence_en, sentence_fr):
                new_table[word_en][word_fr] += 1

        self.correspondance_table = new_table / new_table.sum(axis=1, keepdims=True)


class Corpus_alignment_en:
    def __init__(self, corpus, dictionnary_en):
        self.dictionnary_en = dictionnary_en
        self.corpus = corpus
        self.corpus_seq = self._get_corpus_seq()

    def _get_corpus_seq(self):
        return [
            self.dictionnary_en.words_to_sequence(sentence) for sentence in self.corpus
        ]

    def get_seq_len(self):
        return [len(seq) for seq in self.corpus_seq]


class Corpus_alignment_fr:
    def __init__(self, corpus, dictionnary_fr, translation):
        self.dictionnary_fr = dictionnary_fr
        self.corpus = corpus
        self.corpus_seq = self._get_corpus_seq()
        self.translation = translation

    def _get_corpus_seq(self):
        return [
            self.dictionnary_fr.words_to_sequence(sentence) for sentence in self.corpus
        ]

    def _gen_alignments(self, sentences_en, sentences_fr):
        # TODO: Ajouter <NULL>
        new_sentences_fr = []

        for (sentence_en, sentence_fr) in zip(sentences_en, sentences_fr):
            new_sentences_fr.append(self._get_best_alignment(sentence_en, sentence_fr))

        return new_sentences_fr

    def _get_best_alignment(self, sentence_en, sentence_fr):
        len_en = len(sentence_en)
        new_sentence_fr = np.zeros(len_en)
        completed = np.zeros(len_en)

        i_en = np.arange(len_en)
        j_fr = np.arange(len(sentence_fr))

        while not (np.sum(completed) == len_en):
            max_proba = -math.inf
            best_alignment = {"fr": None, "en": None}

            if len(j_fr) == 0:
                break

            for i in i_en:
                for j in j_fr:
                    score = self.translation.get_prob_wordEN_knowing_wordFR(
                        sentence_en[i], sentence_fr[j]
                    )
                    if score > max_proba:
                        max_proba = score
                        best_alignment["en"] = i
                        best_alignment["fr"] = j

            new_sentence_fr[best_alignment["en"]] = sentence_fr[best_alignment["fr"]]
            i_en = np.delete(i_en, np.where(i_en == best_alignment["en"]))
            j_fr = np.delete(j_fr, np.where(j_fr == best_alignment["fr"]))
            completed[best_alignment["en"]] = 1

        return new_sentence_fr.astype(int)


def translate(translation_table, corpus_en, m=0.95, ls=0.9):
    translate_sentences = []
    for sentence in corpus_en:
        new_sentence = []
        for i in range(len(sentence)):
            word_en = sentence[i]
            if word_en not in train_dict_en.dict_word_number:
                new_sentence.append(word_en)
                continue
            seq_en = train_dict_en.dict_word_number[word_en]
            possible_seq_fr = translation_table[seq_en].argsort()[::-1][:20]
            best_score = -math.inf
            best_word_fr = None

            for seq_fr in possible_seq_fr:
                word_fr = train_dict_fr.dict_number_word[seq_fr]
                if i == 0:
                    pr_seq_fr = unaligned_dict_fr.get_proba("<start>", word_fr, ls)
                else:
                    pr_seq_fr = unaligned_dict_fr.get_proba(
                        new_sentence[i - 1], word_fr, ls
                    )
                score = translation_table[seq_en][seq_fr] + m * pr_seq_fr
                if score > best_score:
                    best_score = score
                    best_word_fr = word_fr

            new_sentence.append(best_word_fr)
        translate_sentences.append(new_sentence)

    return translate_sentences


def remove_null(tr):
    return [[word for word in sentence if not word == "<NULL>"] for sentence in tr]


corpus_en = read_file_np_lower(TRAIN_EN_PATH)
corpus_fr = read_file_np_lower(TRAIN_FR_PATH)

valid_en = read_file_np_lower(VAL_EN_PATH)
valid_fr = read_file_np_lower(VAL_FR_PATH)

test_en = read_file_np_lower(TEST_EN_PATH)
test_fr = read_file_np_lower(TEST_FR_PATH)

unaligned_fr = read_file_np_lower(UNALIGNED_FR_PATH)

train_dict_en = Dictionnary_monolingual(corpus_en)
train_dict_fr = Dictionnary_monolingual(corpus_fr)
unaligned_dict_fr = Unaligned_dictionnary(unaligned_fr)

translation = Proba_wordEN_knowing_wordFR(train_dict_en, train_dict_fr)
train_seq_en = Corpus_alignment_en(corpus_en, train_dict_en)
train_seq_fr = Corpus_alignment_fr(corpus_fr, train_dict_fr, translation)

# Training
for i in range(10):
    print("epoch: ", i + 1)

    new_corpus_fr = train_seq_fr._gen_alignments(
        train_seq_en.corpus_seq, train_seq_fr.corpus_seq
    )
    translation.update_proba_table(train_seq_en.corpus_seq, new_corpus_fr)

# Prediction Train
translation_result_train = translate(translation.correspondance_table, corpus_en)
write_text(remove_null(translation_result_train), "translation_result.fr")

# Prediction Valid
translation_result_valid = translate(translation.correspondance_table, valid_en)
write_text(remove_null(translation_result_valid), "translation_result.fr")

# Prediction Test
translation_result_test = translate(translation.correspondance_table, test_en)
write_text(remove_null(translation_result_test), "translation_result.fr")
