import unittest
from src import utils


class UtilsTest(unittest.TestCase):
    def test_write_filepair(self):
        sentence_pairs = [
            ("red\n", "rouge\n"),
            ("blue\n", "bleu\n"),
            ("green\n", "vert\n"),
        ]
        utils.write_set("/tmp/test.en", "/tmp/test.fr", sentence_pairs)
        f1 = open("/tmp/test.en")
        f2 = open("/tmp/test.fr")
        for sentence1, sentence2 in sentence_pairs:
            line1 = f1.readline()
            line2 = f2.readline()
            self.assertEquals(sentence1, line1)
            self.assertEquals(sentence2, line2)

    def test_split_on_real_data(self):
        (
            train_lang1,
            train_lang2,
            valid_lang1,
            valid_lang2,
            test_lang1,
            test_lang2,
        ) = utils.split_joint_data("data/train.lang1", "data/train.lang2", "/tmp")

        with open(train_lang1) as f:
            data = f.read().splitlines()
            self.assertEquals(
                data[-1],
                "but germany did not accumulate foreign reserves the way that china did",
            )
            self.assertEquals(len(data), 10000)

        with open(train_lang2) as f:
            data = f.read().splitlines()
            self.assertEquals(
                data[-1],
                "Mais l’ Allemagne n’ a pas accumulé des réserves en devises étrangères comme la Chine .",
            )
            self.assertEquals(len(data), 10000)

        with open(valid_lang1) as f:
            data = f.read().splitlines()
            self.assertEquals(
                data[-1],
                "the reshuffle last november that brought kanaan and others into the cabinet was seen to reflect assad ’s preferences",
            )
            self.assertEquals(len(data), 500)

        with open(valid_lang2) as f:
            data = f.read().splitlines()
            self.assertEquals(
                data[-1],
                "Le remaniement de novembre dernier qui mit au pouvoir Kanaan et d’ autres fut considéré comme la réflexion des préférences de M. Assad .",
            )
            self.assertEquals(len(data), 500)

        with open(test_lang1) as f:
            data = f.read().splitlines()
            self.assertEquals(
                data[-1],
                "i would like to stress the quality of the amendments tabled by the rapporteur and the need to make up for lost time",
            )
            self.assertEquals(len(data), 500)

        with open(test_lang2) as f:
            data = f.read().splitlines()
            self.assertEquals(
                data[-1],
                "J' insiste sur la qualité des amendements soumis par le rapporteur et sur la nécessité de gagner du temps .",
            )
            self.assertEquals(len(data), 500)
