import unittest
import tempfile
from src import opennmt_preprocessing
import os
import shutil


class OnmtPreprocessTest(unittest.TestCase):
    def test_generate_onmt_vocabulary_file(self):
        # output = dataloader.read_file(ANY_TEXT_FILE)
        with tempfile.NamedTemporaryFile() as f:
            outputfile = f.name
        opennmt_preprocessing.build_vocabulary(
            "data/splitted_data/train/train_token10000.en", outputfile, 4000
        )
        with open(outputfile) as of:
            vocab = of.read().split("\n")
        # self.assertTrue(len(output) >= 1)
        # As specified in the OpenNMT Doc, those 3 words must be there:
        self.assertTrue(vocab[0] == "<blank>")
        self.assertTrue(vocab[1] == "<s>")
        self.assertTrue(vocab[2] == "</s>")
        self.assertTrue(vocab[-1] == "")  # Empty line at the end
        self.assertEqual(len(vocab), 4004)

    def test_prepare_bpe_models(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            opennmt_preprocessing.prepare_bpe_models(
                "data/splitted_data/train/train_token10000.en",
                "data/splitted_data/train/train_token10000.fr",
                model_dir=tmpdirname,
            )
            src_model, dst_model = opennmt_preprocessing.get_bpe_model_files(
                model_dir=tmpdirname
            )
            src_vocab, dst_vocab = opennmt_preprocessing.get_bpe_vocab_files(
                model_dir=tmpdirname
            )
            self.assertTrue(os.path.exists(src_model))
            self.assertTrue(os.path.exists(dst_model))
            self.assertTrue(os.path.exists(src_vocab))
            self.assertTrue(os.path.exists(dst_vocab))

    def test_prepare_bpe_models_combined(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            opennmt_preprocessing.prepare_bpe_models(
                "data/splitted_data/train/train_token10000.en",
                "data/splitted_data/train/train_token10000.fr",
                model_dir=tmpdirname,
                combined=True,
            )
            src_model, dst_model = opennmt_preprocessing.get_bpe_model_files(
                combined=True, model_dir=tmpdirname,
            )
            src_vocab, dst_vocab = opennmt_preprocessing.get_bpe_vocab_files(
                combined=True, model_dir=tmpdirname
            )
            self.assertTrue(src_model == dst_model)
            self.assertTrue(os.path.exists(dst_model))
            self.assertTrue(src_vocab == dst_vocab)
            self.assertTrue(os.path.exists(dst_vocab))

    def _prepare_bpe_models(self, combined=False):
        tempdir = tempfile.mkdtemp()
        opennmt_preprocessing.prepare_bpe_models(
            "data/splitted_data/train/train_token10000.en",
            "data/splitted_data/train/train_token10000.fr",
            model_dir=tempdir,
            combined=combined,
        )
        src_model, dst_model = opennmt_preprocessing.get_bpe_model_files(
            combined=combined, model_dir=tempdir,
        )
        src_vocab, dst_vocab = opennmt_preprocessing.get_bpe_vocab_files(
            combined=combined, model_dir=tempdir
        )
        return tempdir

    def test_prepare_bpe_file_single(self):
        checkpoint = self._prepare_bpe_models()
        encoded_src, encoded_tgt = opennmt_preprocessing.prepare_bpe_files(
            "data/splitted_data/train/train_token10000.en", None, model_dir=checkpoint
        )
        self.assertTrue(os.path.exists(encoded_src))
        self.assertIsNone(encoded_tgt)
        with open(encoded_src) as of:
            lines = of.read().split("\n")
        self.assertEqual(
            len(lines), 10001
        )  # Split always creates one extra empty line.
        self.assertEqual(
            lines[4], "▁thank ▁you ▁mr ▁president ▁mr ▁v ond ra ▁and ▁mr ▁barroso"
        )

    def test_prepare_bpe_file_both(self):
        checkpoint = self._prepare_bpe_models()
        encoded_src, encoded_tgt = opennmt_preprocessing.prepare_bpe_files(
            "data/splitted_data/train/train_token10000.en",
            "data/splitted_data/train/train_token10000.fr",
            model_dir=checkpoint,
        )
        self.assertTrue(os.path.exists(encoded_src))
        self.assertTrue(os.path.exists(encoded_tgt))
        with open(encoded_src) as of:
            lines = of.read().split("\n")
        self.assertEqual(
            len(lines), 10001
        )  # Split always creates one extra empty line.
        self.assertEqual(
            lines[4], "▁thank ▁you ▁mr ▁president ▁mr ▁v ond ra ▁and ▁mr ▁barroso"
        )
        with open(encoded_tgt) as of:
            lines = of.read().split("\n")
        self.assertEqual(
            len(lines), 10001
        )  # Split always creates one extra empty line.
        self.assertEqual(
            lines[4],
            "▁Merci ▁Monsieur ▁le ▁Président ▁, ▁Monsieur ▁V ond ra ▁et ▁Monsieur ▁Barroso ▁.",
        )

    def test_prepare_bpe_file_both_combined(self):
        checkpoint = self._prepare_bpe_models(combined=True)
        encoded_src, encoded_tgt = opennmt_preprocessing.prepare_bpe_files(
            "data/splitted_data/train/train_token10000.en",
            "data/splitted_data/train/train_token10000.fr",
            model_dir=checkpoint,
            combined=True,
        )
        self.assertTrue(os.path.exists(encoded_src))
        self.assertTrue(os.path.exists(encoded_tgt))
        with open(encoded_src) as of:
            lines = of.read().split("\n")
        self.assertEqual(
            len(lines), 10001
        )  # Split always creates one extra empty line.
        self.assertEqual(
            lines[4], "▁thank ▁you ▁mr ▁president ▁mr ▁v ond ra ▁and ▁mr ▁bar roso"
        )
        with open(encoded_tgt) as of:
            lines = of.read().split("\n")
        self.assertEqual(
            len(lines), 10001
        )  # Split always creates one extra empty line.
        self.assertEqual(
            lines[4],
            "▁Merci ▁Monsieur ▁le ▁Président ▁, ▁Monsieur ▁V ond ra ▁et ▁Monsieur ▁Bar roso ▁.",
        )

    def test_bpe_decode(self):
        combined = False
        checkpoint = self._prepare_bpe_models(combined=combined)
        encoded_src, encoded_tgt = opennmt_preprocessing.prepare_bpe_files(
            "data/splitted_data/train/train_token10000.en",
            "data/splitted_data/train/train_token10000.fr",
            model_dir=checkpoint,
            combined=combined,
        )
        decoded_file = opennmt_preprocessing.decode_bpe_file(
            encoded_tgt, model_dir=checkpoint
        )
        with open("data/splitted_data/train/train_token10000.fr") as f:
            original_lines = f.read().split("\n")
        with open(decoded_file) as f:
            decoded_lines = f.read().split("\n")
        differences = 0
        for ol, dl in zip(original_lines, decoded_lines):
            if ol != dl:
                differences += 1
        self.assertEqual(
            differences, 15
        )  # 15 small errors (mostly punctuation, weird utf-8 sequences) on 10 000 samples. Ok :)

    def test_bpe_decode_combined(self):
        combined = True
        checkpoint = self._prepare_bpe_models(combined=combined)
        encoded_src, encoded_tgt = opennmt_preprocessing.prepare_bpe_files(
            "data/splitted_data/train/train_token10000.en",
            "data/splitted_data/train/train_token10000.fr",
            model_dir=checkpoint,
            combined=combined,
        )
        decoded_file = opennmt_preprocessing.decode_bpe_file(
            encoded_tgt, model_dir=checkpoint, combined=combined
        )
        with open("data/splitted_data/train/train_token10000.fr") as f:
            original_lines = f.read().split("\n")
        with open(decoded_file) as f:
            decoded_lines = f.read().split("\n")
        differences = 0
        for ol, dl in zip(original_lines, decoded_lines):
            if ol != dl:
                differences += 1
        self.assertEqual(
            differences, 15
        )  # 15 small errors (mostly punctuation, weird utf-8 sequences) on 10 000 samples. Ok :)

    def test_shuffle_file(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            fr = f"{tmpdirname}/fr"
            en = f"{tmpdirname}/en"
            orig_fr = "data/splitted_data/train/train_token10000.fr"
            orig_en = "data/splitted_data/train/train_token10000.en"
            shutil.copy(orig_fr, fr)
            shutil.copy(orig_en, en)
            opennmt_preprocessing.shuffle_file(fr)
            opennmt_preprocessing.shuffle_file(en)
            with open(fr) as f:
                fr_lines_1234a = f.read().split("\n")
            with open(en) as f:
                en_lines_1234a = f.read().split("\n")

            shutil.copy(orig_fr, fr)
            shutil.copy(orig_en, en)
            opennmt_preprocessing.shuffle_file(fr)
            opennmt_preprocessing.shuffle_file(en)
            with open(fr) as f:
                fr_lines_1234b = f.read().split("\n")
            with open(en) as f:
                en_lines_1234b = f.read().split("\n")
            shutil.copy(orig_fr, fr)
            shutil.copy(orig_en, en)
            opennmt_preprocessing.shuffle_file(fr, seed=4567)
            opennmt_preprocessing.shuffle_file(en, seed=4567)
            with open(fr) as f:
                fr_lines_5678 = f.read().split("\n")
            with open(en) as f:
                en_lines_5678 = f.read().split("\n")

            # Determinitic shuffle
            self.assertEqual(fr_lines_1234a[0], fr_lines_1234b[0])
            self.assertEqual(en_lines_1234a[0], en_lines_1234b[0])
            # Still as shuffle
            self.assertNotEqual(fr_lines_1234a[0], fr_lines_5678[0])
            # All lines still there.
            self.assertEqual(len(en_lines_5678), 10001)

    def test_concat_file(self):
        with tempfile.NamedTemporaryFile() as f:
            outputfile_all = f.name
        with tempfile.NamedTemporaryFile() as f:
            outputfile_some = f.name
        opennmt_preprocessing.concat_files(
            "data/splitted_data/train/train_token10000.fr",
            "data/splitted_data/train/train_token10000.en",
            outputfile_all,
        )
        opennmt_preprocessing.concat_files(
            "data/splitted_data/train/train_token10000.fr",
            "data/splitted_data/train/train_token10000.en",
            outputfile_some,
            lines1=1000,
            lines2=100,
        )
        with open(outputfile_all) as of:
            all_lines = of.read().split("\n")
        self.assertEqual(len(all_lines), 20001)

        with open(outputfile_some) as of:
            some_lines = of.read().split("\n")
        self.assertEqual(len(some_lines), 1101)
