import argparse
import pandas as pd
import sentencepiece as spm
from os import path
import os
import random
import tempfile


def build_vocabulary(input_file: str, output_file: str, size: int):
    """Build a vocabulary file compatible with OpenNMT

    Arguments:
        input_file {str} -- Tokenized file with sentences.
        output_file {str} -- Output vocabulary file name.
        size {int} -- Number of words to keep in the vocabulary.
    """
    print(f"Reading {input_file}")
    with open(input_file) as f:
        lines = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    lines = [x.strip() for x in lines]
    tokens = {}
    # Count occcurences of each token.
    for line in lines:
        for token in line.split(" "):
            if token in tokens:
                tokens[token] += 1
            else:
                tokens[token] = 1

    df = pd.DataFrame.from_dict(tokens, orient="index")
    # Build of vocabulary of the "size" most used words in the dataset.
    # Needs to contain those three special word for OpenNMT
    # https://opennmt.net/OpenNMT-tf/vocabulary.html
    if int(size) > len(df):
        size = None
    vocabulary = ["<blank>", "<s>", "</s>"] + df.sort_values(by=0, ascending=False)[0][
        0:size
    ].index.to_list()
    with open(output_file, "w") as of:
        for word in vocabulary:
            of.write(word + "\n")


def prepare_bpe_models(
    source_file: str,
    target_file: str,
    prefix="default",
    combined=False,
    vocab_size=4000,
):
    """Prepare Byte-Pair encoding models using SentencePiece.

    Arguments:
        source_file {str} -- Source text file.
        target_file {str} -- Target text file.

    Keyword Arguments:
        prefix {str} -- prefix for the models. (default: {"default"})
        combined {bool} -- used combined vocabulary. (default: {False})
        vocab_size {int} -- vocabulary size (default: {4000})

    """

    src_prefix, tgt_prefix = get_bpe_prefix(combined=combined)
    if not combined:
        # prepare source language BPE
        spm.SentencePieceTrainer.Train(
            f"--input={source_file} --model_prefix={src_prefix} --vocab_size={vocab_size} --model_type=bpe"
        )
        # prepare target language BPE
        spm.SentencePieceTrainer.Train(
            f"--input={target_file} --model_prefix={tgt_prefix} --vocab_size={vocab_size} --model_type=bpe"
        )
    else:
        # Concatenate both source files and target files
        TMPFILE = "concat.tmp"
        concat_files(source_file, target_file, TMPFILE)
        # prepare combined language BPE
        spm.SentencePieceTrainer.Train(
            f"--input={TMPFILE} --model_prefix={src_prefix} --vocab_size={vocab_size} --model_type=bpe"
        )


def concat_files(
    filename1: str, filename2: str, output_file_name: str, lines1=None, lines2=None
):
    """Concatenate the first "n" lines of two files.

    Arguments:
        filename1 {str} -- First filename.
        filename2 {str} -- Second filename.
        output_file_name {str} -- Output filename.

    Keyword Arguments:
        lines1 {[int]} -- Number of lines to keep. If set to None, everythin is kept. (default: {None})
        lines2 {[int]} -- Number of lines to keep. If set to None, everythin is kept. (default: {None})

    """
    # If lines1 or lines2 is none, it will keep all lines. Else, it will keep only the first lines.
    with open(output_file_name, "w") as outfile:
        with open(filename1) as infile:
            outfile.write("\n".join(infile.read().split("\n")[0:lines1]))
        with open(filename2) as infile:
            outfile.write("\n".join(infile.read().split("\n")[0:lines2]))


def shuffle_file(filename: str, inplace=True, seed=1234):
    """Shuffles a file in place. Will erase source file. Only works on temporary files.

    Arguments:
        filename {[str]} -- Temporary file name.

    Keyword Arguments:
        inplace {bool} -- Shuffle should be done in place. (default: {True})
        seed {int} -- Random seed for the shuflle. (default: {1234})

    Raises:
        ValueError: When requesting to shuffle a normal file, or not in place.
    """
    if not inplace:
        raise ValueError("This will overwrite the file.")
    if ".tmp" not in filename:
        raise ValueError("Use this fonction only on temporary files. It works inplace!")
    random.seed(seed)
    with open(filename) as infile:
        lines = infile.read().split("\n")
    random.shuffle(lines)
    with open(filename, "w") as outfile:
        outfile.write("\n".join(lines))


def get_bpe_prefix(
    prefix="default", combined=False, model_dir="checkpoint"
) -> (str, str):
    """Get the filename prefix for the BPE .vocab and .model files.

    Keyword Arguments:
        prefix {str} -- arbitrary additional prefix. (default: {"default"})
        combined {bool} -- use combined vocabulary (default: {False})
        model_dir {str} -- use specific checkpoint directory. (default: {"checkpoint"})

    Returns:
        (str,str)-- source and target language prefix.
    """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not combined:
        return (f"{model_dir}/{prefix}_src_bpe", f"{model_dir}/{prefix}_tgt_bpe")
    else:
        return (
            f"{model_dir}/{prefix}_combined_bpe",
            f"{model_dir}/{prefix}_combined_bpe",
        )


def get_vocab_file_names(model_dir="checkpoint") -> (str, str):
    """Get the default path for the vocabulary files.

    Vocubulary files are not to be confused with the BPE
    vocabulary files. Those files are used py the OpenNMT
    inputer module, while the BPE vocabulary files are used
    by SentencePiece.

    Keyword Arguments:
        model_dir {str} -- folder where to put the checkpoint.
                           (default: {"checkpoint"})

    Returns:
        (str, str)-- Source, targget vocabulary file names.
    """
    src_vocab = f"{model_dir}/src_vocab.txt"
    tgt_vocab = f"{model_dir}/tgt_vocab.txt"
    return src_vocab, tgt_vocab


def get_bpe_vocab_files(combined=False, prefix="default") -> (str, str):
    """Get the path to the BPE vocabulary files (.vocab)

    Vocubulary files are not to be confused with the BPE
    vocabulary files. Those files are used py the OpenNMT
    inputer module, while the BPE vocabulary files are used
    by SentencePiece.

    Keyword Arguments:
        combined {bool} -- use combined vocabulary (default: {False})
        prefix {str} -- arbitrary additional prefix. (default: {"default"})

    Returns:
        (str,str)-- Path to source and target BPE vocabulary files.
    """
    src_bpe_prefix, tgt_bpe_prefix = get_bpe_prefix(prefix=prefix, combined=combined)
    return (src_bpe_prefix + ".vocab", tgt_bpe_prefix + ".vocab")


def get_bpe_model_files(combined=False, prefix="default"):
    """Get the path to the BPE model files (.model)

    Keyword Arguments:
        combined {bool} -- use combined vocabulary (default: {False})
        prefix {str} -- arbitrary additional prefix. (default: {"default"})

    Returns:
        (str,str)-- Path to source and target BPE model files.
    """
    src_bpe_prefix, tgt_bpe_prefix = get_bpe_prefix(prefix=prefix, combined=combined)
    return (src_bpe_prefix + ".model", tgt_bpe_prefix + ".model")


def prepare_bpe_files(
    source_file: str, target_file: str, combined=False, prefix="default"
) -> (str, str):
    """Encode files using BPE models.

    Arguments:
        source_file {str} -- Path to source language file.
        target_file {str} -- Path to target language file.

    Keyword Arguments:
        combined {bool} -- use combined vocabulary (default: {False})
        prefix {str} -- arbitrary additional prefix. (default: {"default"})

    Returns:
        (str, str) -- path to encoded source and encoded target files.
    """
    src_model, tgt_model = get_bpe_model_files(combined=combined, prefix=prefix)
    if not path.exists(src_model):
        print(
            f"Unable to find BPE model file {src_model}. Have you forgot to copy it along the checkpoint?"
        )
    if not path.exists(tgt_model) and target_file is not None:
        print(
            f"Unable to find BPE model file {tgt_model}. Have you forgot to copy it along the checkpoint?"
        )

    sp_source = spm.SentencePieceProcessor()
    sp_source.Load(src_model)

    # Will generate .bpe files.
    # Source:
    with tempfile.NamedTemporaryFile() as tf:
        bpe_source_file = tf.name

    with open(source_file) as f:
        source_lines = f.readlines()
        source_lines = [x.strip() for x in source_lines]
    with open(bpe_source_file, "w") as of:
        for line in source_lines:
            of.write(" ".join(sp_source.EncodeAsPieces(line)) + "\n")
    # source_file += ".bpe"
    # Target:
    with tempfile.NamedTemporaryFile() as tf:
        bpe_target_file = tf.name

    if target_file is not None:
        sp_target = spm.SentencePieceProcessor()
        sp_target.Load(tgt_model)
        with open(target_file) as f:
            target_lines = f.readlines()
            target_lines = [x.strip() for x in target_lines]
        with open(bpe_target_file, "w") as of:
            for line in target_lines:
                of.write(" ".join(sp_target.EncodeAsPieces(line)) + "\n")
    return bpe_source_file, bpe_target_file


def decode_bpe_file(
    filename: str, target=True, combined=False, prefix="default"
) -> str:
    """Decode a BPE file into normal text.

    Arguments:
        filename {str} -- BPE encoded file name.

    Keyword Arguments:
        target {bool} -- Use target BPE model. (Else will use source model.) (default: {True})
        combined {bool} -- use combined vocabulary (default: {False})
        prefix {str} -- arbitrary additional prefix. (default: {"default"})

    Returns:
        str -- Decoded file name.
    """

    src_model, tgt_model = get_bpe_model_files(combined=combined, prefix=prefix)

    model = tgt_model
    if not target:
        model = src_model

    sp = spm.SentencePieceProcessor()
    sp.Load(model)

    with open(filename) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    output_file_name = f"{filename}.decoded"
    with open(output_file_name, "w") as of:
        for line in lines:
            # Small patch for stray _ : .replace("▁", " ") (+0.3 BLEU)
            of.write(sp.DecodePieces(line.split(" ")).replace("▁", " ") + "\n")
    return output_file_name


def prepare_bpe_en_fr_test():
    prepare_bpe_models(
        "data/splitted_data/train/train_token10000.en",
        "data/splitted_data/train/train_token10000.fr",
    )
    prepare_bpe_files(
        "data/splitted_data/train/train_token10000.en",
        "data/splitted_data/train/train_token10000.fr",
    )
    prepare_bpe_files(
        "data/splitted_data/valid/val_token10000.en",
        "data/splitted_data/valid/val_token10000.fr",
    )
    prepare_bpe_files(
        "data/splitted_data/test/test_token10000.en",
        "data/splitted_data/test/test_token10000.fr",
    )


def decode_bpe_file_test():
    output_file = decode_bpe_file("data/splitted_data/train/train_token10000.fr.bpe")


def prepare_bpe_combined_test():
    prepare_bpe_models(
        "data/splitted_data/train/train_token10000.en",
        "data/splitted_data/train/train_token10000.fr",
        combined=True,
    )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("output", help="Output Vocabulary File")
    parser.add_argument(
        "--src",
        required=True,
        help="Path to the text file from which to extract the vocabulary",
    )
    parser.add_argument("--tgt", help="Path to the target file.")
    args = parser.parse_args()
    build_vocabulary(args.src, args.output)


def build_vocabulary_test():
    build_vocabulary(
        "data/splitted_data/train/train_token10000.en", "vocab.txt", size=16000
    )


if __name__ == "__main__":
    # main()
    prepare_bpe_en_fr_test()
    prepare_bpe_combined_test()
    decode_bpe_file_test()
