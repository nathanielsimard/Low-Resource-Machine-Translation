import argparse
import pandas as pd
import sentencepiece as spm
from os import path
import os


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
    source_file, target_file, prefix="default", combined=False, vocab_size=4000
):
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
    # If lines1 or lines2 is none, it will keep all lines. Else, it will keep only the first lines.
    with open(output_file_name, "w") as outfile:
        with open(filename1) as infile:
            outfile.write("\n".join(infile.read().split("\n")[0:lines1]))
        with open(filename2) as infile:
            outfile.write("\n".join(infile.read().split("\n")[0:lines2]))


import random


def shuffle_file(filename, inplace=True, seed=1234):
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


def get_bpe_prefix(prefix="default", combined=False, model_dir="checkpoint"):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not combined:
        return (f"{model_dir}/{prefix}_src_bpe", f"{model_dir}/{prefix}_tgt_bpe")
    else:
        return (
            f"{model_dir}/{prefix}_combined_bpe",
            f"{model_dir}/{prefix}_combined_bpe",
        )


def get_bpe_vocab_files(combined=False, prefix="default"):
    src_bpe_prefix, tgt_bpe_prefix = get_bpe_prefix(prefix=prefix, combined=combined)
    return (src_bpe_prefix + ".vocab", tgt_bpe_prefix + ".vocab")


def get_bpe_model_files(combined=False, prefix="default"):
    src_bpe_prefix, tgt_bpe_prefix = get_bpe_prefix(prefix=prefix, combined=combined)
    return (src_bpe_prefix + ".model", tgt_bpe_prefix + ".model")


def prepare_bpe_files(source_file, target_file, combined=False, prefix="default"):
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
    with open(source_file) as f:
        source_lines = f.readlines()
        source_lines = [x.strip() for x in source_lines]
    with open(f"{source_file}.bpe", "w") as of:
        for line in source_lines:
            of.write(" ".join(sp_source.EncodeAsPieces(line)) + "\n")
    source_file += ".bpe"
    # Target:
    if target_file is not None:
        sp_target = spm.SentencePieceProcessor()
        sp_target.Load(tgt_model)
        with open(target_file) as f:
            target_lines = f.readlines()
            target_lines = [x.strip() for x in target_lines]
        with open(f"{target_file}.bpe", "w") as of:
            for line in target_lines:
                of.write(" ".join(sp_target.EncodeAsPieces(line)) + "\n")
        target_file += ".bpe"
    return source_file, target_file


def decode_bpe_file(filename, target=True, combined=False, prefix="default"):

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
