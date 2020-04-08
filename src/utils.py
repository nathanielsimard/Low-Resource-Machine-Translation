#

import random


def split_joint_data(
    input_lang1: str,
    input_lang2: str,
    output_folder: str,
    valid=500,
    test=500,
    shuffle=True,
    lang1_suffix: str = "en",
    lang2_suffix: str = "fr",
    seed=1234,
):
    """Splits a dataset consisting of two aligned files into a train set, a validation set and a test set.

    Arguments:
        input_lang1 {str} : Path to the file in the first language
        input_lang2 {str} : Path to the file in the second language
        output_folder {str} : Output folder for the generated files.
        valid {int} : Size of the validation set, in samples. (default: {500})
        test {int} : Size of the test set, in samples (default: {500})
        lang1_suffix {str} : Suffix for the file in the first language
        lang2_suffix {str} : Suffix for the file in the second language
        shuffle {bool} : Whether the data should be shuffle or not
        seed {int} : Random seed for the shuffle


    Limitations:
        Since everything is loaded in memory, this code will not work for files more than a 100 megabytes.
    """
    with open(input_lang1) as f:
        lang1 = f.readlines()
    with open(input_lang2) as f:
        lang2 = f.readlines()

    sentence_pairs = list(zip(lang1, lang2))
    if shuffle:
        random.seed(seed)
        random.shuffle(sentence_pairs)

    train_lang1 = f"{output_folder}/train.{lang1_suffix}"
    train_lang2 = f"{output_folder}/train.{lang2_suffix}"
    valid_lang1 = f"{output_folder}/valid.{lang1_suffix}"
    valid_lang2 = f"{output_folder}/valid.{lang2_suffix}"
    test_lang1 = f"{output_folder}/test.{lang1_suffix}"
    test_lang2 = f"{output_folder}/test.{lang2_suffix}"

    # Write test set, valid set and train set.
    write_set(test_lang1, test_lang2, sentence_pairs[0:test])
    write_set(valid_lang1, valid_lang2, sentence_pairs[test : valid + test])
    write_set(train_lang1, train_lang2, sentence_pairs[valid + test :])

    # Return paths to the written file for furter use
    return (train_lang1, train_lang2, valid_lang1, valid_lang2, test_lang1, test_lang2)


def write_set(filename1: str, filename2: str, sentence_pairs: list):
    """Helper for the split_joint_data.

    Arguments:
        filename1 {str} : Output filename in the first languange
        filename2 {str} : Output filename in the seconde language
        sentence_pairs {list} : List of sentence tuples to write.
    """
    file1 = open(filename1, "w")
    file2 = open(filename2, "w")
    lst1, lst2 = zip(*sentence_pairs)
    for sentence1, sentence2 in sentence_pairs:
        file1.write(sentence1)
        file2.write(sentence2)
    file1.close()
    file2.close()
