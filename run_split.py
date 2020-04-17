import random
import os
import argparse

random.seed(1111)

DATA_OUTPUT_DIRECTORY = "/home/olivier/Documents/local_git/Low-Resource-Machine-Translation/data/splitted_french_data/"

if not os.path.exists(DATA_OUTPUT_DIRECTORY):
    os.makedirs(DATA_OUTPUT_DIRECTORY)

parser = argparse.ArgumentParser()
parser.add_argument("input_file", type=str, help="input text file to split")
parser.add_argument(
    "--training_ratio",
    type=float,
    default=0.8,
    help="what portion to attribute to the training dataset",
)

parser.add_argument(
    "output_prefix", type=str, help="specifiy a prefix for the splitted data"
)


def split(file, split_val1: float, split_val2: float):
    dataset = read_token_file(file)
    random.shuffle(dataset)
    train_index = int(split_val1 * len(dataset))
    valid_index = int(split_val2 * len(dataset))

    train_dataset = dataset[:train_index]
    valid_dataset = dataset[train_index:valid_index]
    test_dataset = dataset[valid_index:]

    return train_dataset, valid_dataset, test_dataset


def read_token_file(file_name: str):
    out = []
    with open(file_name, "r") as stream:
        for line in stream:
            tokens = line.strip().split()
            out.append(tokens)
    return out


def write_text_from_tokens(tokens, output_file):
    with open(output_file, "w+") as out_stream:
        for token in tokens:
            out_stream.write(" ".join(token) + "\n")


def main():
    args = parser.parse_args()
    split_val1 = args.training_ratio
    split_val2 = split_val1 + (1 - split_val1) / 2.0

    train, valid, test = split(args.input_file, split_val1, split_val2)

    output_prefix = args.output_prefix

    write_text_from_tokens(train, DATA_OUTPUT_DIRECTORY + output_prefix + "_train.fr")
    write_text_from_tokens(valid, DATA_OUTPUT_DIRECTORY + output_prefix + "_valid.fr")
    write_text_from_tokens(test, DATA_OUTPUT_DIRECTORY + output_prefix + "_test.fr")


if __name__ == "__main__":
    main()
