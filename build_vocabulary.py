import argparse
import pandas as pd


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
    vocabulary = ["<blank>", "<s>", "</s>"] + df.sort_values(by=0, ascending=False)[0][
        0:size
    ].index.to_list()
    with open(output_file, "w") as of:
        for word in vocabulary:
            of.write(word + "\n")


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


if __name__ == "__main__":
    # main()
    build_vocabulary(
        "data/splitted_data/train/train_token10000.en", "vocab.txt", size=16000
    )
