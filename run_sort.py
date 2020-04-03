import os
import numpy as np

DATA_DIRECTORY = "/home/olivier/Documents/local_git/Low-Resource-Machine-Translation/data/splitted_english_data/"


def main():
    for dire in os.listdir(DATA_DIRECTORY):
        directory = os.listdir(os.path.join(DATA_DIRECTORY, dire))
        i = 0
        while i < 2:
            for file in directory:
                path = os.path.join(os.path.join(DATA_DIRECTORY, dire), file)
                if path.endswith(".txt"):
                    continue
                elif path.endswith(".en"):
                    tokens, indexes = get_order_by_length(path)
                    np.savetxt(
                        os.path.join(DATA_DIRECTORY, dire) + "/en_indexes.txt",
                        indexes,
                        "%s",
                    )
                else:
                    tokens, _ = get_order_by_length(path)

                if os.path.exists(
                    os.path.join(DATA_DIRECTORY, dire) + "/en_indexes.txt"
                ):
                    indexes = np.loadtxt(
                        os.path.join(DATA_DIRECTORY, dire) + "/en_indexes.txt",
                        dtype=int,
                    )
                    new_token = [tokens[indexes[i]] for i in range(len(tokens))]
                    write_text_from_tokens(
                        new_token, os.path.join(DATA_DIRECTORY, "sorted_" + file)
                    )
                    i += 1


def get_order_by_length(path):
    token = read_token_file(path)
    length_token = [len(token[i]) for i in range(len(token))]
    indexes = np.argsort(length_token)
    return token, indexes


def write_text_from_tokens(tokens, output_file):
    with open(output_file, "w+") as out_stream:
        for token in tokens:
            out_stream.write(" ".join(token) + "\n")


def read_token_file(file_name: str):
    out = []
    with open(file_name, "r") as stream:
        for line in stream:
            tokens = line.strip().split()
            out.append(tokens)
    return out


if __name__ == "__main__":
    main()
