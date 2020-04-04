import argparse
import os

import matplotlib.pyplot as plt

from src.training.base import History


def parse_args():
    """Parse the user's arguments.

    The default arguments are to be used in order to reproduce
    the original experiments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history_path",
        help=f"Path of the history output in results folder",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_path",
        help=f"Output folder path for all the vizualisation",
        type=str,
        required=True,
    )

    return parser.parse_args()


def plot_loss(training_loss, validation_loss, output_path):
    plt.plot(training_loss, label="Training loss")
    plt.plot(validation_loss, label="validation loss")
    plt.title("Training and Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(output_path + "/train_valid_loss.png")
    plt.close()


def plot_bleu(training_bleu, validation_bleu, output_path):
    plt.plot(training_bleu, label="Training bleu")
    plt.plot(validation_bleu, label="validation bleu")
    plt.title("Training and Validation bleu")
    plt.xlabel("Epochs")
    plt.ylabel("Bleu")
    plt.legend()
    plt.grid()
    plt.savefig(output_path + "/train_valid_bleu.png")
    plt.close()


def main():
    args = parse_args()

    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    history = History()
    history = history.load(args.history_path)

    training_loss = history.logs["train_loss"]
    validation_loss = history.logs["valid_loss"]

    training_bleu = history.logs["train_bleu"]
    validation_bleu = history.logs["valid_bleu"]

    plot_loss(training_loss, validation_loss, output_path)
    plot_bleu(training_bleu, validation_bleu, output_path)


if __name__ == "__main__":
    main()
