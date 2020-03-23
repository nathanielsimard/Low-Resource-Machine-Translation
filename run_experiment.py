import argparse
import random

import tensorflow as tf

from src import dataloader, training
from src.model import lstm


def create_lstm(args, input_vocab_size, target_vocab_size):
    model = lstm.Lstm(input_vocab_size, target_vocab_size)
    return model


MODELS = {lstm.NAME: create_lstm}


def parse_args():
    """Parse the user's arguments.

    The default arguments are to be used in order to reproduce
    the original experiments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", help="Number of epoch to train", default=25, type=int
    )
    parser.add_argument(
        "--test",
        help="Test a trained model on the test set. The value must be the model's checkpoint",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--train", help="Train a model.", action="store_true",
    )
    parser.add_argument(
        "--seed", help="Seed for the experiment", default=1234, type=int
    )
    parser.add_argument(
        "--random_seed",
        help="Will overide the default seed and use a random one",
        action="store_true",
    )
    parser.add_argument(
        "--checkpoint",
        help="The checkpoint to load before training.",
        default=None,
        type=int,
    )
    parser.add_argument("--lr", help="Learning rate", default=0.001, type=float)
    parser.add_argument(
        "--model",
        help=f"Name of the model to train, available models are:\n{list(MODELS.keys())}",
        type=str,
        required=True,
    )
    parser.add_argument("--batch_size", help="Batch size", default=16, type=int)
    parser.add_argument(
        "--vocab_size", help="Size of the vocabulary", default=80000, type=int
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.random_seed:
        random.seed(args.seed)
        tf.random.set_seed(args.seed)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    if args.train:
        train(args, loss_fn)

    if args.test is not None:
        test(args, loss_fn)


def train(args, loss_fn):
    """Train the model."""
    optim = tf.keras.optimizers.Adam(args.lr)
    train_dl = dataloader.Dataloader(
        file_name_input="data/splitted_data/sorted_train_token.en",
        file_name_target="data/splitted_data/sorted_train_token.fr",
        vocab_size=args.vocab_size,
    )
    valid_dl = dataloader.Dataloader(
        file_name_input="data/splitted_data/sorted_val_token.en",
        file_name_target="data/splitted_data/sorted_val_token.fr",
        vocab_size=args.vocab_size,
        encoder_input=train_dl.encoder_input,
        encoder_target=train_dl.encoder_target,
    )
    model = MODELS[args.model](
        args, train_dl.encoder_input.vocab_size, train_dl.encoder_target.vocab_size
    )
    training.run(
        model,
        loss_fn,
        optim,
        train_dataloader=train_dl,
        valid_dataloader=valid_dl,
        batch_size=args.batch_size,
        num_epoch=args.epochs,
    )


def test(args, loss_fn):
    """Test the model."""
    # Used to load the train text encoders.
    train_dl = dataloader.Dataloader(
        file_name_input="data/splitted_data/sorted_train_token.en",
        file_name_target="data/splitted_data/sorted_train_token.fr",
        vocab_size=args.vocab_size,
    )
    test_dl = dataloader.Dataloader(
        file_name_input="data/splitted_data/sorted_test_token.en",
        file_name_target="data/splitted_data/sorted_test_token.fr",
        vocab_size=args.vocab_size,
        encoder_input=train_dl.encoder_input,
        encoder_target=train_dl.encoder_target,
    )
    model = MODELS[args.model](
        args, train_dl.encoder_input.vocab_size, train_dl.encoder_target.vocab_size
    )
    training.test(model, loss_fn, test_dl, args.batch_size, args.test)


if __name__ == "__main__":
    main()
