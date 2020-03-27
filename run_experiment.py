import argparse
import random

import tensorflow as tf

from src import dataloader
from src.model import lstm, transformer
from src.training import base
from src.training.back_translation import BackTranslationTraining
from src.training.base import BasicMachineTranslationTraining


def create_lstm(args, input_vocab_size, target_vocab_size):
    model = lstm.Lstm(input_vocab_size, target_vocab_size)
    return model


def create_transformer(args, input_vocab_size, target_vocab_size):
    model = transformer.Transformer(
        num_layers=2,
        num_heads=8,
        dff=512,
        d_model=512,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        pe_input=input_vocab_size,
        pe_target=target_vocab_size,
        rate=0.1,
    )
    return model


MODELS = {lstm.NAME: create_lstm, transformer.NAME: create_transformer}


def parse_args():
    """Parse the user's arguments.

    The default arguments are to be used in order to reproduce
    the original experiments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", help="Number of epoch to basic_training", default=25, type=int
    )
    parser.add_argument(
        "--test",
        help="Test a trained model on the test set. The value must be the model's checkpoint",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--basic_training", help="Train a model.", action="store_true",
    )
    parser.add_argument(
        "--back_translation_training",
        help="Train a model with back translation.",
        action="store_true",
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
        help=f"Name of the model to run, available models are:\n{list(MODELS.keys())}",
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

    if args.basic_training and args.back_translation_training:
        raise ValueError(
            "Both basic training and back translation training were chosen, only one can be use at the same time."
        )

    if not args.random_seed:
        random.seed(args.seed)
        tf.random.set_seed(args.seed)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    if args.basic_training:
        basic_training(args, loss_fn)

    if args.back_translation_training:
        back_translation_training(args, loss_fn)

    if args.test is not None:
        test(args, loss_fn)


def basic_training(args, loss_fn):
    """Train the model."""
    optim = tf.keras.optimizers.Adam(args.lr)
    train_dl = dataloader.AlignedDataloader(
        file_name_input="data/splitted_data/sorted_train_token.en",
        file_name_target="data/splitted_data/sorted_nopunctuation_lowercase_train_token.fr",
        vocab_size=args.vocab_size,
    )
    valid_dl = dataloader.AlignedDataloader(
        file_name_input="data/splitted_data/sorted_val_token.en",
        file_name_target="data/splitted_data/sorted_nopunctuation_lowercase_val_token.fr",
        vocab_size=args.vocab_size,
        encoder_input=train_dl.encoder_input,
        encoder_target=train_dl.encoder_target,
    )
    model = MODELS[args.model](
        args, train_dl.encoder_input.vocab_size, train_dl.encoder_target.vocab_size
    )
    training = BasicMachineTranslationTraining(model, train_dl, valid_dl)
    training.run(
        loss_fn, optim, batch_size=args.batch_size, num_epoch=args.epochs,
    )


def back_translation_training(args, loss_fn):
    """Train the model with back translation."""
    optim = tf.keras.optimizers.Adam(args.lr)
    print("Creating training unaligned dataloader ...")
    train_dl = dataloader.UnalignedDataloader("data/unaligned.en", args.vocab_size)
    print(f"English vocab size: {train_dl.encoder.vocab_size}")

    print("Creating reversed training unaligned dataloader ...")
    train_dl_reverse = dataloader.UnalignedDataloader(
        "data/unaligned.fr", args.vocab_size
    )
    print(f"French vocab size: {train_dl_reverse.encoder.vocab_size}")

    print("Creating training aligned dataloader ...")
    aligned_train_dl = dataloader.AlignedDataloader(
        file_name_input="data/splitted_data/sorted_train_token.en",
        file_name_target="data/splitted_data/sorted_train_token.fr",
        vocab_size=args.vocab_size,
        encoder_input=train_dl.encoder,
        encoder_target=train_dl_reverse.encoder,
    )

    print("Creating reversed training aligned dataloader ...")
    aligned_train_dl_reverse = dataloader.AlignedDataloader(
        file_name_input="data/splitted_data/sorted_train_token.fr",
        file_name_target="data/splitted_data/sorted_train_token.en",
        vocab_size=args.vocab_size,
        encoder_input=aligned_train_dl.encoder_target,
        encoder_target=aligned_train_dl.encoder_input,
    )

    print("Creating valid aligned dataloader ...")
    aligned_valid_dl = dataloader.AlignedDataloader(
        file_name_input="data/splitted_data/sorted_val_token.en",
        file_name_target="data/splitted_data/sorted_val_token.fr",
        vocab_size=args.vocab_size,
        encoder_input=aligned_train_dl.encoder_input,
        encoder_target=aligned_train_dl.encoder_target,
    )

    print("Creating reversed valid aligned dataloader ...")
    aligned_valid_dl_reverse = dataloader.AlignedDataloader(
        file_name_input="data/splitted_data/sorted_val_token.fr",
        file_name_target="data/splitted_data/sorted_val_token.en",
        vocab_size=args.vocab_size,
        encoder_input=aligned_train_dl_reverse.encoder_input,
        encoder_target=aligned_train_dl_reverse.encoder_target,
    )

    model = MODELS[args.model](
        args,
        aligned_train_dl.encoder_input.vocab_size,
        aligned_train_dl.encoder_target.vocab_size,
    )

    model_reverse = MODELS[args.model](
        args,
        aligned_train_dl_reverse.encoder_input.vocab_size,
        aligned_train_dl_reverse.encoder_target.vocab_size,
    )

    training = BackTranslationTraining(
        model,
        model_reverse,
        train_dl,
        train_dl_reverse,
        aligned_train_dl,
        aligned_train_dl_reverse,
        aligned_valid_dl,
        aligned_valid_dl_reverse,
    )

    training.run(
        loss_fn,
        optim,
        batch_size=args.batch_size,
        num_epoch=args.epochs,
        checkpoint=args.checkpoint,
    )


def test(args, loss_fn):
    """Test the model."""
    # Used to load the train text encoders.
    train_dl = dataloader.AlignedDataloader(
        file_name_input="data/splitted_data/sorted_train_token.en",
        file_name_target="data/splitted_data/sorted_train_token.fr",
        vocab_size=args.vocab_size,
    )
    test_dl = dataloader.AlignedDataloader(
        file_name_input="data/splitted_data/sorted_test_token.en",
        file_name_target="data/splitted_data/sorted_test_token.fr",
        vocab_size=args.vocab_size,
        encoder_input=train_dl.encoder_input,
        encoder_target=train_dl.encoder_target,
    )
    model = MODELS[args.model](
        args, train_dl.encoder_input.vocab_size, train_dl.encoder_target.vocab_size
    )
    base.test(model, loss_fn, test_dl, args.batch_size, args.test)


if __name__ == "__main__":
    main()
