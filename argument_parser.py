import argparse

from src import logging


def parse_args():
    """Parse the user's arguments.

    The default arguments are to be used in order to reproduce
    the original experiments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", help="Number of epoch to default_training", default=25, type=int
    )
    parser.add_argument(
        "--task",
        help=f"Task to execute, possible ones are:\n",
        default="default-training",
        type=str,
    )
    parser.add_argument(
        "--seed", help="Seed for the experiment", default=1234, type=int
    )
    parser.add_argument(
        "--debug", help="Enable debug logging.", action="store_true",
    )
    parser.add_argument(
        "--std", help="Also log into std.", action="store_true",
    )
    parser.add_argument(
        "--no_cache", help="Disable caching for text encoders.", action="store_true",
    )
    parser.add_argument(
        "--hyperparameters",
        help="Path to the hyperparameters json config file",
        type=str,
        required=True,
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
    parser.add_argument(
        "--lr",
        help="The learning rate, if it's not a float, a learning rate scheduler will be used.",
        default=0.001,
    )
    parser.add_argument(
        "--text_encoder", help="Text Encoder type", default="subword", type=str
    )
    parser.add_argument(
        "--model",
        help=f"Name of the model to run, available models are:\n",
        type=str,
        required=True,
    )
    parser.add_argument("--batch_size", help="Batch size", default=16, type=int)
    parser.add_argument(
        "--max_seq_length", help="Max sequence length", default=None, type=int
    )
    parser.add_argument(
        "--vocab_size", help="Size of the vocabulary", default=30000, type=int
    )
    parser.add_argument(
        "--src_train",
        help="Source training aligned file for aligned training schedules, such as default training.",
        default="data/splitted_data/sorted_train_token.en",
    )
    parser.add_argument(
        "--target_train",
        help="Target training aligned file for aligned training schedules, such as default training.",
        default="data/splitted_data/sorted_nopunctuation_lowercase_train_token.fr",
    )
    parser.add_argument(
        "--src_valid",
        help="Target validation aligned file for aligned training schedules, such as default training.",
        default="data/splitted_data/sorted_val_token.en",
    )
    parser.add_argument(
        "--target_valid",
        help="Source training aligned file.",
        default="data/splitted_data/sorted_nopunctuation_lowercase_val_token.fr",
    )
    parser.add_argument(
        "--pretrained",
        help="Path to the unaligned dataset used for pretraining",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    logger = logging.initialize(experiment=args.model, debug=args.debug, std=args.std)

    return args, logger


args, logger = parse_args()
