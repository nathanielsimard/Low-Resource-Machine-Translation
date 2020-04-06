import argparse

from src import logging


def parse_args():
    """Parse the user's arguments for translation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", help="Enable debug logging.", action="store_true",
    )
    parser.add_argument(
        "--checkpoint", help="The checkpoint to load before training.", required=True,
    )
    parser.add_argument(
        "--text_encoder", help="Text Encoder type", type=str, required=True
    )
    parser.add_argument(
        "--model", help=f"Name of the model to run", type=str, required=True,
    )
    parser.add_argument(
        "--vocab_size", help="Size of the vocabulary", type=int, required=True,
    )
    parser.add_argument(
        "--message", help="Message to translate", type=str, required=True
    )
    args = parser.parse_args()
    logger = logging.initialize(experiment_name=args.model, debug=args.debug)

    print("Allo")
    return args, logger


args, logger = parse_args()
