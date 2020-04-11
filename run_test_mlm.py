import argparse
from src.training import pretraining
from src.dataloader import UnalignedDataloader
from src import preprocessing
import tensorflow as tf
import models
from src.text_encoder import TextEncoderType


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="Model to use for predicting a masked token",
        type=str,
        default="demi-bert",
    )
    parser.add_argument(
        "--checkpoint", help="Which epoch to load", type=int, required=True
    )
    parser.add_argument(
        "--message",
        type=str,
        help="Message to decrypt. Must contain a <mask> token.",
        required=True,
    )
    parser.add_argument(
        "--std",
        help="For users who want to also log their tests in STDOUT",
        action="store_true",
    )
    parser.add_argument(
        "--debug", help="To allow debugging in the logs", action="store_true"
    )

    args = parser.parse_args()

    return args


def predict(args):
    """Translate user's input."""
    # Used to load the train text encoders.
    print("Instanciating dataloader...")
    train_dl = UnalignedDataloader(
        file_name="data/splitted_english_data/sorted_clean_train.en",
        cache_dir=".cache/data/splitted_english_data/sorted_clean_train.en",
        text_encoder_type=TextEncoderType("subword"),
        vocab_size=15000,
    )
    encoder = train_dl.encoder
    print("Creating model...")

    # Load the model.
    model = models.find(args, encoder.vocab_size, encoder.vocab_size)
    model.load(str(args.checkpoint))

    # Create the message to translate.
    message = preprocessing.add_start_end_token([args.message])[0]
    print(message)
    x = tf.convert_to_tensor([encoder.encode(message)])
    print(x)
    pretraining.test(x, model, encoder)


def main():
    args = parse_args()
    predict(args)


if __name__ == "__main__":
    try:
        main()
    except ValueError:
        # Logging is already done
        pass
