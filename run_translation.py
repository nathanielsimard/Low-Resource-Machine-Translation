import translation_argument_parser  # isort:skip

import tensorflow as tf

import models
from src import dataloader, logging, preprocessing
from src.text_encoder import TextEncoderType

logger = logging.create_logger(__name__)


def translate(args):
    """Translate user's input."""
    # Used to load the train text encoders.
    text_encoder_type = TextEncoderType(args.text_encoder)
    train_dl = dataloader.AlignedDataloader(
        file_name_input="data/splitted_data/sorted_train_token.en",
        file_name_target="data/splitted_data/sorted_nopunctuation_lowercase_train_token.fr",
        vocab_size=args.vocab_size,
        text_encoder_type=text_encoder_type,
    )
    encoder_input = train_dl.encoder_input
    encoder_target = train_dl.encoder_target

    # Load the model.
    model = models.find(args, encoder_input.vocab_size, encoder_target.vocab_size)
    model.load(str(args.checkpoint))

    # Create the message to translate.
    message = preprocessing.add_start_end_token([args.message])[0]
    x = tf.convert_to_tensor([train_dl.encoder_input.encode(message)])

    # Translate the message.
    translated = model.translate(x, encoder_target, args.max_seq_length)
    translated_message = model.predictions(translated, encoder_target, logit=False)
    logger.info(f"Translation is {translated_message}")


def main():
    args = translation_argument_parser.args
    translate(args)


if __name__ == "__main__":
    try:
        main()
    except ValueError:
        # Logging is already done
        pass
