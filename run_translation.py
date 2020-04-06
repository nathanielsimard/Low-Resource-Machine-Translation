import translation_argument_parser  # isort:skip

import tensorflow as tf

from src.text_encoder import TextEncoderType
from src import dataloader, logging
from src import preprocessing
import models


logger = logging.create_logger(__name__)


def translate(args):
    """Translate user's input."""
    text_encoder_type = TextEncoderType(args.text_encoder)
    # Used to load the train text encoders.
    train_dl = dataloader.AlignedDataloader(
        file_name_input="data/splitted_data/sorted_train_token.en",
        file_name_target="data/splitted_data/sorted_nopunctuation_lowercase_val_token.fr",
        vocab_size=args.vocab_size,
        text_encoder_type=text_encoder_type,
    )
    model = models.find(
        args, train_dl.encoder_input.vocab_size, train_dl.encoder_target.vocab_size
    )
    model.load(str(args.checkpoint))
    message = preprocessing.add_start_end_token([args.message])[0]
    x = tf.convert_to_tensor([train_dl.encoder_input.encode(message)])

    pred = model.translate(x, train_dl.encoder_input)
    pred_message = train_dl.encoder_target.decode(pred)
    print(f"Translation is {pred_message}")


def main():
    args = translation_argument_parser.args
    translate(args)


if __name__ == "__main__":
    try:
        main()
    except ValueError:
        # Logging is already done
        pass
