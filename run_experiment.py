import argument_parser  # isort:skip

import random

import tensorflow as tf

from src import dataloader, logging
from src.model import gru_attention, lstm, lstm_luong_attention, transformer, masked_lm
from src.text_encoder import TextEncoderType
from src.training import base
from src.training.back_translation import BackTranslationTraining
from src.training.base import BasicMachineTranslationTraining

logger = logging.create_logger(__name__)


def create_lstm(args, input_vocab_size, target_vocab_size):
    return lstm.Lstm(input_vocab_size + 1, target_vocab_size + 1)


def create_transformer(args, input_vocab_size, target_vocab_size):
    model = transformer.Transformer(
        num_layers=2,
        num_heads=2,
        dff=256,
        d_model=256,
        input_vocab_size=input_vocab_size + 1,
        target_vocab_size=target_vocab_size + 1,
        pe_input=input_vocab_size + 1,
        pe_target=target_vocab_size + 1,
        rate=0.1,
    )
    return model


def create_gru_attention(args, input_vocab_size, target_vocab_size):
    return gru_attention.GRU(input_vocab_size + 1, target_vocab_size + 1)


def create_lstm_luong_attention(args, input_vocab_size, target_vocab_size):
    return lstm_luong_attention.LSTM_ATTENTION(
        input_vocab_size + 1, target_vocab_size + 1
    )


def create_demi_bert(args, input_vocab_size, target_vocab_size):
    return masked_lm.DemiBERT(
        num_layers=2,
        embedding_size=256,
        num_heads=4,
        dff=256,
        vocab_size=input_vocab_size,
        max_pe=input_vocab_size,
        dropout=0.1,
    )


MODELS = {
    lstm.NAME: create_lstm,
    transformer.NAME: create_transformer,
    gru_attention.NAME: create_gru_attention,
    lstm_luong_attention.NAME: create_lstm_luong_attention,
    masked_lm.NAME: create_demi_bert,
}


def find_model(args, input_vocab_size, target_vocab_size):
    try:
        return MODELS[args.model](args, input_vocab_size, target_vocab_size)
    except KeyError as e:
        logger.error(
            f"Model {args.model} is not supported, available models are {list(MODELS.keys())}."
        )
        raise ValueError(e)


def punctuation_training(args, loss_fn):
    """Train the model for the punctuation task."""
    text_encoder_type = TextEncoderType(args.text_encoder)

    optim = tf.keras.optimizers.Adam(learning_rate=args.lr)
    train_dl = dataloader.AlignedDataloader(
        file_name_input="data/splitted_english_data/sorted_clean_train.en",
        file_name_target="data/splitted_english_data/sorted_target_train.en",
        vocab_size=args.vocab_size,
        text_encoder_type=text_encoder_type,
        max_seq_lenght=args.max_seq_lenght,
    )
    valid_dl = dataloader.AlignedDataloader(
        file_name_input="data/splitted_english_data/sorted_clean_valid.en",
        file_name_target="data/splitted_english_data/sorted_target_valid.en",
        vocab_size=args.vocab_size,
        text_encoder_type=text_encoder_type,
        encoder_input=train_dl.encoder_input,
        encoder_target=train_dl.encoder_target,
        max_seq_lenght=args.max_seq_lenght,
    )
    model = find_model(
        args, train_dl.encoder_input.vocab_size, train_dl.encoder_target.vocab_size
    )
    training = BasicMachineTranslationTraining(model, train_dl, valid_dl, [])
    training.run(
        loss_fn,
        optim,
        batch_size=args.batch_size,
        num_epoch=args.epochs,
        checkpoint=args.checkpoint,
    )


def default_training(args, loss_fn):
    """Train the model."""
    text_encoder_type = TextEncoderType(args.text_encoder)

    optim = tf.keras.optimizers.Adam(learning_rate=args.lr)
    train_dl = dataloader.AlignedDataloader(
        file_name_input="data/splitted_data/sorted_train_token.en",
        file_name_target="data/splitted_data/sorted_nopunctuation_lowercase_train_token.fr",
        vocab_size=args.vocab_size,
        text_encoder_type=text_encoder_type,
        max_seq_length=args.max_seq_length,
    )
    valid_dl = dataloader.AlignedDataloader(
        file_name_input="data/splitted_data/sorted_val_token.en",
        file_name_target="data/splitted_data/sorted_nopunctuation_lowercase_val_token.fr",
        vocab_size=args.vocab_size,
        text_encoder_type=text_encoder_type,
        encoder_input=train_dl.encoder_input,
        encoder_target=train_dl.encoder_target,
        max_seq_length=args.max_seq_length,
    )
    model = find_model(
        args, train_dl.encoder_input.vocab_size, train_dl.encoder_target.vocab_size
    )
    training = BasicMachineTranslationTraining(
        model, train_dl, valid_dl, [base.Metrics.BLEU]
    )
    training.run(
        loss_fn,
        optim,
        batch_size=args.batch_size,
        num_epoch=args.epochs,
        checkpoint=args.checkpoint,
    )

def pretraining

def back_translation_training(args, loss_fn):
    """Train the model with back translation."""
    text_encoder_type = TextEncoderType(args.text_encoder)

    optim = tf.keras.optimizers.Adam(args.lr)
    logger.info("Creating training unaligned dataloader ...")
    train_dl = dataloader.UnalignedDataloader(
        "data/unaligned.en",
        args.vocab_size,
        text_encoder_type=text_encoder_type,
        max_seq_length=args.max_seq_length,
    )
    logger.info(f"English vocab size: {train_dl.encoder.vocab_size}")

    logger.info("Creating reversed training unaligned dataloader ...")
    train_dl_reverse = dataloader.UnalignedDataloader(
        "data/unaligned.fr",
        args.vocab_size,
        text_encoder_type=text_encoder_type,
        max_seq_length=args.max_seq_length,
    )
    logger.info(f"French vocab size: {train_dl_reverse.encoder.vocab_size}")

    logger.info("Creating training aligned dataloader ...")
    aligned_train_dl = dataloader.AlignedDataloader(
        file_name_input="data/splitted_data/sorted_train_token.en",
        file_name_target="data/splitted_data/sorted_train_token.fr",
        vocab_size=args.vocab_size,
        encoder_input=train_dl.encoder,
        encoder_target=train_dl_reverse.encoder,
        text_encoder_type=text_encoder_type,
        max_seq_length=args.max_seq_length,
    )

    logger.info("Creating reversed training aligned dataloader ...")
    aligned_train_dl_reverse = dataloader.AlignedDataloader(
        file_name_input="data/splitted_data/sorted_train_token.fr",
        file_name_target="data/splitted_data/sorted_train_token.en",
        vocab_size=args.vocab_size,
        encoder_input=aligned_train_dl.encoder_target,
        encoder_target=aligned_train_dl.encoder_input,
        text_encoder_type=text_encoder_type,
        max_seq_length=args.max_seq_length,
    )

    logger.info("Creating valid aligned dataloader ...")
    aligned_valid_dl = dataloader.AlignedDataloader(
        file_name_input="data/splitted_data/sorted_val_token.en",
        file_name_target="data/splitted_data/sorted_val_token.fr",
        vocab_size=args.vocab_size,
        encoder_input=aligned_train_dl.encoder_input,
        encoder_target=aligned_train_dl.encoder_target,
        text_encoder_type=text_encoder_type,
        max_seq_length=args.max_seq_length,
    )

    logger.info("Creating reversed valid aligned dataloader ...")
    aligned_valid_dl_reverse = dataloader.AlignedDataloader(
        file_name_input="data/splitted_data/sorted_val_token.fr",
        file_name_target="data/splitted_data/sorted_val_token.en",
        vocab_size=args.vocab_size,
        encoder_input=aligned_train_dl_reverse.encoder_input,
        encoder_target=aligned_train_dl_reverse.encoder_target,
        text_encoder_type=text_encoder_type,
        max_seq_length=args.max_seq_length,
    )

    model = find_model(
        args,
        aligned_train_dl.encoder_input.vocab_size,
        aligned_train_dl.encoder_target.vocab_size,
    )

    model_reverse = find_model(
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
    text_encoder_type = TextEncoderType(args.text_encoder)
    # Used to load the train text encoders.
    train_dl = dataloader.AlignedDataloader(
        file_name_input="data/splitted_data/sorted_train_token.en",
        file_name_target="data/splitted_data/sorted_train_token.fr",
        vocab_size=args.vocab_size,
        text_encoder_type=text_encoder_type,
        max_seq_length=args.max_seq_length,
    )
    test_dl = dataloader.AlignedDataloader(
        file_name_input="data/splitted_data/sorted_test_token.en",
        file_name_target="data/splitted_data/sorted_test_token.fr",
        vocab_size=args.vocab_size,
        encoder_input=train_dl.encoder_input,
        encoder_target=train_dl.encoder_target,
        text_encoder_type=text_encoder_type,
        max_seq_length=args.max_seq_length,
    )
    model = find_model(
        args, train_dl.encoder_input.vocab_size, train_dl.encoder_target.vocab_size
    )
    base.test(model, loss_fn, test_dl, args.batch_size, args.test)


TASK = {
    "default-training": default_training,
    "punctuation-training": punctuation_training,
    "back-translation-training": back_translation_training,
    "test": test,
    "pretraining": pretraining,
}


def _log_args(args):
    args_output = "Arguments Value: \n"
    for arg in vars(args):
        args_output += f"{arg}:  {getattr(args, arg)}\n"
    logger.info(args_output)


def main():
    args = argument_parser.args
    _log_args(args)

    if not args.random_seed:
        random.seed(args.seed)
        tf.random.set_seed(args.seed)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    try:
        logger.info(f"Executing task {args.task}.")
        training = TASK[args.task]
        training(args, loss_function)
    except KeyError:
        logger.error(
            f"Task {args.task} is not supported, available tasks are {TASK.keys()}."
        )


if __name__ == "__main__":
    try:
        main()
    except ValueError:
        # Logging is already done
        pass
