import argument_parser  # isort:skip

import random

import tensorflow as tf

import models
from src import dataloader, logging
from src.text_encoder import TextEncoderType
from src.training import base, scheduler
from src.training.back_translation import BackTranslationTraining
from src.training.default import Training
from src.training.pretraining import Pretraining

logger = logging.create_logger(__name__)

CACHE_DIR = ".cache"


def punctuation_training(args, loss_fn):
    """Train the model for the punctuation task."""
    text_encoder_type = _text_encoder_type(args.text_encoder)

    train_dl = dataloader.AlignedDataloader(
        file_name_input="data/splitted_english_data/sorted_clean_train.en",
        file_name_target="data/splitted_english_data/sorted_target_train.en",
        vocab_size=args.vocab_size,
        text_encoder_type=text_encoder_type,
        max_seq_length=args.max_seq_length,
        cache_dir=_cache_dir(args),
    )
    valid_dl = dataloader.AlignedDataloader(
        file_name_input="data/splitted_english_data/sorted_clean_valid.en",
        file_name_target="data/splitted_english_data/sorted_target_valid.en",
        vocab_size=args.vocab_size,
        text_encoder_type=text_encoder_type,
        encoder_input=train_dl.encoder_input,
        encoder_target=train_dl.encoder_target,
        max_seq_length=args.max_seq_length,
        cache_dir=_cache_dir(args),
    )
    model = models.find(
        args, train_dl.encoder_input.vocab_size, train_dl.encoder_target.vocab_size
    )
    optim = _create_optimizer(model.embedding_size, args)
    training = Training(
        model, train_dl, valid_dl, [base.Metrics.ABSOLUTE_ACC, base.Metrics.BLEU]
    )
    training.run(
        loss_fn,
        optim,
        batch_size=args.batch_size,
        num_epoch=args.epochs,
        checkpoint=args.checkpoint,
    )


def default_training(args, loss_fn):
    """Train the model."""
    text_encoder_type = _text_encoder_type(args.text_encoder)

    train_dl = dataloader.AlignedDataloader(
        file_name_input=args.src_train,
        file_name_target=args.target_train,
        vocab_size=args.vocab_size,
        text_encoder_type=text_encoder_type,
        max_seq_length=args.max_seq_length,
        cache_dir=_cache_dir(args),
    )
    valid_dl = dataloader.AlignedDataloader(
        file_name_input=args.src_valid,
        file_name_target=args.target_valid,
        vocab_size=args.vocab_size,
        text_encoder_type=text_encoder_type,
        encoder_input=train_dl.encoder_input,
        encoder_target=train_dl.encoder_target,
        max_seq_length=args.max_seq_length,
        cache_dir=_cache_dir(args),
    )
    model = models.find(
        args, train_dl.encoder_input.vocab_size, train_dl.encoder_target.vocab_size
    )
    optim = _create_optimizer(model.embedding_size, args)
    training = Training(model, train_dl, valid_dl, [base.Metrics.BLEU])
    training.run(
        loss_fn,
        optim,
        batch_size=args.batch_size,
        num_epoch=args.epochs,
        checkpoint=args.checkpoint,
    )


def pretraining(args, loss_fn):
    """Pretraining the model."""
    text_encoder_type = _text_encoder_type(args.text_encoder)

    train_dl = dataloader.UnalignedDataloader(
        file_name=args.src_train,
        vocab_size=args.vocab_size,
        text_encoder_type=text_encoder_type,
        max_seq_length=args.max_seq_length,
        cache_dir=_cache_dir(args),
    )
    valid_dl = dataloader.UnalignedDataloader(
        file_name=args.src_valid,
        vocab_size=args.vocab_size,
        text_encoder_type=text_encoder_type,
        encoder=train_dl.encoder,
        max_seq_length=args.max_seq_length,
        cache_dir=_cache_dir(args),
    )
    model = models.find(args, train_dl.encoder.vocab_size, train_dl.encoder.vocab_size)
    optim = _create_optimizer(model.embedding_size, args)
    pretraining = Pretraining(model, train_dl, valid_dl)
    pretraining.run(
        loss_fn,
        optim,
        batch_size=args.batch_size,
        num_epoch=args.epochs,
        checkpoint=args.checkpoint,
    )


def back_translation_training(args, loss_fn):
    """Train the model with back translation."""
    text_encoder_type = _text_encoder_type(args.text_encoder)

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
        file_name_target="data/splitted_data/sorted_nopunctuation_lowercase_val_token.fr",
        vocab_size=args.vocab_size,
        encoder_input=train_dl.encoder,
        encoder_target=train_dl_reverse.encoder,
        text_encoder_type=text_encoder_type,
        max_seq_length=args.max_seq_length,
        cache_dir=_cache_dir(args),
    )

    logger.info("Creating reversed training aligned dataloader ...")
    aligned_train_dl_reverse = dataloader.AlignedDataloader(
        file_name_input="data/splitted_data/sorted_nopunctuation_lowercase_val_token.fr",
        file_name_target="data/splitted_data/sorted_train_token.en",
        vocab_size=args.vocab_size,
        encoder_input=aligned_train_dl.encoder_target,
        encoder_target=aligned_train_dl.encoder_input,
        text_encoder_type=text_encoder_type,
        max_seq_length=args.max_seq_length,
        cache_dir=_cache_dir(args),
    )

    logger.info("Creating valid aligned dataloader ...")
    aligned_valid_dl = dataloader.AlignedDataloader(
        file_name_input="data/splitted_data/sorted_val_token.en",
        file_name_target="data/splitted_data/sorted_nopunctuation_lowercase_val_token.fr",
        vocab_size=args.vocab_size,
        encoder_input=aligned_train_dl.encoder_input,
        encoder_target=aligned_train_dl.encoder_target,
        text_encoder_type=text_encoder_type,
        max_seq_length=args.max_seq_length,
        cache_dir=_cache_dir(args),
    )

    logger.info("Creating reversed valid aligned dataloader ...")
    aligned_valid_dl_reverse = dataloader.AlignedDataloader(
        file_name_input="data/splitted_data/sorted_nopunctuation_lowercase_val_token.frs",
        file_name_target="data/splitted_data/sorted_val_token.en",
        vocab_size=args.vocab_size,
        encoder_input=aligned_train_dl_reverse.encoder_input,
        encoder_target=aligned_train_dl_reverse.encoder_target,
        text_encoder_type=text_encoder_type,
        max_seq_length=args.max_seq_length,
        cache_dir=_cache_dir(args),
    )

    model = models.find(
        args,
        aligned_train_dl.encoder_input.vocab_size,
        aligned_train_dl.encoder_target.vocab_size,
    )

    optim = _create_optimizer(model.embedding_size, args)
    model_reverse = models.find(
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
    text_encoder_type = _text_encoder_type(args.text_encoder)
    # Used to load the train text encoders.
    train_dl = dataloader.AlignedDataloader(
        file_name_input=args.src_train,
        file_name_target=args.target_train,
        vocab_size=args.vocab_size,
        text_encoder_type=text_encoder_type,
        max_seq_length=args.max_seq_length,
        cache_dir=_cache_dir(args),
    )
    test_dl = dataloader.AlignedDataloader(
        file_name_input="data/splitted_data/test/test_token10000.en",
        file_name_target="data/splitted_data/test/test_token10000.fr",
        vocab_size=args.vocab_size,
        encoder_input=train_dl.encoder_input,
        encoder_target=train_dl.encoder_target,
        text_encoder_type=text_encoder_type,
        max_seq_length=args.max_seq_length,
        cache_dir=_cache_dir(args),
    )
    model = models.find(
        args, train_dl.encoder_input.vocab_size, train_dl.encoder_target.vocab_size
    )
    base.test(model, loss_fn, test_dl, args.batch_size, args.checkpoint)


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


def _create_optimizer(embedding_size, args):
    if type(args.lr) is float:
        learning_rate = args.lr
    else:
        learning_rate = scheduler.Schedule(embedding_size)

    return tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-09
    )


def _text_encoder_type(text_encoder: str) -> TextEncoderType:
    try:
        return TextEncoderType(text_encoder)
    except Exception as e:
        logger.error(f"Text encoder type {text_encoder} is not valid.")
        raise ValueError(e)


def _cache_dir(args):
    if args.no_cache:
        return None

    return CACHE_DIR


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

        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    if args.task not in TASK.keys():
        logger.error(
            f"Task {args.task} is not supported, available tasks are {TASK.keys()}."
        )
    else:
        logger.info(f"Executing task {args.task}.")
        task = TASK[args.task]
        task(args, loss_function)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Logging is already done
        print(e)
