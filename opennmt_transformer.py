# Based on code from:
# https://github.com/OpenNMT/OpenNMT-tf/blob/master/examples/library/custom_transformer_training.py
"""This example demonstrates how to train a Transformer model with a custom
training loop in about 200 lines of code.
The purpose of this example is to showcase selected lower-level OpenNMT-tf APIs
that can be useful in other projects:
* efficient training dataset (with shuffling, bucketing, batching, prefetching, etc.)
* inputter/encoder/decoder API
* dynamic decoding API
Producing a SOTA model is NOT a goal: this usually requires extra steps such as
training a bigger model, using a larger batch size via multi GPU training and/or
gradient accumulation, etc.
"""
import tempfile
import argparse
import logging
import tensorflow as tf
import tensorflow_addons as tfa
import opennmt as onmt

# from tmp_helpers.metrics import compute_bleu

from src.opennmt_preprocessing import (
    prepare_bpe_models,
    prepare_bpe_files,
    decode_bpe_file,
    build_vocabulary,
    concat_files,
    shuffle_file,
    get_vocab_file_names,
)

from src.opennmt import train, translate

tf.get_logger().setLevel(logging.INFO)


def init_model():
    """Initialise the OpenNMT transformer model.

    Returns:
    onmt.models.Model -- the OpenNMT model.
    tf.train.Checkpoint -- TensorFlow checkpoint for the model.
    tf.keras.optimizers.Optimizer -- Optimized for the model.
    tf.keras.optimizers.schedules.LearningRateSchedule -- Learning
                                        rate schedule for the model
    """
    model = onmt.models.TransformerBase()
    learning_rate = onmt.schedules.NoamDecay(
        scale=2.0, model_dim=512, warmup_steps=8000
    )
    optimizer = tfa.optimizers.LazyAdam(learning_rate)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    return model, checkpoint, optimizer, learning_rate


def init_checkpoint_manager_and_load_latest_checkpoint(
    checkpoint, model_dir="checkpoint"
) -> tf.train.CheckpointManager:
    """Initialise the checkpoint manager, and load the latest checkpoint.

    Arguments:
        checkpoint {TensorFlow Checkpoint} -- TensorFlow checkpoint.

    Keyword Arguments:
        model_dir {str} -- folder where to read/store the
                           checkpoints (default: {"checkpoint"})

    Returns:
        [tf.train.CheckpointManager] -- The checkpoint manager.
    """
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, model_dir, max_to_keep=5
    )
    if checkpoint_manager.latest_checkpoint is not None:
        tf.get_logger().info(
            "Restoring parameters from %s", checkpoint_manager.latest_checkpoint
        )
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
    return checkpoint_manager


def init_data_config(model, src_vocab, tgt_vocab):
    data_config = {
        "source_vocabulary": src_vocab,
        "target_vocabulary": tgt_vocab,
    }
    model.initialize(data_config)


def main():
    model, checkpoint, optimizer, learning_rate = init_model()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("run", choices=["train", "translate"], help="Run type.")
    parser.add_argument("--src", required=True, help="Path to the source file.")
    parser.add_argument("--tgt", help="Path to the target file.")
    parser.add_argument("--valsrc", help="Path to the validation source file.")
    parser.add_argument("--valtgt", help="Path to the validation target file.")
    parser.add_argument("--bpe", help="Enables Byte-Pair Encoding", action="store_true")
    parser.add_argument("--vocab_size", help="Vocabulary Size", default=16000)
    parser.add_argument("--bpe_vocab_size", help="BPE Vocabulary Size", default=4000)
    parser.add_argument("--seed", help="Random seed for the experiment", default=1234)
    parser.add_argument(
        "--monosrc",
        help="Monolingual data source (Target language).",
        type=str,
        default="",
    )
    parser.add_argument("--btsrc", help="Back-translation source file")
    parser.add_argument("--bttgt", help="Back-translation target file")

    parser.add_argument(
        "--monolen", help="Number of monolingual samples to consider.", default=20000
    )
    parser.add_argument(
        "--bpe_combined",
        help="Use combined BPE vocabulary for both languages",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--validate_now",
        help="Skips training and validate at current checkpoint",
        action="store_true",
    )
    parser.add_argument(
        "--output", help="Filename for translated output.", default="output.txt"
    )

    # parser.add_argument(
    #    "--src_vocab", required=True, help="Path to the source vocabulary."
    # )
    # parser.add_argument(
    #    "--tgt_vocab", required=True, help="Path to the target vocabulary."
    # )
    parser.add_argument(
        "--model_dir",
        default="checkpoint",
        help="Directory where checkpoint are written.",
    )
    args = parser.parse_args()

    # Random seed.
    tf.random.set_seed(args.seed)

    combined = args.bpe_combined
    if args.monosrc != "":
        combined = True  # Combined vocabulary must be used for monolingual data!
        tf.get_logger().info(
            "Using combined BPE vocabulary since monolingual data is used!"
        )
    src = args.src
    tgt = args.tgt
    valsrc = args.valsrc
    valtgt = args.valtgt
    src_vocab, tgt_vocab = get_vocab_file_names(args.model_dir)
    vocab_size = args.vocab_size

    # if args.run == "translate":
    #    tgt = None

    if args.bpe:
        # Prepare Byte-Pair Encore model + Byte-Pair Encoded Files.
        vocab_size = args.bpe_vocab_size
        if args.run == "train":
            prepare_bpe_models(src, tgt, combined=combined, vocab_size=vocab_size)
            valsrc, valtgt = prepare_bpe_files(valsrc, valtgt, combined=combined)
        src, tgt = prepare_bpe_files(src, tgt, combined=combined)
        # src += ".bpe"
        # if tgt is not None:
        #    tgt += ".bpe"
        # valtgt += ".bpe" We compare againt the real version of the validation file.

    # Rebuilds the vocabulary from scratch using only the input data.
    if args.run == "train":
        if not combined:
            build_vocabulary(src, src_vocab, vocab_size)
            build_vocabulary(tgt, tgt_vocab, vocab_size)
        else:
            # Combined vocabulary!
            concat_files(src, tgt, "all.tmp")
            build_vocabulary("all.tmp", src_vocab, vocab_size)
            build_vocabulary("all.tmp", tgt_vocab, vocab_size)

    # Add back-tranlated data if requested.
    if args.btsrc is not None:
        btsrc = args.btsrc
        bttgt = args.bttgt
        if bttgt is None:
            tf.get_logger().error("Back-translation target must be supplied")
            exit()
        if args.bpe:
            btsrc, bttgt = prepare_bpe_files(btsrc, bttgt, combined=combined)
            # btsrc += ".bpe"
            # bttgt += ".bpe"
        else:
            tf.get_logger.info(
                "Warning: Back-translation was not tested without BPE. There could be bugs!"
            )
        tmp_btsrc = "btsrc.tmp"
        tmp_bttgt = "bttgt.tmp"
        concat_files(btsrc, src, tmp_btsrc)
        concat_files(bttgt, tgt, tmp_bttgt)
        shuffle_file(tmp_btsrc, seed=args.seed)
        shuffle_file(tmp_bttgt, seed=args.seed)
        src = tmp_btsrc
        tgt = tmp_bttgt

    # Add additionnal monolingual data if requested.
    if args.monosrc != "":
        tmp_monosrc = "monosrc.tmp"
        tmp_monotgt = "monotgt.tmp"
        if not args.bpe:
            tf.get_logger().error("Monolingual data can only be used with BPE!")
            exit()
        prepare_bpe_files(args.monosrc, None, combined=combined)
        concat_files(
            src, args.monosrc + ".bpe", tmp_monosrc, lines1=None, lines2=args.monolen
        )
        concat_files(
            tgt, args.monosrc + ".bpe", tmp_monotgt, lines1=None, lines2=args.monolen
        )
        shuffle_file(tmp_monosrc, seed=args.seed, inplace=True)
        shuffle_file(tmp_monotgt, seed=args.seed, inplace=True)
        src = tmp_monosrc
        tgt = tmp_monotgt

    init_data_config(model, src_vocab, tgt_vocab)
    # data_config = {
    #    "source_vocabulary": src_vocab,
    #    "target_vocabulary": tgt_vocab,
    # }
    #
    # model.initialize(data_config)

    checkpoint_manager = init_checkpoint_manager_and_load_latest_checkpoint(
        checkpoint, args.model_dir
    )

    # checkpoint_manager = tf.train.CheckpointManager(
    #    checkpoint, args.model_dir, max_to_keep=5
    # )
    # if checkpoint_manager.latest_checkpoint is not None:
    #    tf.get_logger().info(
    #        "Restoring parameters from %s", checkpoint_manager.latest_checkpoint
    #    )
    #    checkpoint.restore(checkpoint_manager.latest_checkpoint)

    if args.run == "train":
        tf.get_logger().info(
            f"Training on {src}, {tgt}\nValidating on {valsrc}, {valtgt}.\n"
            + f"Vocab = {src_vocab}, {tgt_vocab}\n BPE={args.bpe}"
        )
        train(
            model,
            optimizer,
            learning_rate,
            src,
            tgt,
            checkpoint_manager,
            validation_source_file=valsrc,
            validation_target_file=valtgt,
            validate_now=args.validate_now,
            bpe=args.bpe,
            bpe_combined=combined,
        )
    elif args.run == "translate":
        temp = tempfile.NamedTemporaryFile()
        tf.get_logger().info(f"Translating {src} file to {temp}")
        with tempfile.NamedTemporaryFile() as f:
            temp = f.name
        translate(model, src, output_file=temp)
        if args.bpe:
            output_file_name = decode_bpe_file(temp)
        else:
            import shutil

            shutil.copyfile(temp, output_file_name)
        tf.get_logger().info(f"BPE decoded {temp} file to {output_file_name}")


if __name__ == "__main__":
    main()
