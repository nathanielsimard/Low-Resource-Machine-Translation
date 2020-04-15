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
from tmp_helpers.metrics import compute_bleu
from src.opennmt_preprocessing import (
    prepare_bpe_models,
    prepare_bpe_files,
    decode_bpe_file,
    get_bpe_vocab_files,
    build_vocabulary,
    concat_files,
    shuffle_file,
    get_vocab_file_names,
)

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


def train(
    model: onmt.models.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    learning_rate: tf.keras.optimizers.schedules.LearningRateSchedule,
    source_file: str,
    target_file: str,
    checkpoint_manager: tf.train.CheckpointManager,
    maximum_length=100,
    shuffle_buffer_size=-1,  # Uniform shuffle.
    train_steps=100000,
    save_every=1000,
    report_every=100,
    validation_source_file=None,
    validation_target_file=None,
    validate_every=2000,
    validate_now=False,
    bpe=False,
    bpe_combined=False,
):
    """Train a OpenNMT model.

    Arguments:
        model {onmt.models.Model} -- Model to train.
        optimizer {tf.keras.optimizers.Optimizer} -- Optimizer to use.
        learning_rate {tf.keras.optimizers.schedules.LearningRateSchedule} --
                        Learning rate schedule to use.
        source_file {str} -- Aligned source language file.
        target_file {str} -- Aligned target language file.
        checkpoint_manager {tf.train.CheckpointManager} -- Checkpoint manager.

    Keyword Arguments:
        maximum_length {int} -- [description] (default: {100})
        shuffle_buffer_size {int} -- [description] (default: {-1})
        save_every {int} -- [description] (default: {1000})
        report_every {int} -- [description] (default: {100})
        validation_source_file {[type]} -- [description] (default: {None})
        validation_target_file {[type]} -- [description] (default: {None})
        validate_every {int} -- [description] (default: {2000})
        validate_now {bool} -- [description] (default: {False})
        bpe {bool} -- [description] (default: {False})
        bpe_combined {bool} -- [description] (default: {False})

    Returns:
        [type] -- [description]
    """

    # Create the training dataset.
    dataset = model.examples_inputter.make_training_dataset(
        source_file,
        target_file,
        batch_size=3072,
        batch_type="tokens",
        shuffle_buffer_size=shuffle_buffer_size,
        length_bucket_width=1,  # Bucketize sequences by the same length for efficiency.
        maximum_features_length=maximum_length,
        maximum_labels_length=maximum_length,
    )

    @tf.function(input_signature=dataset.element_spec)
    def training_step(source, target):
        # Run the encoder.
        source_inputs = model.features_inputter(source, training=True)
        encoder_outputs, _, _ = model.encoder(
            source_inputs, source["length"], training=True
        )

        # Run the decoder.
        target_inputs = model.labels_inputter(target, training=True)
        decoder_state = model.decoder.initial_state(
            memory=encoder_outputs, memory_sequence_length=source["length"]
        )
        logits, _, _ = model.decoder(
            target_inputs, target["length"], state=decoder_state, training=True
        )

        # Compute the cross entropy loss.
        loss_num, loss_den, _ = onmt.utils.cross_entropy_sequence_loss(
            logits,
            target["ids_out"],
            target["length"],
            label_smoothing=0.1,
            average_in_time=True,
            training=True,
        )
        loss = loss_num / loss_den

        # Compute and apply the gradients.
        variables = model.trainable_variables
        gradients = optimizer.get_gradients(loss, variables)
        optimizer.apply_gradients(list(zip(gradients, variables)))
        return loss

    # Runs the training loop.
    for source, target in dataset:
        loss = training_step(source, target)
        step = optimizer.iterations.numpy()

        if step % validate_every == 0 or validate_now:
            output_file_name = f"predictions.{step}.txt"
            if validation_source_file is not None:
                tf.get_logger().info(
                    f"Saving validation predictions from {validation_source_file} to {output_file_name}"
                )
                translate(model, validation_source_file, output_file=output_file_name)
                if bpe:
                    output_file_name = decode_bpe_file(
                        output_file_name, combined=bpe_combined
                    )
                tf.get_logger().info(
                    f"Computing BLEU between from {validation_target_file} to {output_file_name}"
                )
                per_sentence_score, mean_score = compute_bleu(
                    output_file_name, validation_target_file
                )
                tf.get_logger().info(f"BLEU score {mean_score}")

        if step % report_every == 0:
            tf.get_logger().info(
                "Step = %d ; Learning rate = %f ; Loss = %f",
                step,
                learning_rate(step),
                loss,
            )
        if step % save_every == 0:
            tf.get_logger().info("Saving checkpoint for step %d", step)
            checkpoint_manager.save(checkpoint_number=step)
            tf.get_logger().info("Checkpoint saved.")

        if step == train_steps:
            break


def translate(
    model,
    source_file,
    batch_size=32,
    beam_size=1,
    output_file=None,
    show_progress=False,
):
    """Runs translation.
  Args:
    source_file: The source file.
    batch_size: The batch size to use.
    beam_size: The beam size to use. Set to 1 for greedy search.
  """

    # Create the inference dataset.
    dataset = model.examples_inputter.make_inference_dataset(source_file, batch_size)

    # @tf.function(input_signature=(dataset.element_spec,))
    def predict(source):
        # Run the encoder.
        source_length = source["length"]
        batch_size = tf.shape(source_length)[0]
        source_inputs = model.features_inputter(source)
        encoder_outputs, _, _ = model.encoder(source_inputs, source_length)

        # Prepare the decoding strategy.
        if beam_size > 1:
            encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
            source_length = tfa.seq2seq.tile_batch(source_length, beam_size)
            decoding_strategy = onmt.utils.BeamSearch(beam_size)
        else:
            decoding_strategy = onmt.utils.GreedySearch()

        # Run dynamic decoding.
        decoder_state = model.decoder.initial_state(
            memory=encoder_outputs, memory_sequence_length=source_length
        )
        decoded = model.decoder.dynamic_decode(
            model.labels_inputter,
            tf.fill([batch_size], onmt.START_OF_SENTENCE_ID),
            end_id=onmt.END_OF_SENTENCE_ID,
            initial_state=decoder_state,
            decoding_strategy=decoding_strategy,
            maximum_iterations=200,
        )
        target_lengths = decoded.lengths
        target_tokens = model.labels_inputter.ids_to_tokens.lookup(
            tf.cast(decoded.ids, tf.int64)
        )
        return target_tokens, target_lengths

    f = None
    if output_file is not None:
        f = open(output_file, "w")
    if show_progress and output_file is not None:
        print(f"Writing predictions to {output_file}")
    for source in dataset:
        if show_progress:
            print(".", end="", flush=True)
        batch_tokens, batch_length = predict(source)
        for tokens, length in zip(batch_tokens.numpy(), batch_length.numpy()):
            sentence = b" ".join(tokens[0][: length[0]])
            if f is not None:
                f.write(sentence.decode("utf-8") + "\n")
            else:
                print(sentence.decode("utf-8"))
    if output_file is not None:
        f.close()


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
            f"Training on {src}, {tgt}\nValidating on {valsrc}, {valtgt}.\nVocab = {src_vocab}, {tgt_vocab}\n BPE={args.bpe}"
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
