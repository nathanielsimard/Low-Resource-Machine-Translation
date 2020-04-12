# From:
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

import argparse
import logging
import tensorflow as tf
import tensorflow_addons as tfa
import opennmt as onmt
from tmp_helpers.metrics import compute_bleu
from build_vocabulary import (
    prepare_bpe_models,
    prepare_bpe_files,
    decode_bpe_file,
    get_bpe_vocab_files,
    build_vocabulary,
    concat_files,
    shuffle_file,
)

tf.get_logger().setLevel(logging.INFO)

# tfa.options.TF_ADDONS_PY_OPS=True

# Define the model. For the purpose of this example, the model components
# (encoder, decoder, etc.) will be called separately.
model = onmt.models.SequenceToSequence(
    source_inputter=onmt.inputters.WordEmbedder(embedding_size=512),
    target_inputter=onmt.inputters.WordEmbedder(embedding_size=512),
    encoder=onmt.encoders.SelfAttentionEncoder(
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
    ),
    decoder=onmt.decoders.SelfAttentionDecoder(
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
    ),
)

model = onmt.models.TransformerBase()

# Define the learning rate schedule and the optimizer.
learning_rate = onmt.schedules.NoamDecay(scale=2.0, model_dim=512, warmup_steps=8000)
optimizer = tfa.optimizers.LazyAdam(learning_rate)

# Track the model and optimizer weights.
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)


def train(
    source_file,
    target_file,
    checkpoint_manager,
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
):
    """Runs the training loop.
  Args:
    source_file: The source training file.
    target_file: The target training file.
    checkpoint_manager: The checkpoint manager.
    maximum_length: Filter sequences longer than this.
    shuffle_buffer_size: How many examples to load for shuffling.
    train_steps: Train for this many iterations.
    save_every: Save a checkpoint every this many iterations.
    report_every: Report training progress every this many iterations.
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
                translate(validation_source_file, output_file=output_file_name)
                if bpe:
                    output_file_name = decode_bpe_file(output_file_name)
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


def translate(source_file, batch_size=32, beam_size=1, output_file=None):
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

    for source in dataset:
        batch_tokens, batch_length = predict(source)
        for tokens, length in zip(batch_tokens.numpy(), batch_length.numpy()):
            sentence = b" ".join(tokens[0][: length[0]])
            if f is not None:
                f.write(sentence.decode("utf-8") + "\n")
            else:
                print(sentence.decode("utf-8"))
    if output_file is not None:
        f.close()


def main():
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

    combined = args.bpe_combined
    if args.monosrc != "":
        combined = True  # Combined vocabulary must be used for monolingual data!
        print("Using combined BPE vocabulary since monolingual data is used!")
    src = args.src
    tgt = args.tgt
    valsrc = args.valsrc
    valtgt = args.valtgt
    src_vocab = "src_vocab.txt"
    tgt_vocab = "tgt_vocab.txt"
    vocab_size = args.vocab_size

    # if args.run == "translate":
    #    tgt = None

    if args.bpe:
        # Prepare Byte-Pair Encore model + Byte-Pair Encoded Files.
        vocab_size = args.bpe_vocab_size
        if args.run == "train":
            prepare_bpe_models(src, tgt, combined=combined, vocab_size=vocab_size)
            prepare_bpe_files(valsrc, valtgt, combined=combined)
            valsrc += ".bpe"
        prepare_bpe_files(src, tgt, combined=combined)
        src += ".bpe"
        if tgt is not None:
            tgt += ".bpe"

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
            prepare_bpe_files(btsrc, bttgt, combined=combined)
            btsrc += ".bpe"
            bttgt += ".bpe"
        else:
            tf.get_logger.info(
                "Warning: Back-translation was not tested without BPE. There could be bugs!"
            )
        tmp_btsrc = "btsrc.tmp"
        tmp_bttgt = "bttgt.tmp"
        concat_files(btsrc, src, tmp_btsrc)
        concat_files(bttgt, tgt, tmp_bttgt)
        shuffle_file(tmp_btsrc)
        shuffle_file(tmp_bttgt)
        src = tmp_btsrc
        tgt = tmp_bttgt

    # Add additionnal monolingual data if requested.
    if args.monosrc != "":
        tmp_monosrc = "monosrc.tmp"
        tmp_monotgt = "monotgt.tmp"
        prepare_bpe_files(args.monosrc, args.monosrc)
        concat_files(
            src, args.monosrc + ".bpe", tmp_monosrc, lines1=None, lines2=args.monolen
        )
        concat_files(
            tgt, args.monosrc + ".bpe", tmp_monotgt, lines1=None, lines2=args.monolen
        )
        shuffle_file(tmp_monosrc, seed=1234, inplace=True)
        shuffle_file(tmp_monotgt, seed=1234, inplace=True)
        src = tmp_monosrc
        tgt = tmp_monotgt

    data_config = {
        "source_vocabulary": src_vocab,
        "target_vocabulary": tgt_vocab,
    }

    model.initialize(data_config)

    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, args.model_dir, max_to_keep=5
    )
    if checkpoint_manager.latest_checkpoint is not None:
        tf.get_logger().info(
            "Restoring parameters from %s", checkpoint_manager.latest_checkpoint
        )
        checkpoint.restore(checkpoint_manager.latest_checkpoint)

    if args.run == "train":
        tf.get_logger().info(
            f"Training on {src}, {tgt}\nValidating on {valsrc}, {valtgt}.\nVocab = {src_vocab}, {tgt_vocab}\n BPE={args.bpe}"
        )
        train(
            src,
            tgt,
            checkpoint_manager,
            validation_source_file=valsrc,
            validation_target_file=valtgt,
            validate_now=args.validate_now,
            bpe=args.bpe,
        )
    elif args.run == "translate":
        tf.get_logger().info(f"Translating {src} file to {args.output}")
        translate(src, output_file=args.output)
        if args.bpe:
            output_file_name = decode_bpe_file(args.output)
        tf.get_logger().info(f"BPE decoded {args.output} file to {output_file_name}")


if __name__ == "__main__":
    main()
