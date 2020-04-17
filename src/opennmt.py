import opennmt as onmt
import tensorflow as tf
from tmp_helpers.metrics import compute_bleu
from src.opennmt_preprocessing import decode_bpe_file


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
