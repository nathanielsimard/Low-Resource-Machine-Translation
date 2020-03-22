import os
import subprocess
from datetime import datetime
from typing import List

import numpy as np
import tensorflow as tf

from src.dataloader import Dataloader


def run(
    model,
    loss_fn: tf.keras.losses,
    optimizer: tf.keras.optimizers,
    train_dataloader: Dataloader,
    valid_dataloader: Dataloader,
    batch_size: int,
    num_epoch: int,
):
    """Training session."""
    train_dataset = train_dataloader.create_dataset()
    valid_dataset = valid_dataloader.create_dataset()

    directory = os.path.join("results", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    os.makedirs(directory, exist_ok=True)

    for epoch in range(1, num_epoch + 1):
        train_predictions: List[str] = []
        valid_predictions: List[str] = []
        for enc_inputs, dec_inputs, targets in train_dataset.padded_batch(
            batch_size, padded_shapes=([None], [None], [None])
        ):
            with tf.GradientTape() as tape:
                outputs = model((enc_inputs, dec_inputs), training=True)
                print(outputs)
                print(targets)
                for sentence in np.argmax(outputs.numpy(), axis=2):
                    train_predictions.append(
                        train_dataloader.encoder_target.decode(sentence + 1)
                    )
                loss = loss_fn(targets, outputs)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        for enc_inputs, dec_inputs, targets in valid_dataset.padded_batch(
            batch_size, padded_shapes=([None], [None], [None])
        ):
            outputs = model((enc_inputs, dec_inputs), training=False)
            loss = loss_fn(outputs, targets)

            valid_predictions += valid_dataloader.encoder_target.decode(outputs)

        train_path = os.path.join(directory, f"train-{epoch}")
        valid_path = os.path.join(directory, f"valid-{epoch}")

        write_text(train_predictions, train_path)
        write_text(valid_predictions, valid_path)

        train_bleu = compute_bleu(train_path, train_dataloader.file_name_target)
        valid_bleu = compute_bleu(valid_path, valid_dataloader.file_name_target)

        print(
            f"Epoch {epoch}: train bleu score: {train_bleu} valid bleu score: {valid_bleu}"
        )


def write_text(sentences, output_file):
    """Write text from sentences."""
    with open(output_file, "w+") as out_stream:
        for sentence in sentences:
            out_stream.write(sentence + "\n")


def compute_bleu(pred_file_path: str, target_file_path: str):
    """Compute bleu score.

    Args:
        pred_file_path: the file path that contains the predictions.
        target_file_path: the file path that contains the targets (also called references).

    Returns: Bleu score

    """
    out = subprocess.run(
        [
            "sacrebleu",
            "--input",
            pred_file_path,
            target_file_path,
            "--tokenize",
            "none",
            "--sentence-level",
            "--score-only",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines = out.stdout.split("\n")
    scores = [float(x) for x in lines[:-1]]
    return sum(scores) / len(scores)
