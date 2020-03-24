import abc
import os
import subprocess
from datetime import datetime
from typing import List

import tensorflow as tf

from src.dataloader import AlignedDataloader
from src.model import base


class Training(abc.ABC):
    @abc.abstractmethod
    def run(
        self,
        loss_fn: tf.keras.losses,
        optimizer: tf.keras.optimizers,
        batch_size: int,
        num_epoch: int,
        checkpoint=None,
    ):
        pass


class BasicMachineTranslationTraining(Training):
    def __init__(
        self,
        model,
        train_dataloader: AlignedDataloader,
        valid_dataloader: AlignedDataloader,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

    def run(
        self,
        loss_fn: tf.keras.losses,
        optimizer: tf.keras.optimizers,
        batch_size: int,
        num_epoch: int,
        checkpoint=None,
    ):
        """Training session."""
        train_dataset = self.train_dataloader.create_dataset()
        valid_dataset = self.valid_dataloader.create_dataset()

        train_dataset = self.model.preprocessing(train_dataset)
        valid_dataset = self.model.preprocessing(valid_dataset)

        directory = os.path.join(
            "results", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        os.makedirs(directory, exist_ok=True)

        if checkpoint is not None:
            self.model.load(str(checkpoint))
        else:
            checkpoint = 0

        print(checkpoint)
        for epoch in range(checkpoint + 1, num_epoch + 1):
            train_predictions: List[str] = []

            for inputs, targets in train_dataset.padded_batch(
                batch_size, padded_shapes=self.model.padded_shapes
            ):
                with tf.GradientTape() as tape:
                    outputs = self.model(inputs, training=True)

                    # Calculate the training prediction tokens
                    predictions = self.model.predictions(
                        outputs, self.train_dataloader.encoder_target
                    )
                    train_predictions += predictions

                    # Calculate the loss and update the parameters
                    loss = loss_fn(targets, outputs)
                    print(f"Training loss {loss}")

                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    optimizer.apply_gradients(
                        zip(gradients, self.model.trainable_variables)
                    )

            valid_predictions = _generate_predictions(
                self.model,
                valid_dataset,
                self.valid_dataloader.encoder_target,
                batch_size,
            )

            train_path = os.path.join(directory, f"train-{epoch}")
            valid_path = os.path.join(directory, f"valid-{epoch}")

            write_text(train_predictions, train_path)
            write_text(valid_predictions, valid_path)

            train_bleu = compute_bleu(
                train_path, self.train_dataloader.file_name_target
            )
            valid_bleu = compute_bleu(
                valid_path, self.valid_dataloader.file_name_target
            )

            self.model.save(epoch)
            print(
                f"Epoch {epoch}: train bleu score: {train_bleu} valid bleu score: {valid_bleu}"
            )


def test(
    model: base.Model,
    loss_fn,
    dataloader: AlignedDataloader,
    batch_size: int,
    checkpoint: int,
):
    """Test a model at a specific checkpoint."""
    dataset = dataloader.create_dataset()
    dataset = model.preprocessing(dataset)
    model.load(str(checkpoint))

    predictions = _generate_predictions(
        model, dataset, dataloader.encoder_target, batch_size
    )

    directory = os.path.join("results/test", model.title)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"test-{checkpoint}")
    write_text(predictions, path)

    bleu = compute_bleu(path, dataloader.file_name_target)
    print(f"Bleu score {bleu}")


def _generate_predictions(model, dataset, encoder, batch_size):
    predictions = []
    for inputs, targets in dataset.padded_batch(
        batch_size, padded_shapes=model.padded_shapes
    ):
        outputs = model(inputs, training=False)
        predictions += model.predictions(outputs, encoder)

    return predictions


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
