import argparse
import subprocess
import tempfile
from typing import List

import models
from src import dataloader, logging
from src.text_encoder import TextEncoderType
from src.training import base

logger = logging.create_logger(__name__)


class French2EnglishSettings(object):
    text_encoder = TextEncoderType.WORD_NO_FILTER
    vocab_size = 30000
    src_train = "data/splitted_data/train/train_token10000.fr"
    target_train = "data/splitted_data/train/train_token10000.en"
    checkpoint = 34
    max_seq_length = 750
    batch_size = 64
    model = "transformer"
    hyperparameters = "experiments/transformer/basic-hyperparameters.json"


def generate_predictions(input_file_path: str, pred_file_path: str):
    """Generates predictions for the machine translation task (EN->FR).

    You are allowed to modify this function as needed, but one again, you cannot
    modify any other part of this file. We will be importing only this function
    in our final evaluation script. Since you will most definitely need to import
    modules for your code, you must import these inside the function itself.

    Args:
        input_file_path: the file path that contains the input data.
        pred_file_path: the file path where to store the predictions.

    Returns: None

    """
    logger.info(f"Generate predictions with input {input_file_path} {pred_file_path}")
    settings = French2EnglishSettings()
    train_dl = dataloader.AlignedDataloader(
        file_name_input=settings.src_train,
        file_name_target=settings.target_train,
        vocab_size=settings.vocab_size,
        text_encoder_type=settings.text_encoder,
    )
    encoder_input = train_dl.encoder_input
    encoder_target = train_dl.encoder_target

    # Load the model.
    model = models.find(settings, encoder_input.vocab_size, encoder_target.vocab_size)
    model.load(str(settings.checkpoint))

    dl = dataloader.UnalignedDataloader(
        file_name=input_file_path,
        vocab_size=settings.vocab_size,
        text_encoder_type=settings.text_encoder,
        max_seq_length=settings.max_seq_length,
        cache_dir=None,
        encoder=encoder_input,
    )

    predictions = _generate_predictions(
        model, dl, encoder_input, encoder_target, settings.batch_size
    )
    base.write_text(predictions, "allo.txt")


def _generate_predictions(
    model, dataloader, encoder_input, encoder_target, batch_size,
):
    dataset = dataloader.create_dataset()
    predictions: List[str] = []
    for i, inputs in enumerate(dataset.padded_batch(batch_size, padded_shapes=[None])):
        logger.info(f"Batch #{i} : evaluator")
        outputs = model.translate(inputs, encoder_input, encoder_target)
        predictions += model.predictions(outputs, encoder_target, logit=False)

    return predictions


def compute_bleu(pred_file_path: str, target_file_path: str, print_all_scores: bool):
    """

    Args:
        pred_file_path: the file path that contains the predictions.
        target_file_path: the file path that contains the targets (also called references).
        print_all_scores: if True, will print one score per example.

    Returns: None

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
    if print_all_scores:
        print("\n".join(lines[:-1]))
    else:
        scores = [float(x) for x in lines[:-1]]
        print("final avg bleu score: {:.2f}".format(sum(scores) / len(scores)))


def main():
    parser = argparse.ArgumentParser("script for evaluating a model.")
    parser.add_argument(
        "--target-file-path", help="path to target (reference) file", required=True
    )
    parser.add_argument("--input-file-path", help="path to input file", required=True)
    parser.add_argument(
        "--print-all-scores",
        help="will print one score per sentence",
        action="store_true",
    )
    parser.add_argument(
        "--do-not-run-model",
        help="will use --input-file-path as predictions, instead of running the "
        "model on it",
        action="store_true",
    )

    args = parser.parse_args()

    if args.do_not_run_model:
        compute_bleu(args.input_file_path, args.target_file_path, args.print_all_scores)
    else:
        _, pred_file_path = tempfile.mkstemp()
        generate_predictions(args.input_file_path, pred_file_path)
        compute_bleu(pred_file_path, args.target_file_path, args.print_all_scores)


if __name__ == "__main__":
    main()
