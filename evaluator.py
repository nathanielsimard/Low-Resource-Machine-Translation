import argparse
import subprocess
import tempfile


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
    combined = True  # Winning model has combined vocabulary
    from opennmt_transformer import (
        init_model,
        translate,
        get_vocab_file_names,
        init_data_config,
        init_checkpoint_manager_and_load_latest_checkpoint,
    )
    from src.opennmt_preprocessing import prepare_bpe_files, decode_bpe_file
    import shutil
    import tempfile
    import os

    input_file_path = os.path.expanduser(input_file_path)
    bpe_src, _ = prepare_bpe_files(input_file_path, None, combined=combined)

    model, checkpoint, optimizer, learning_rate = init_model()
    checkpoint_manager = init_checkpoint_manager_and_load_latest_checkpoint(checkpoint)
    src_vocab, tgt_vocab = get_vocab_file_names()
    init_data_config(model, src_vocab, tgt_vocab)
    with tempfile.NamedTemporaryFile() as f:
        TMP_OUTPUTS = f.name

    print(f"Writing non BPE-decoded outputs to {TMP_OUTPUTS}")
    translate(model, bpe_src, output_file=TMP_OUTPUTS, show_progress=True)
    print(f"Decoding {TMP_OUTPUTS}")
    bpe_decoded_file = decode_bpe_file(TMP_OUTPUTS, combined=combined)
    print(
        f"Copying decoded file {bpe_decoded_file} to the final expected path {pred_file_path}"
    )
    shutil.copy(bpe_decoded_file, pred_file_path)
    # Cleanup. Some exception is thrown if cleanup is not done manually for some reason.
    del checkpoint_manager, model, checkpoint, optimizer, learning_rate
    return


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
