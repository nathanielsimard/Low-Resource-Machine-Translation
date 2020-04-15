import re
import subprocess


def read_file(file_name: str):
    out = []
    with open(file_name, 'r') as stream:
        for line in stream:
            tokens = line.strip().split()
            out.append(tokens)
    return out


def read_file_np_lower(file_name):
    tokens = read_file(file_name)
    tokens = [' '.join(sentence).strip() for sentence in tokens]
    return [re.findall(r"[\w']+", text.lower()) for text in tokens]


def read_file_np(file_name):
    tokens = read_file(file_name)
    tokens = [' '.join(sentence).strip() for sentence in tokens]
    return [re.findall(r"[\w']+", text) for text in tokens]


def write_text(tokens, output_file):
    with open(output_file, 'w+') as out_stream:
        for token in tokens:
            out_stream.write(' '.join(token) + '\n')


def write_text_skip_line(tokens, output_file):
    with open(output_file, 'w+') as out_stream:
        for token in tokens:
            out_stream.write(' '.join(token) + '\n\n')


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
