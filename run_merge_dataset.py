import argparse
import random

from src import dataloader
from src.training import base


def main():
    parser = argparse.ArgumentParser("script for evaluating a model.")
    parser.add_argument(
        "--dataset-1", help="Path to first dataset to be merged", required=True
    )
    parser.add_argument(
        "--dataset-2", help="path to second dataset to be merged", required=True
    )
    parser.add_argument(
        "--dataset-out", help="Path where the new dataset will be saved", required=True,
    )

    args = parser.parse_args()
    dataset_1 = dataloader.read_file(args.dataset_1)
    dataset_2 = dataloader.read_file(args.dataset_2)

    dataset_out = dataset_1 + dataset_2
    random.shuffle(dataset_out)

    base.write_text(dataset_out, args.dataset_out)


if __name__ == "__main__":
    main()
