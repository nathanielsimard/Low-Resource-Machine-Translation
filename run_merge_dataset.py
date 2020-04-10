import argparse
import random

from src import dataloader
from src.training import base


def main():
    parser = argparse.ArgumentParser("script for evaluating a model.")
    parser.add_argument(
        "--src_dataset-1",
        help="Path to first source dataset to be merged",
        required=True,
    )
    parser.add_argument(
        "--target_dataset-1",
        help="Path to first target dataset to be merged",
        required=True,
    )
    parser.add_argument(
        "--src_dataset-2",
        help="path to second source dataset to be merged",
        required=True,
    )
    parser.add_argument(
        "--target_dataset-2",
        help="path to second target dataset to be merged",
        required=True,
    )
    parser.add_argument(
        "--src_dataset-out",
        help="Path where the new source dataset will be saved",
        required=True,
    )
    parser.add_argument(
        "--target_dataset-out",
        help="Path where the new target dataset will be saved",
        required=True,
    )

    args = parser.parse_args()

    src_dataset_1 = dataloader.read_file(args.src_dataset_1)
    src_dataset_2 = dataloader.read_file(args.src_dataset_2)

    target_dataset_1 = dataloader.read_file(args.target_dataset_1)
    target_dataset_2 = dataloader.read_file(args.target_dataset_2)

    src_dataset_out = src_dataset_1 + src_dataset_2
    target_dataset_out = target_dataset_1 + target_dataset_2

    dataset_out = list(zip(src_dataset_out, target_dataset_out))
    random.shuffle(dataset_out)
    src_dataset_out, target_dataset_out = zip(*dataset_out)  # type: ignore

    base.write_text(src_dataset_out, args.src_dataset_out)
    base.write_text(target_dataset_out, args.target_dataset_out)


if __name__ == "__main__":
    main()
