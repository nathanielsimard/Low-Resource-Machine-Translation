#!/bin/bash

# After 18 epochs:
#   - Train : 24.1215
#   - Valid : 2.3812
#   - Test : 2.6832
#
# Note: Does not predict correctly end of sentence tokens and
#       enters an endless loop of repreating the same
#       part of the sentence.

python run_experiment.py \
    --model lstm_luong_attention \
    --text_encoder word-no-filter \
    --vocab_size 30000 \
    --epochs 18 \
    --batch_size 16 \
    --max_seq_length 500 \
    --src_train data/splitted_data/train/train_token10000.en \
    --target_train data/splitted_data/train/train_token10000.fr \
    --src_valid data/splitted_data/valid/val_token10000.en \
    --target_valid data/splitted_data/valid/val_token10000.fr \
    --hyperparameters experiments/lstm_luong_attention/basic-hyperparameters.json \
    --no_cache
