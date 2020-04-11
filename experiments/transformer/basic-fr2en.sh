#!/bin/bash

# Bleu Score after 34 epochs (Better epoch on 50 epochs):
#   - Train 38.40
#   - Valid 10.0162
#   - Test ??

python run_experiment.py \
    --model transformer \
    --text_encoder word-no-filter \
    --vocab_size 30000 \
    --epochs 50 \
    --lr None \
    --batch_size 64 \
    --max_seq_length 500 \
    --src_train data/splitted_data/train/train_token10000.fr \
    --target_train data/splitted_data/train/train_token10000.en \
    --src_valid data/splitted_data/valid/val_token10000.fr \
    --target_valid data/splitted_data/valid/val_token10000.en \
    --hyperparameters experiments/transformer/basic-hyperparameters.json \
    --no_cache
