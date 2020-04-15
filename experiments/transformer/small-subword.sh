#!/bin/bash

# Bleu Score after 38 epochs:
#   - Train 32.5439
#   - Valid 5.3417
#   - Test 5.2406

python run_experiment.py \
    --model transformer \
    --text_encoder subword \
    --vocab_size 8192 \
    --epochs 50 \
    --lr None \
    --batch_size 64 \
    --max_seq_length 500 \
    --src_train data/splitted_data/train/train_token10000.en \
    --target_train data/splitted_data/train/train_token10000.fr \
    --src_valid data/splitted_data/valid/val_token10000.en \
    --target_valid data/splitted_data/valid/val_token10000.fr \
    --hyperparameters experiments/transformer/small-hyperparameters.json \
    --no_cache
