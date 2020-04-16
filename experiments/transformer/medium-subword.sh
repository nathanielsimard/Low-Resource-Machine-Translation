#!/bin/bash

# Bleu Score after 29 epochs with 15 000 vocab size:
#   - Train 13.94
#   - Valid 3.53
#   - Test 3.454

# Bleu Score after 13 epochs with 8192 vocab size:
#   - Train 4.926
#   - Valid 3.694
#   - Test 3.28

python run_experiment.py \
    --model transformer \
    --text_encoder subword \
    --vocab_size 15000 \
    --epochs 50 \
    --lr None \
    --batch_size 64 \
    --max_seq_length 500 \
    --src_train data/splitted_data/train/train_token10000.en \
    --target_train data/splitted_data/train/train_token10000.fr \
    --src_valid data/splitted_data/valid/val_token10000.en \
    --target_valid data/splitted_data/valid/val_token10000.fr \
    --hyperparameters experiments/transformer/medium-hyperparameters.json \
    --task test \
    --checkpoint 29 \
    --no_cache
