#!/bin/bash

# Bleu Score after 12 epochs on 30+:
#   - Train 5.535
#   - Valid 3.79
#   - Test ??

python run_experiment.py \
    --model transformer \
    --text_encoder word-no-filter \
    --vocab_size 30000 \
    --epochs 50 \
    --lr None \
    --batch_size 32 \
    --max_seq_length 512 \
    --src_train data/splitted_data/train/train_token10000.en \
    --target_train data/splitted_data/train/train_token10000.fr \
    --src_valid data/splitted_data/valid/val_token10000.en \
    --target_valid data/splitted_data/valid/val_token10000.fr \
    --hyperparameters experiments/transformer/openNMT-default-hyperparameters.json \
    --no_cache
