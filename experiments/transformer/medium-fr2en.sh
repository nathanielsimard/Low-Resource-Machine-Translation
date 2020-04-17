#!/bin/bash

python run_experiment.py \
    --model transformer-pretrained \
    --text_encoder subword \
    --vocab_size 8192 \
    --epochs 50 \
    --batch_size 32 \
    --max_seq_length 750 \
    --src_train data/splitted_data/train/train_token10000.fr \
    --target_train data/splitted_data/train/train_token10000.en \
    --src_valid data/splitted_data/valid/val_token10000.fr \
    --target_valid data/splitted_data/valid/val_token10000.en \
    --std \
    --lr None \
    --hyperparameters experiments/demi-bert/medium-hyperparameters.json \
    --pretrained data/splitted_french_data/target_train.fr \
    --no_cache \
    --task test \
    --checkpoint 42