#!/bin/bash

# With the whole pretrained encoder vocab size 8192
# Best epoch 50
# Train bleu : 83....
# Valid bleu : 9.38
# Test bleu : 9.14


python run_experiment.py \
    --model transformer-pretrained \
    --text_encoder subword \
    --vocab_size 8192 \
    --epochs 50 \
    --batch_size 64 \
    --max_seq_length 750 \
    --src_train data/splitted_data/train/train_token10000.en \
    --target_train data/splitted_data/train/train_token10000.fr \
    --src_valid data/splitted_data/valid/val_token10000.en \
    --target_valid data/splitted_data/valid/val_token10000.fr \
    --std \
    --lr None \
    --hyperparameters experiments/demi-bert/small-hyperparameters.json \
    --pretrained \
    --no_cache 