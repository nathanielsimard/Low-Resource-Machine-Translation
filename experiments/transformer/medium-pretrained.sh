#!/bin/bash

# With the whole pretrained encoder and vocab size of 15000
# Best epoch: 18
# Train Bleu: ?
# Valid Bleu: 7.12(old)
# Test Bleu: 9.58 

# With the whole pretrained encoder vocab size 8192
# Best epoch 22
# Train bleu : 38.86
# Valid bleu : 9.63
# Test bleu : 9.50

# With just the embedding layer

python run_experiment.py \
    --model transformer-pretrained \
    --text_encoder subword \
    --vocab_size 8192 \
    --epochs 50 \
    --batch_size 32 \
    --max_seq_length 750 \
    --src_train data/splitted_data/train/train_token10000.en \
    --target_train data/splitted_data/train/train_token10000.fr \
    --src_valid data/splitted_data/valid/val_token10000.en \
    --target_valid data/splitted_data/valid/val_token10000.fr \
    --std \
    --lr None \
    --hyperparameters experiments/demi-bert/medium-hyperparameters.json \
    --pretrained \
    --no_cache 