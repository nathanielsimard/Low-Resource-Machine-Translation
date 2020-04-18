#!/bin/bash

# With the whole pretrained encoder vocab size 8192
# Best epoch 
# Train bleu : 
# Valid bleu : 
# Test bleu : 

python run_experiment.py \
    --model transformer-pretrained \
    --text_encoder subword \
    --vocab_size 8192 \
    --epochs 50 \
    --batch_size 32 \
    --max_seq_length 750 \
    --src_train data/src_backtranslation.en \
    --target_train data/target_backtranslation.fr \
    --src_valid data/splitted_data/valid/val_token10000.en \
    --target_valid data/splitted_data/valid/val_token10000.fr \
    --std \
    --lr None \
    --hyperparameters experiments/demi-bert/medium-hyperparameters.json \
    --pretrained data/splitted_english_data/sorted_clean_train.en \
    --no_cache 