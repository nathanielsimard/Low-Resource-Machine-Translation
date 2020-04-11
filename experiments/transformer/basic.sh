#!/bin/bash

# Bleu Score after 25 epochs:
#   - Train 30.117
#   - Valid 6.674
#   - Test 6.816

python run_experiment.py \
	--model transformer \
	--text_encoder word-no-filter \
	--vocab_size 30000 \
	--epochs 25 \
	--lr None \
	--batch_size 64 \
	--max_seq_length 500 \
	--src_train data/splitted_data/train/train_token10000.en \
	--target_train data/splitted_data/train/train_token10000.fr \
	--src_valid data/splitted_data/valid/val_token10000.en \
	--target_valid data/splitted_data/valid/val_token10000.fr \
    --hyperparameters experiments/transformer/hyperparameters.json \
	--no_cache
