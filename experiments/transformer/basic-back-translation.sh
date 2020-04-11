#!/bin/bash

# Bleu Score after 31 epochs:
#   - Train 37.107
#   - Valid 7.782
#   - Test ??

python run_experiment.py \
	--model transformer \
	--text_encoder word-no-filter \
	--vocab_size 30000 \
	--epochs 25 \
	--lr None \
	--batch_size 64 \
	--max_seq_length 500 \
	--src_train data/back-translation.en \
	--target_train data/back-translation.fr \
	--src_valid data/splitted_data/valid/val_token10000.en \
	--target_valid data/splitted_data/valid/val_token10000.fr \
    --hyperparameters experiments/transformer/basic-hyperparameters.json \
	--no_cache
