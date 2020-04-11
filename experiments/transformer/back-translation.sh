#!/bin/bash

# Bleu Score after 34 epochs:

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
    --hyperparameters experiments/transformer/back-translation-hyperparameters.json \
	--no_cache
