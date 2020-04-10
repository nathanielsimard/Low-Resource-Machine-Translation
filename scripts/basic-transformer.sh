#!/bin/bash

# hyperparameters = {
#        "num_layers": 6,
#        "embedding_size": 256,
#        "num_heads": 4,
#        "dff": 512,
#        "vocab_size": input_vocab_size + 1,
#        "max_pe": input_vocab_size + 1,
#        "dropout": 0.1,
#    }

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
	--no_cache \
	--std
