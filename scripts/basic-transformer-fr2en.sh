#!/bin/bash

# hyperparameters = {
#        "num_layers": 4,
#        "num_heads": 4,
#        "dff": 256,
#        "d_model": 256,
#        "input_vocab_size": input_vocab_size + 1,
#        "target_vocab_size": target_vocab_size + 1,
#        "pe_input": input_vocab_size + 1,
#        "pe_target": target_vocab_size + 1,
#        "rate": 0.1,
#    }

# Bleu Score after 34 epochs (Better epoch on 50 epochs):
#   - Train 38.40
#   - Valid 10.0162
#   - Test ??

python run_experiment.py \
	--model transformer \
	--text_encoder word-no-filter \
	--vocab_size 30000 \
	--epochs 50 \
	--lr None \
	--batch_size 64 \
	--max_seq_length 500 \
	--src_train data/splitted_data/train/train_token10000.fr \
	--target_train data/splitted_data/train/train_token10000.en \
	--src_valid data/splitted_data/valid/val_token10000.fr \
	--target_valid data/splitted_data/valid/val_token10000.en \
	--no_cache
