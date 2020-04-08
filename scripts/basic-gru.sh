#!/bin/bash

# hyperparameters = {
#	  "input_vocab_size": input_vocab_size + 1,
#	  "output_vocab_size": target_vocab_size + 1,
#	  "embedding_size": 256,
#	  "layers_size": 512,
#	  "dropout": 0.3,
#	  "attention_size": 8,
# }


python run_experiment.py \
	--model gru-attention \
	--text_encoder word-no-filter \
	--vocab_size 30000 \
	--epochs 100 \
	--batch_size 16 \
	--max_seq_length 500 \
	--src_train data/splitted_data/train/train_token10000.en \
	--target_train data/splitted_data/train/train_token10000.fr \
	--src_valid data/splitted_data/valid/val_token10000.en \
	--target_valid data/splitted_data/valid/val_token10000.fr \
	--no_cache
