#!/bin/bash


python run_experiment.py \
	--model gru-attention \
	--text_encoder word-no-filter \
	--vocab_size 30000 \
	--epochs 100 \
	--src_train data/splitted_data/train/train_token10000.en \
	--target_train data/splitted_data/train/train_token10000.fr \
	--src_valid data/splitted_data/valid/val_token10000.en \
	--target_valid data/splitted_data/valid/val_token10000.fr \
