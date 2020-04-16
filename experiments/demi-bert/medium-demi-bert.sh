#!/bin/bash

# Tried it with 15000 vocab size

python run_experiment.py \
	--model demi-bert \
	--text_encoder subword \
	--vocab_size 8192 \
	--epochs 10 \
	--lr None \
	--batch_size 32 \
	--max_seq_length 750 \
	--src_train data/splitted_french_data/target_train.fr \
	--src_valid data/splitted_french_data/target_valid.fr \
	--hyperparameters experiments/demi-bert/medium-hyperparameters.json \
	--std \
    --task pretraining 