#!/bin/bash

python run_experiment.py \
	--model demi-bert \
	--text_encoder subword \
	--vocab_size 8192 \
	--epochs 10 \
	--lr None \
	--batch_size 32 \
	--max_seq_length 750 \
	--src_train data/splitted_english_data/target_train.en \
	--src_valid data/splitted_english_data/target_valid.en \
	--hyperparameters experiments/demi-bert/medium-hyperparameters.json \
	--std \
    --task pretraining 