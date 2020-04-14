#!/bin/bash


python run_experiment.py \
	--model demi-bert \
	--text_encoder subword \
	--vocab_size 15000 \
	--epochs 10 \
	--lr None \
	--batch_size 16 \
	--max_seq_length 750 \
	--src_train data/splitted_english_data/sorted_clean_train.en \
	--src_valid data/splitted_english_data/sorted_clean_valid.en \
	--std \
    --task pretraining \
	--hyperparameters experiments/demi-bert/large-hyperparameters.json