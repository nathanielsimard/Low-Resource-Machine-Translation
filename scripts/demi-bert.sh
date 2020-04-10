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



python run_experiment.py \
	--model demi-bert \
	--text_encoder subword \
	--vocab_size 15000 \
	--epochs 10 \
	--lr None \
	--batch_size 64 \
	--max_seq_length 750 \
	--src_train data/splitted_english_data/sorted_clean_train.en \
	--src_valid data/splitted_english_data/sorted_clean_valid.en \
	--std \
    --task pretraining 
