#!/bin/bash

# hyperparameters = {
#        "num_layers": 6,
#        "num_heads": 4,
#        "dff": 512,
#        "d_model": 512,
#        "input_vocab_size": input_vocab_size + 1,
#        "target_vocab_size": target_vocab_size + 1,
#        "pe_input": input_vocab_size + 1,
#        "pe_target": target_vocab_size + 1,
#        "rate": 0.1,
#    }

python run_experiment.py \
	--model transformer \
	--text_encoder word-no-filter \
	--vocab_size 30000 \
	--epochs 25 \
	--lr None \
	--batch_size 64 \
	--max_seq_length 500 \
    --task back-translation-training \
	--no_cache
