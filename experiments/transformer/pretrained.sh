#!/bin/bash

# Best epoch: 18
# Train Bleu: ?
# Valid Bleu: 7.12 

python run_experiment.py \
--model transformer-pretrained \
--text_encoder subword \
 --vocab_size 15000 \
  --epochs 50 \
   --batch_size 32 \
    --max_seq_length 750 \
     --src_train data/splitted_data/train/train_token10000.en \
      --target_train data/splitted_data/train/train_token10000.fr \
       --src_valid data/splitted_data/valid/val_token10000.en \
        --target_valid data/splitted_data/valid/val_token10000.fr \
         --std \
         --lr None \
          --hyperparameters experiments/demi-bert/basic-hyperparameters.json \
          --pretrained