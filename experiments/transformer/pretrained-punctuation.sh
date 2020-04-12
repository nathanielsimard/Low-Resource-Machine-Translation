#!/bin/bash

python run_experiment.py \
--model transformer-pretrained \
--text_encoder subword \
 --vocab_size 15000 \
  --epochs 20 \
  --task punctuation-training \
   --batch_size 64 \
    --max_seq_length 750 \
     --src_train data/splitted_english_data/sorted_clean_train.en \
      --target_train data/splitted_english_data/sorted_target_train.en \
       --src_valid data/splitted_english_data/sorted_clean_valid.en \
        --target_valid data/splitted_english_data/sorted_target_valid.en \
         --lr None \
          --hyperparameters experiments/demi-bert/basic-hyperparameters.json 