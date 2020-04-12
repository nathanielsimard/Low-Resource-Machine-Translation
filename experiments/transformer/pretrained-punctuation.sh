#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M

module load python/3.7
source /project/cq-training-1/project2/teams/team10/env/bin/activate

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