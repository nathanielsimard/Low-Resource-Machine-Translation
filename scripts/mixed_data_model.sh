#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M

module load python/3.7
source /project/cq-training-1/project2/teams/team10/env/bin/activate

python run_experiment.py --model transformer --vocab_size 15000 --text_encoder subword --src_train data/mixed_data/mixed_train_data --target_train data/mixed_data/mixed_target_data --src_valid data/splitted_data/valid/val_token10000.en --target_valid data/splitted_data/valid/val_token.en