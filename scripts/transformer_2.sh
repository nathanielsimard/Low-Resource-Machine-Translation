#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M

module load python/3.7
source /project/cq-training-1/project2/teams/team10/env/bin/activate

python run_experiment.py --model=transformer-2 --task=default-training --epochs=3 --vocab_size=10000 --debug