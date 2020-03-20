#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M

module load python/3.7
source /project/cq-training-1/project1/teams/team10/env/bin/activate

# Sleep for 5 hours
sleep 18000
