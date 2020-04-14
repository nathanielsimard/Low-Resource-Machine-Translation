#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M

# Setup the environment
module load python/3.7
source ~/onmt_helios/bin/activate

# Run the evaluator with all the inputs
python evaluator.py $@

