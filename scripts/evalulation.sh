#!/bin/bash

# Setup the environment
module load python/3.7
source /project/cq-training-1/project2/teams/team10/env/bin/activate

# Go to the right folder
cd /project/cq-training-1/project2/submissions/team10

# Run the evaluator with all the inputs
python evaluator.py $@

