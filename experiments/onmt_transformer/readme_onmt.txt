In order to setup the python environment required for evaluation and training,
please run the following script:
bash Low-Resource-Machine-Translation/helios_setup.sh

it will create the onmt_helios python environment in your home directory.

In order to run the evaluation or training script:
(From the submission folder, go to the cloned repo folder):
cd Low-Resource-Machine-Translation

Then, run the evalution script:
sbatch scripts/evaluation_onmt.sh --input-file-path=data/splitted_data/test/test_token10000.en --target-file-path=data/splitted_data/test/test_token10000.fr

For the training script
sbatch scripts/training_onmt.sh

Both script will source the onmt_helios python environment from your home directory when starting.





