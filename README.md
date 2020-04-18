# Low Resource Machine Translation

## Run Evaluator

To evaluate the model on helios, the evaluator script must be used with the input and target samples.

```bash
python evaluator.py --input-file-path {input-file-path} --target-file-path {target-file-path}
```

For more information, see `python evaluator.py --help`.

## Experiments 

All experiments can be reproduced with the scripts in the folders `experiments`.
Custom experiments can be made with the help of the script `run_experiment.py`.
For more information, see `python run_experiment.py --help`.

### Artifacts

Artifacts are stored in different locations :

* Logs: `logging/{model-name}/{datetime}/experiment.log`,
* Weights: `models/{model-name}-{id}/{epoch}.*`,
* Valid Predictions: `results/{model}-{id}/{datetime}/valid-{epoch}`,
* Train Predictions: `results/{model}-{id}/{datetime}/train-{epoch}`,
* Training History (Learning Curves): `results/{model}-{id}/{datetime}/history-{epoch}`.

### Generate Graphs Example
```python generate_graphs.py --history_path='results/lstm_luong_attention/2020-04-01 21:58:21/history-17' output_path='results/lstm_luong_attention/2020-04-01 21:58:21/graphs' ```
