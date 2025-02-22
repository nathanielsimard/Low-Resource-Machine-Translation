# Low Resource Machine Translation

## Run Evaluator

To evaluate the model on helios, the evaluator script must be used with the input and target samples.

```bash
python evaluator.py --input-file-path {input-file-path} --target-file-path {target-file-path}
```

For more information, see `python evaluator.py --help`.

Note that the dependencies are in `requirements.txt`.
They can be install with `pip`.

```bash
pip install -r requirements.txt
```

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

### Replacing a mask token in a sentence
Testing a masked language model can be done using the run_test_mlm.py script.
Example: ```python run_test_mlm.py --checkpoint 2 --message "father help <mask>me pick apples" ``` 
Predicted token: `ed`