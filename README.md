# On Reporting Robust and Trustworthy Conclusions from Model Comparison Studies Involving Neural Networks and Randomness
The code used to generate results used in the paper On Reporting Robust and Trustworthy Conclusions from Model Comparison Studies Involving Neural Networks and Randomness.


## Running the code
Dependencies are specified in `environment.yml`.

`src/main.py` trains and tests a single model.

To run multiple experiments with different seeds, specify a config file in `configs/config.yaml`, then run `python scripts/run_experiment.py`

To train using a single seed, then test with multiple seeds, specify a config file for a model then run `python src/main.py --config PATH_TO_CONFIG --num-tests NUMBER_OF_TIMES_TO_TEST`

To run the TensorFlow LSTM implementation, run `python src/lstm/tf_lstm.py`. To change arguments (dataset, etc.) you have to change the main method in the same file.
