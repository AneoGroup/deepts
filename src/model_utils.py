import random

import mxnet as mx
import numpy as np
import pandas as pd

from gluonts.dataset.common import ListDataset
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deep_factor import DeepFactorEstimator
from gluonts.model.deepstate import DeepStateEstimator
from gluonts.model.estimator import Estimator
from gluonts.model.forecast import SampleForecast
from gluonts.model.n_beats import NBEATSEstimator
from gluonts.model.seq2seq import MQCNNEstimator, MQRNNEstimator, RNN2QRForecaster
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.wavenet import WaveNetEstimator
from gluonts.trainer import Trainer

from dataloader import DataLoader
from lstm.lstm_estimator import DeepARLSTMEstimator
from result_utils import write_results


model_dict = {
    "deep_ar": DeepAREstimator,
    "deep_factor": DeepFactorEstimator,
    "deepstate": DeepStateEstimator,
    "lstm": DeepARLSTMEstimator,
    "n_beats": NBEATSEstimator,
    "mqcnn": MQCNNEstimator,
    "mqrnn": MQRNNEstimator,
    "rnn2qr": RNN2QRForecaster,
    "simple_feed_forward": SimpleFeedForwardEstimator,
    "wavenet": WaveNetEstimator,
}


def create_model(name: str,
                 trainer: Trainer,
                 **kwargs) -> Estimator:
    estimator = model_dict[name](
        trainer=trainer,
        **kwargs
    )
    return estimator


def evaluate_model(predictor: Estimator,
                   test_ds: ListDataset,
                   num_samples: int) -> (SampleForecast, pd.DataFrame, pd.DataFrame):
    forecast_it, ts_it = make_evaluation_predictions(
        predictor=predictor,
        dataset=test_ds,
        num_samples=100
    )

    forecasts = list(forecast_it)
    ts = list(ts_it)

    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    _, item_metrics = evaluator(iter(ts), iter(forecasts))

    return forecasts, ts, item_metrics


def evaluate_n_times(predictor: Estimator,
                     dataloader: DataLoader,
                     prediction_length: int,
                     num_tests: int,
                     initial_seed: int,
                     experiment_path: str) -> None:
    random.seed(initial_seed)
    for i in range(num_tests):
        seed = random.randint(0, 4294967295)
        np.random.seed(seed)
        mx.random.seed(seed)

        print(f"\nRunning evaluation number {i + 1} with seed {seed}")

        forecasts, targets, metrics = evaluate_model(predictor, dataloader.test_data, 100)
        write_results(forecasts, targets, metrics, prediction_length, f"{experiment_path}/samples/seed_{seed}_")
