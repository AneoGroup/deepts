# importing libraries
import json
import yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mxnet as mx
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.trainer import Trainer
from mxnet import gluon


# make a dataset in correct foramt
def make_list_dataset(custom_dataset, freq, start_str, prediction_length):
    start = pd.Timestamp(start_str, freq=freq)  # can be different for each time series
    # train dataset: cut the last window of length "prediction_length", add "target" and "start" fields
    train_ds = ListDataset([{'target': x, 'start': start}
                            for x in custom_dataset[:, :-prediction_length]],
                        freq=freq)
    # test dataset: use the whole dataset, add "target" and "start" fields
    test_ds = ListDataset([{'target': x, 'start': start}
                        for x in custom_dataset],
                        freq=freq)
    return train_ds, test_ds

def create_models(name, trainer, **kwargs):
    model_name_dict = {
        "simple feed forward": SimpleFeedForwardEstimator
    }
    estimator = model_name_dict[name](trainer=trainer, **kwargs)
    return estimator


def train_model(estimator, train_ds):
    return estimator.train(train_ds)


def evaluate(predictor, test_ds, num_samples):
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,  # test dataset
        predictor=predictor,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation 
    )

    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(iter(ts_it), iter(forecast_it), num_series=len(test_ds))

    print(json.dumps(agg_metrics, indent=4))


if __name__ == "__main__":
    # open config
    with open("./configs/test_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    N = 10  # number of time series
    T = 100  # number of timesteps
    prediction_length = 24
    freq = "1H"
    custom_dataset = np.random.normal(size=(N, T))
    start_str = "01-01-2019"

    # convert to dataset
    train_ds, test_ds = make_list_dataset(custom_dataset, config["hyperparams"]["freq"], start_str, config["hyperparams"]["prediction_length"])

    # define a trainer
    trainer =  Trainer(**config["trainer"])

    # test the model
    estm = create_models("simple feed forward", trainer, **config["hyperparams"])
    prdc = train_model(estm, train_ds)
    evaluate(prdc, test_ds, 100)
