# importing libraries
import json
import yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mxnet as mx
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from mxnet import gluon

from utils import get_data

def create_models(name, trainer, **kwargs):
    model_name_dict = {
        "simple feed forward": SimpleFeedForwardEstimator,
        "deep ar": DeepAREstimator
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

    # load dataset
    train_ds, test_ds, metadata = get_data(config)

    # define a trainer
    trainer =  Trainer(**config["trainer"])

    # test the model
    estm = create_models("simple feed forward", trainer, **config["hyperparams"])
    prdc = train_model(estm, train_ds)
    evaluate(prdc, test_ds, 100)
