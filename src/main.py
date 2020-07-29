import argparse
import os
import shutil
import yaml

from gluonts.trainer import Trainer
import mxnet as mx
import numpy as np

from cross_val import nested_cross_validation
from dataloader import DataLoader
from model_utils import create_model, evaluate_model
from result_utils import plot_forecast, write_results


parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    required=True,
                    help="Path to a experiment config file"
                    )


def run_experiment(exp_name: str, config: dict):
    dataloader = DataLoader(
        config["dataset_name"],
        config["dataset_path"],
        config["hyperparams"]["freq"],
        config["hyperparams"]["prediction_length"]
    )

    if config["hyperparams"]["freq"] is None:
        config["hyperparams"]["freq"] = dataloader.freq
    if config["hyperparams"]["prediction_length"] is None:
        config["hyperparams"]["prediction_length"] = dataloader.prediction_length

    if config['cross_val']:
        nested_cross_validation(
            exp_name,
            dataloader.test_data,
            config["model_name"],
            config["hyperparams"],
            config["trainer"]
        )
    else:
        # Run a single experiment
        # Define a trainer
        trainer = Trainer(**config["trainer"])

        # Create, train and test the model
        estimator = create_model(
            config["model_name"],
            trainer,
            **config["hyperparams"]
        )
        predictor = estimator.train(dataloader.train_data)
        forecasts, targets, metrics = evaluate_model(predictor, dataloader.test_data, 100)

        # Save results
        write_results(forecasts, targets, metrics, config["hyperparams"]["prediction_length"], exp_name)
        plot_forecast(targets, forecasts, f"{exp_name}/plot")


if __name__ == "__main__":
    args = vars(parser.parse_args())
    
    np.random.seed(0)
    mx.random.seed(0)

    with open(args['config'], "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    exp_name = config['model_name'] + '__' + config['dataset_name'] + '__' + str(config['exp_num'])
    run_experiment(exp_name, config)
