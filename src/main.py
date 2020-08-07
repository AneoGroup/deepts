import argparse
import os
import random
import shutil
import yaml

from gluonts.trainer import Trainer
import mxnet as mx
import numpy as np

from cross_val import nested_cross_validation
from dataloader import DataLoader
from model_utils import create_model, evaluate_model, evaluate_n_times
from result_utils import plot_forecast, write_results


parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    required=True,
                    help="Path to a experiment config file"
                    )
parser.add_argument("--num-tests",
                    type=int,
                    default=0,
                    required=False,
                    help="The number of times to test the model using diferent seeds")
parser.add_argument("--test-seed",
                    type=int,
                    default=0,
                    required=False,
                    help="If set, this is used as the initial seed to sample other seeds for testing")


def run_experiment(exp_path: str, config: dict, num_tests: int = 0, test_seed: int = 0):
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
            exp_path,
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
        if not os.path.exists(f"{exp_path}/samples"):
            os.mkdir(f"{exp_path}/samples")

        write_results(forecasts, targets, metrics, config["hyperparams"]["prediction_length"], f"{exp_path}/")
        plot_forecast(targets, forecasts, f"{exp_path}/plot.png")

        # Evaluate the model multiple times with different seeds
        evaluate_n_times(
            predictor,
            dataloader,
            config["hyperparams"]["prediction_length"],
            num_tests,
            test_seed,
            exp_path
        )


if __name__ == "__main__":
    args = vars(parser.parse_args())

    with open(args['config'], "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if config["random_seed"] is not None:
        np.random.seed(config["random_seed"])
        mx.random.seed(config["random_seed"])

    exp_path = config["path"]
    run_experiment(exp_path, config, args["num_tests"], args["test_seed"])
