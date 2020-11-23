import argparse
import os
import pathlib
import yaml

from gluonts.trainer import Trainer
import mxnet as mx
import numpy as np

from dataloader import DataLoader
from model_utils import create_model, evaluate_model, evaluate_n_times
from result_utils import plot_forecast, write_results
from trainer import TrackingTrainer


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
        config["use_val_data"],
        config["hyperparams"]["freq"],
        config["hyperparams"]["prediction_length"]
    )

    if config["hyperparams"]["freq"] is None:
        config["hyperparams"]["freq"] = dataloader.freq
    if config["hyperparams"]["prediction_length"] is None:
        config["hyperparams"]["prediction_length"] = dataloader.prediction_length

    # Run a single experiment
    # Define a trainer
    if not config.get("track_training"):
        trainer = Trainer(**config["trainer"])
    else:
        trainer = TrackingTrainer(
            config["model_name"],
            config["dataset_name"],
            **config["trainer"])

    # Create, train and test the model
    estimator = create_model(
        config["model_name"],
        trainer,
        **config["hyperparams"]
    )
    
    predictor = estimator.train(
        dataloader.train_data,
        validation_data=dataloader.val_data if config["use_val_data"] else None
    )
    forecasts, targets, metrics = evaluate_model(predictor, dataloader.test_data, 100)

    # Save weights
    if config.get("save_weights"):
        predictor.serialize_prediction_net(pathlib.Path(exp_path))
        os.remove(f"{exp_path}/prediction_net-network.json")
    
    # Store results
    write_results(forecasts, targets, metrics, config["hyperparams"]["prediction_length"], f"{exp_path}/")
    plot_forecast(targets, forecasts, f"{exp_path}/plot.png")

    # Sample multiple times with the same seed if we specify it.
    if num_tests > 0:
        # Create a folder for aditional tests
        if not os.path.exists(f"{exp_path}/samples"):
            os.mkdir(f"{exp_path}/samples")

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

    exp_path = config["path"]
    run_experiment(exp_path, config, args["num_tests"], args["test_seed"])
