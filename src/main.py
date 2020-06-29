import yaml

from gluonts.trainer import Trainer

from cross_val import nested_cross_validation
from dataloader import DataLoader
from model_utils import create_model, evaluate_model


def run_experiment(config):
    dataloader = DataLoader(
        config["dataset_name"],
        config["dataset_path"],
        config["hyperparams"]["freq"],
        config["hyperparams"]["prediction_length"]
    )

    nested_cross_validation(
        dataloader.test_data,
        config["model_name"],
        config["hyperparams"],
        config["trainer"]
    )


if __name__ == "__main__":
    config_path = "./configs/test_config.yaml"

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    run_experiment(config)
