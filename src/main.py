import argparse
import yaml
import os
import shutil

from gluonts.trainer import Trainer
import mxnet as mx
import numpy as np

from dataloader import DataLoader
from cross_val import nested_cross_validation, single_experiment
from model_utils import create_model, evaluate_model


parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    required=True,
                    help="Path to a experiment config file"
                    )


def run_experiment(exp_name, config):
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
        single_experiment(
            exp_name,
            dataloader.train_data,
            dataloader.test_data,
            config['model_name'],
            config['hyperparams'],
            config['trainer']
        )


def prepare_folders(exp_name):
    if os.path.exists(f'results/{exp_name}') or os.path.exists(f'images/{exp_name}'):
        answer = input("The files for the experiment already exists. Do you want to overwrite? [y/n] ")
        if answer is 'n':
            exit(0)
    if not os.path.exists(f'results/{exp_name}'):
        # make results
        os.mkdir('results/' + exp_name)
    if not os.path.exists(f'images/{exp_name}'):
        # make images
        os.mkdir('images/' + exp_name)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    
    np.random.seed(0)
    mx.random.seed(0)

    with open(args['config'], "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    exp_name = config['model_name'] + '__' + config['dataset_name'] + '__' + str(config['exp_num'])
    prepare_folders(exp_name)

    run_experiment(exp_name, config)
