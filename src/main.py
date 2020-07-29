import yaml
import os
import shutil

from gluonts.trainer import Trainer
import mxnet as mx
import numpy as np

from dataloader import DataLoader
from cross_val import nested_cross_validation, single_experiment
from model_utils import create_model, evaluate_model


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


def prepare_folders(config_path, exp_name):
    if os.path.exists(f'results/{exp_name}') or os.path.exists(f'images/{exp_name}'):
        answer = input("experiment is already exists. Want overwrite? [y/n] ")
        if answer is 'n':
            exit(0)
    if not os.path.exists(f'results/{exp_name}'):
        # make results
        os.mkdir('results/' + exp_name)
    if not os.path.exists(f'images/{exp_name}'):
        # make images
        os.mkdir('images/' + exp_name)
    
    # copy config
    shutil.copyfile(config_path,  './configs/' + exp_name + '.yaml')



if __name__ == "__main__":
    np.random.seed(0)
    mx.random.seed(0)

    config_path = "./configs/config.yaml"

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    exp_name = config['model_name'] + '__' + config['dataset_name'] + '__' + str(config['exp_num'])
    prepare_folders(config_path, exp_name)

    run_experiment(exp_name, config)
