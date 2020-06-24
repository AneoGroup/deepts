import yaml

import numpy as np
import pandas as pd


from copy import deepcopy

from model_uni import evaluate
from model_uni import create_models
from model_uni import train_model
from utils import get_data

from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.seq2seq import MQCNNEstimator
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer

# we get the longest time-series in the dataset to do the nestes cross validation
def get_longest_series(list_test):
    idx = 0
    max = 0
    return_list = deepcopy(list_test)
    for i in range(len(list_test.list_data)):
        # print(len(series['target']))
        if len(list_test.list_data[i]['target']) > max:
            max = len(list_test.list_data[i]['target'])
            idx = i
    return_list.list_data = [list_test.list_data[i]]
    print("\n\n this is it")
    print(return_list.list_data[0])
    return return_list

def nested_cross_validation(list_test, metadata):
    # values for the sliding windo
    block_length = config['hyperparams']['context_length']
    prediction_length = config['hyperparams']['prediction_length']

    # get the longest timeseries
    max_len_timeseries = get_longest_series(list_test)

    # get the maximum time series in a format of dataset
    target_test = deepcopy(max_len_timeseries)

    # for loop through all the test set
    for i in range(0, len(target_test.list_data[0]['target'])-prediction_length, block_length):
        config['hyperparams']['context_length'] += i
        # make the train part
        train_part = deepcopy(target_test)
        train_part.list_data[0]['target'] = train_part.list_data[0]['target'][0 : i + block_length]

        # make the validate part
        validate_part = deepcopy(target_test)
        validate_part.list_data[0]['target'] = validate_part.list_data[0]['target'][i + block_length : prediction_length + i + block_length]

        if len(validate_part.list_data[0]['target']) < prediction_length:
            break

        print("Training on " + str(i + block_length) + " samples")
        # define a trainer
        trainer =  Trainer(**config["trainer"])

        # test the model
        estm = create_models(config["model"], trainer, **config["hyperparams"])
        prdc = train_model(estm, train_ds)
        evaluate(prdc, validate_part, 100)



if __name__ == "__main__":
    # from the config get the information
    # open config
    with open("./configs/test_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # get the dataset
    train_ds, test_ds, metadata = get_data(config)
    # apply nested cross validation
    nested_cross_validation(test_ds, metadata)