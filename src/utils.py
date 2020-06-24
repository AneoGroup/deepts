import numpy as np
import pandas as pd

from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.common import ListDataset


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


def load_te_data():
    raise NotImplementedError


def load_random_data(config):
    # Define number and length of time series
    N = 100
    T = 1000

    # Define starting point
    start = "01-01-2000"

    # Generate random data
    random_data = np.random.normal(size=(N, T))

    # Convert to ListDataset
    freq = config["hyperparams"]["freq"]
    prediction_length = config["hyperparams"]["prediction_length"]

    train_ds, test_ds = make_list_dataset(random_data, freq, start, prediction_length)
    return train_ds, test_ds, {"freq": freq, "start": start, "prediction_length": prediction_length}


def convert_file_to_list(data_file, freq):
    list_data = list(iter(data_file))
    list_data = ListDataset(list_data, freq = freq)
    return list_data


def get_data(config):
    if config["data"] == "tronderenergi":
        return load_te_data()
    elif config["data"] == "generate":
        return load_random_data(config)
    elif config["data"] in list(dataset_recipes.keys()):
        dataset = get_dataset(config["data"], regenerate=False)
        # convert the dataset files to dataset lists
        train_ds = convert_file_to_list(dataset.train, dataset.metadata.freq)
        test_ds  = convert_file_to_list(dataset.test, dataset.metadata.freq)
        return train_ds, test_ds, dataset.metadata
    else:
        raise ValueError(f"Invalid dataset name: {config['data']}")
