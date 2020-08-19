import argparse
import os

import pandas as pd


parser = argparse.ArgumentParser(description="Deletes redundant samples in the forecasts.csv files of models that"
                                              "only produce point forecasts")
parser.add_argument("model name",
                    type=str,
                    help="The model where we want to delete all but one sample in every forecasts.csv file")


def modify_csv(path):
    df = pd.read_csv(path)
    df = df[df.columns[:5]]
    df.to_csv(path, index=False)  # We already have the row numbers as the index, so index=False


if __name__ == "__main__":
    args = vars(parser.parse_args())

    root_path = "experiments/" + args["model name"]
    answer = input(f"Delete all but the first samples for experiments at {root_path}? [y/n] ")

    if answer is "n":
        exit(0)

    # Loop over all datasets, experiment-IDs and repetitions
    for dataset in os.listdir(root_path):
        dataset_path = root_path + f"/{dataset}"
        for id in os.listdir(dataset_path):
            id_path = dataset_path + f"/{id}"
            for rep in os.listdir(id_path):
                rep_path = id_path + f"/{rep}"
                modify_csv(rep_path + "/forecasts.csv")
