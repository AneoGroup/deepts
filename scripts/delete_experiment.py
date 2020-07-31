import argparse
import os
import shutil


parser = argparse.ArgumentParser(description="Deletes results and config-files of experiments")
parser.add_argument("model name",
                    type=str,
                    help="The model used in the experiment")
parser.add_argument("dataset",
                    type=str,
                    help="The dataset used in the experiment")
parser.add_argument("experiment number",
                    type=int,
                    help="The experiment number")


def delete_experiment_in_folder(experiment_prefix, directory):
    for d in os.listdir(directory):
        if d.startswith(experiment_prefix):
            if d.endswith(".yaml"):
                os.remove(directory + d)
            else:
                shutil.rmtree(directory + d)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    experiment_prefix = f"{args['model name']}__{args['dataset']}__{args['experiment number']}"

    answer = input(f"Delete all results, images and configs starting with prefix {experiment_prefix}? [y/n] ")

    if answer is "n":
        exit(0)

    print(f"Deleting all folders and files with prefix {experiment_prefix}...")
    delete_experiment_in_folder(experiment_prefix, "./results/")
    delete_experiment_in_folder(experiment_prefix, "./images/")
    delete_experiment_in_folder(experiment_prefix, "./configs/")
