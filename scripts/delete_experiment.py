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
parser.add_argument("experiment ID",
                    type=str,
                    help="The experiment ID")


if __name__ == "__main__":
    args = vars(parser.parse_args())
    experiment_path = f"./experiments/{args['model name']}/{args['dataset']}/{args['experiment ID']}"

    answer = input(f"Delete all files at {experiment_path}? [y/n] ")

    if answer is "n":
        exit(0)

    print(f"Deleting all folders and files at {experiment_path}...")
    shutil.rmtree(experiment_path)

    # If the dataset or model folders become empty, remove those aswell
    if len(os.listdir(f"./experiments/{args['model name']}/{args['dataset']}")) == 0:
        os.rmdir(f"./experiments/{args['model name']}/{args['dataset']}")
    
    if len(os.listdir(f"./experiments/{args['model name']}")) == 0:
        os.rmdir(f"./experiments/{args['model name']}")
        
