import os
import subprocess
import yaml


# Open the config file for the experiment and load parameters
with open("./configs/config.yaml", "r") as f:
    experiment_config = yaml.load(f, Loader=yaml.FullLoader)

computer_id = experiment_config["computer_id"]
experiment_id = experiment_config["experiment_id"]
models = experiment_config["models"]
datasets = experiment_config["datasets"]
ctx = experiment_config["context"]
epochs =  experiment_config["epochs"]
learning_rates = experiment_config["learning_rates"]
seed = experiment_config["random_seed"]
num_repetitions = experiment_config["num_repetitions"]
track_training = experiment_config["track_training"]

# Set increment_seed = True if we want repetitions with different seeds
increment_seed = False
if seed == "increment":
    increment_seed = True
    seed = 0

# Loop over parameters
for model in models:
    for dataset in datasets:
        exp_num = experiment_id
        for epoch in epochs:
            for lr in learning_rates:
                for i in range(num_repetitions):
                    # The path for the current repetition
                    path = f"./experiments/{model}/{dataset}/{exp_num}{computer_id}/repetition{i}"

                    # Create a config with the current parameters
                    config_dict = {
                        "model_name": model,
                        "dataset_name": dataset,
                        "hyperparams": {
                            "freq": None,
                            "context_length": None,
                            "prediction_length": None,
                        },
                        "trainer": {
                            "ctx": ctx,
                            "epochs": epoch,
                            "learning_rate": lr
                        },
                        "random_seed": seed,
                        "cross_val": False,
                        "dataset_path": None,
                        "track_training": track_training,
                        "path": path
                    }

                    # Ask before overwriting old experiments
                    if os.path.exists(path):
                        overwrite = input(f"A folder at this path ({path}) already exists. "
                                          "Do you want to overwrite? [y/n] ")
                        if overwrite == "n":
                            exit(0)

                    if not os.path.exists(path):
                        os.makedirs(path)

                    with open(path + "/config.yaml", "w") as f:
                            yaml.dump(config_dict, f)
                    
                    # Run experiment
                    subprocess.run(["python", "src/main.py",
                                    "--config", path + "/config.yaml"])

                    if increment_seed:
                        seed += 1
                
                exp_num += 1
