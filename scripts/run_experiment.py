import os
import subprocess
import yaml


# Open the config file for the experiment and load parameters
with open("./configs/config.yaml", "r") as f:
    experiment_config = yaml.load(f, Loader=yaml.FullLoader)

computer_id = experiment_config["computer_id"]
experiment_id = experiment_config["experiment_id"]

model = experiment_config["model"]
dataset = experiment_config["dataset"]
model_args = experiment_config["model_args"]
trainer_args = experiment_config["trainer_args"]

num_repetitions = experiment_config["num_repetitions"]
increment = experiment_config["increment"]
use_val_data = experiment_config["use_val_data"]
track_training = experiment_config["track_training"]
save_weights = experiment_config["save_weights"]

# Loop over seeds
for i in range(num_repetitions):
    # The path for the current repetition
    path = f"./experiments/{model}/{dataset}/{experiment_id}{computer_id}/repetition{i}"

    # Create a config with the current parameters
    config_dict = {
        "model_name": model,
        "dataset_name": dataset,
        "hyperparams": model_args,
        "trainer": trainer_args,
        "increment": increment,
        "cross_val": False,
        "use_val_data": use_val_data,
        "track_training": track_training,
        "save_weights": save_weights,
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

    if increment == "weights":
        trainer_args["weight_seed"] += 1
    elif increment == "batch":
        trainer_args["batch_seed"] += 1
    elif increment == "both":
        trainer_args["weight_seed"] += 1
        trainer_args["batch_seed"] += 1
