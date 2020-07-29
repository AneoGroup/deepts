import os
import yaml
import subprocess


# Open the config file for the experiment and load parameters
with open("./configs/config.yaml", "r") as f:
    experiment_config = yaml.load(f, Loader=yaml.FullLoader)

models = experiment_config["models"]
datasets = experiment_config["datasets"]
epochs =  experiment_config["epochs"]
learning_rates = experiment_config["learning_rates"]
num_repeats = experiment_config["num_repeats"]
experiment_id = experiment_config["experiment_id"]

# Loop over parameters
for model in models:
    for dataset in datasets:
        exp_number = experiment_id
        for epoch in epochs:
            for lr in learning_rates:
                # create a config with the current parameters
                config_dict = {
                    "model_name": model,
                    "dataset_name": dataset,
                    "hyperparams": {
                        "freq": None,
                        "context_length": None,
                        "prediction_length": None
                    },
                    "trainer": {
                        "ctx": "gpu",
                        "epochs": epoch,
                        "learning_rate": lr
                    },
                    "cross_val": False,
                    "dataset_path": None
                }

                for i in range(num_repeats):
                    # save the config of this experiment
                    repetition_id = f"{exp_number}__{i}"
                    config_dict["exp_num"] = repetition_id
                    experiment_path = f"./configs/{model}__{dataset}__{repetition_id}.yaml"

                    if os.path.exists(experiment_path):
                        raise ValueError("Experiment config already exists ({experiment_path})")
                    else:
                        with open(experiment_path, "w") as f:
                            yaml.dump(config_dict, f)
                    
                    # run experiment
                    subprocess.run(["python", "src/main.py",
                                    "--config", experiment_path])
                
                exp_number += 1
