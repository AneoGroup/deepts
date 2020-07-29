import os
import subprocess
import yaml


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
                # Areate a config with the current parameters
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
                    # Save the config of this experiment
                    repetition_id = f"{exp_number}__{i}"
                    config_dict["exp_num"] = repetition_id
                    experiment_name = f"{model}__{dataset}__{repetition_id}.yaml"
                    config_path = f"./configs/{experiment_name}.yaml"

                    # Ask before overwriting old experiments
                    if os.path.exists(config_path):
                        overwrite = input(f"A config file with the name {experiment_name} already exists. "
                                          "Do you want to overwrite? [y/n] ")
                        if overwrite == "n":
                            exit(0)

                    if not os.path.exists(f'results/{experiment_name}'):
                        os.mkdir('results/' + experiment_name)
                    if not os.path.exists(f'images/{experiment_name}'):
                        os.mkdir('images/' + experiment_name)

                    with open(config_path, "w") as f:
                            yaml.dump(config_dict, f)
                    
                    # Run experiment
                    subprocess.run(["python", "src/main.py",
                                    "--config", config_path])
                
                exp_number += 1
