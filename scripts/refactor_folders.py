import os
import shutil
import yaml


# The host_id is used to ID experiments originating from "this" computer.
# Needs to be set to the correct value before running the script.
host_id = "B"

for f in os.listdir("./configs"):
    if "__" in f:
        with open("./configs/" + f, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        experiment_num, repetition_num = config["exp_num"].split("__")
        new_path = f"./experiments/{config['model_name']}/{config['dataset_name']}/{experiment_num}{host_id}/repetition{repetition_num}"
        old_path = f"{config['model_name']}__{config['dataset_name']}__{config['exp_num']}"

        os.makedirs(new_path)
        os.rename(f"./configs/{old_path}.yaml", f"{new_path}/config.yaml")
        os.rename(f"./images/{old_path}/plot.png", f"{new_path}/plot.png")
        os.rename(f"./results/{old_path}/forcasts.csv", f"{new_path}/forecasts.csv")  # Fixes the spelling mistake in the file name
        os.rename(f"./results/{old_path}/metrics.csv", f"{new_path}/metrics.csv")

        os.rmdir(f"./images/{old_path}")
        os.rmdir(f"./results/{old_path}")

shutil.rmtree("./images")
shutil.rmtree("./results")
