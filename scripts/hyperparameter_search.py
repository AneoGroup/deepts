import os
import subprocess
import yaml

"""
The script used to do a grid search for hyperparameters. The model, dataset, etc. is set in the first block
of code. The hyperparameters we search over is set in the second block of code.
"""
model = "deep_ar"
dataset = "electricity"
use_val_data = False
track_training = False
save_weights = False
seed = 97

lrs = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
epochs = [10, 20, 100]
batch_sizes = [16, 32, 64, 256]
num_batches_per_epochs = [50, 100]

model_args = {"prediction_length": None, "freq": None}

search_num = 0
for lr in lrs:
    for epoch in epochs:
        for batch_size in batch_sizes:
            for num_batches in num_batches_per_epochs:
                path = f"./experiments/{model}/{dataset}/hps/seed{seed}_{search_num}"

                trainer_args = {
                    "learning_rate": lr,
                    "epochs": epoch,
                    "batch_size": batch_size,
                    "num_batches_per_epoch": num_batches,
                    "ctx": "gpu"
                }

                config_dict = {
                    "model_name": model,
                    "dataset_name": dataset,
                    "hyperparams": model_args,
                    "trainer": trainer_args,
                    "random_seed": seed,
                    "cross_val": False,
                    "use_val_data": use_val_data,
                    "track_training": track_training,
                    "save_weights": save_weights,
                    "path": path
                }

                if not os.path.exists(path):
                    os.makedirs(path)
                
                with open(f"{path}/config.yaml", "w") as f:
                    yaml.dump(config_dict, f)
                
                subprocess.run(["python", "src/main.py",
                    "--config", path + "/config.yaml"])

                search_num += 1
