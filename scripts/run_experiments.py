import yaml
import subprocess

model_lists = ["rnn2qr"]

dataset_list = ['constant']

epochs_list =  [1]
lr_list = [0.00001]
lr_decay_list = [0.1]

config_path = "./configs/config.yaml"


# first loop through models
for model in model_lists:
    # then loop through datasets
    for dataset in dataset_list:
        # we need to set the experiment number
        exp_number = 1
        # then we loop on hyperparameters
        for epoch in epochs_list:
            for lr in lr_list:
                for lr_decay in lr_decay_list:
                    # change the yaml
                    with open(config_path, "r") as f:
                        config = yaml.load(f, Loader=yaml.FullLoader)

                    config['model_name'] = model
                    config['dataset_name'] = dataset
                    config['trainer']['learning_rate'] = lr
                    config['trainer']['learning_rate_decay_factor'] = lr_decay
                    config['trainer']['epochs'] = epoch
                    config['exp_num'] = exp_number

                    with open(config_path, "w") as f:
                        yaml.dump(config, f)
                    exp_number += 1

                    exp_name = config['model_name'] + '__' + config['dataset_name'] + '__' + str(config['exp_num'])
                    
                    subprocess.run( ['python', 'src/main.py'] )

