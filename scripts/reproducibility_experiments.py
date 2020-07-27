import yaml
import subprocess
import mxnet as mx
import numpy as np

mx.random.seed(0)
np.random.seed(0)

# basically same as  run_eperiment but loop over the same experiment more than one, learning decay is not changed also.
model_lists = ["deep_ar"]

dataset_list = ['electricity']

epochs_list =  [20]
lr_list = [0.001]
# number of experiment repetition
exp_repeat = 100

config_path = "./configs/test_config.yaml"
 

# first loop through models
for model in model_lists:
    # then loop through datasets
    for dataset in dataset_list:
        # we need to set the experiment number
        exp_number = 1
        # then we loop on hyperparameters
        for epoch in epochs_list:
            for lr in lr_list:
                # change the yaml
                with open(config_path, "r") as f:
                    config = yaml.load(f, Loader=yaml.FullLoader)

                config['model_name'] = model
                config['dataset_name'] = dataset
                config['trainer']['learning_rate'] = lr
                config['trainer']['epochs'] = epoch


                # loop through number of repetition which we want
                for current_rep in range(67, exp_repeat):
                    config['exp_num'] = f'{exp_number}__{current_rep}'
                    
                    # save the config
                    with open(config_path, "w") as f:
                        yaml.dump(config, f)

                    subprocess.run( ['python', 'src/main.py'] )
                exp_number += 1

