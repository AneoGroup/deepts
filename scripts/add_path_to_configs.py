import os
import yaml

"""Adds a field called path to all config files in the experiments folder.
Used to make old config files compatable with changes made in src/main.py"""

for r, d, f in os.walk("./experiments"):
    if "config.yaml" in f:
        with open(r + "/config.yaml", "r") as config:
            config_dict = yaml.load(config, yaml.FullLoader)
        
        config_dict["path"] = r
        with open(r + "/config.yaml", "w") as config:
            yaml.dump(config_dict, config)
