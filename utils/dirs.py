import json
import os

from utils.config import load_config, compare_configs

def create_dirs(config):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        checkpoint_dir = config.checkpoint_dir
        name = config.name
        dir = os.path.join(checkpoint_dir, name)
        config.checkpoint_dir = dir + "/"
        if not os.path.exists(dir):
            os.makedirs(dir)
            os.makedirs(dir + "/summaries/")
        else:
            config_dir = dir + "/config.json"
            if os.path.isfile(config_dir):
                old_config = load_config(config_dir)
                for param in ["best_loss", "best_step", "best_epoch"]:
                    if (param not in vars(config)) and (param in vars(old_config)):
                        value = getattr(old_config, param)
                        setattr(config, param, value)
                compare_configs(old_config, config)
            config.checkpoint_dir = dir + "/"
        return config

    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)
        

                