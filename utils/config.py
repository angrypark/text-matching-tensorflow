import os
import json
import tensorflow as tf

from utils.logger import setup_logger

def load_config(config_dir):
    with open(config_dir, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config = tf.contrib.training.HParams(**config_dict)
    return config

def save_config(path, config):
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(config), f)

def compare_configs(old_config, new_config):
    logger = setup_logger()
    for param, value in vars(new_config).items():
        if param not in old_config:
            logger.warn("New parameter : {} {}".format(param, value))
        else:
            old_value = getattr(old_config, param)
            if value != old_value:
                logger.info("Parameter changed : {} {} -> {}".format(param.upper(), old_value, value))