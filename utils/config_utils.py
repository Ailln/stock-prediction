import yaml


def read_config(config_path):
    with open(config_path, "r") as f_config:
        config = yaml.load(f_config)
    return config
