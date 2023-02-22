import json


def read_config(config_file: str = "config.json") -> dict:
    """Function used to read the json configuration.

    Returns:
        dict: Dictionary of json configuration.
    """
    config = {}
    with open(config_file, "r") as cfg:
        config = json.loads(cfg.read())
    return config
