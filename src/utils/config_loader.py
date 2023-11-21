import json

def load_config():
    with open('src/config.json', 'r') as config_file:
        config = json.load(config_file)
    return config
