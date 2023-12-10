import json

REPO_WEIGHTS_FILE_NAME = "repo_rating_weights"
USER_WEIGHTS_FILE_NAME = "user_rating_weights"

def load_config():
    with open("assets/config.json", "r") as config_file:
        config = json.load(config_file)
    return config

def load_repo_weights():
    with open(f"assets/{REPO_WEIGHTS_FILE_NAME}.json", "r") as weights_file:
        weights = json.load(weights_file)
    return weights

def load_user_weights():
    with open(f"assets/{USER_WEIGHTS_FILE_NAME}.json", "r") as weights_file:
        weights = json.load(weights_file)
    return weights
