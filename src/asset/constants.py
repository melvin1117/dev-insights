from utils.file_loader import load_config, load_repo_weights

# Define a global constant by reading the JSON file
CONFIG_DATA = load_config()
AUTH_HEADER_NAME = "Authorization"
AUTH_BEARER = "Bearer"

# Collection Name
GITHUB = {
    "repo": "gh-repo",
    "user": "gh-user",
    "tracker": "gh-tracker"
}

REPO_RATING_WEIGHTS = load_repo_weights()

REPO_DESCRIPTION_RATING_WEIGHTS = {
    "description": 0.3,
    "sentiment": 0.3,
    "subjectivity": 0.4,
}