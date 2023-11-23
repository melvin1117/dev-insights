from utils.config_loader import load_config

# Define a global constant by reading the JSON file
CONFIG_DATA = load_config()
AUTH_HEADER_NAME = "Authorization"
AUTH_BEARER = "Bearer"

# Collection Name
GITHUB = {
  'repo': 'gh-repo',
  'user': 'gh-user',
  'tracker': 'gh-tracker'
}
