from utils.file_loader import load_config, load_repo_weights, load_user_weights

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
USER_RATING_WEIGHTS = load_user_weights()

REPO_DESCRIPTION_RATING_WEIGHTS = {
    "description": 0.3,
    "sentiment": 0.3,
    "subjectivity": 0.4,
}

USER_BIO_RATING_WEIGHTS = {
    "bio": 0.2,
    "sentiment": 0.35,
    "subjectivity": 0.45,
}

ETL_TASKS = {
    "clear_repo_ratings": 'CLR_DB_REPO_RATING',
    "calculate_repo_weight": 'CAL_REPO_WEIGHT',
    "calculate_repo_rating": 'CAL_REPO_RATING',
    "normalize_repo_rating": 'NORM_REPO_RATING',
    "calculate_user_weight": 'CAL_USER_WEIGHT',
    "calculate_user_rating": 'CAL_USER_RATING',
    "normalize_user_rating": 'NORM_USER_RATING',
    "assign_user_proficiency": 'PROF_USER',
}

DATA_MINER_TASKS = {
    "mine_gh": 'MINE_GH'
}

BEGINNER = 'Beginner'
INTERMEDIATE = 'Intermediate'
EXPERT = 'Expert'

PROFICIENCY = [BEGINNER, INTERMEDIATE, EXPERT]
