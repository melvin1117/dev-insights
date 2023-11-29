from log_config import LoggerConfig
from datetime import datetime
from etl.repo.repository_rating_orchestrator import RepositoryRatingOrchestrator
# from etl.repo.repo_weight_classifier import RepositoryWeightClassifier    
from database.session import Session

# Initialize the logger for this module
logger = LoggerConfig(__name__).logger


class ETLProcess:
    def __init__(self):
        pass

    def start(self):
        # self.clear_ratings()
        # RepositoryWeightClassifier().start()
        repo_rating_orchestrator = RepositoryRatingOrchestrator()
        repo_rating_orchestrator.start()

    def clear_ratings(self):
        with Session() as session:
            res = session['gh-repo'].update_many({}, {'$unset': {'n_rating': 1, 'rating': 1}})
            # Print the number of documents modifiedEmpty DataFrame
            print(f"Number of documents modified: {res.modified_count}")
