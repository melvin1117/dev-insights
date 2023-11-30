from log_config import LoggerConfig
from etl.gh.repo.repository_rating_orchestrator import RepositoryRatingOrchestrator
from etl.gh.repo.repo_weight_classifier import RepositoryWeightClassifier
from database.session import Session
from os import getenv
from asset.constants import ETL_TASKS, GITHUB
from etl.gh.user.user_weight_classifier import UserWeightClassifier

# Initialize the logger for this module
logger = LoggerConfig(__name__).logger

class ETLProcess:
    def __init__(self):
        pass

    def start(self, module_code: str):
        """Starts the relevant module and its task

        Args:
            module_code (str): module and task code separated by dot
        """
        if module_code == f"{getenv('MODULE_ETL', 'ETL')}.{ETL_TASKS['clear_repo_ratings']}":
            self.clear_ratings()
        elif module_code == f"{getenv('MODULE_ETL', 'ETL')}.{ETL_TASKS['calculate_repo_weight']}":
            RepositoryWeightClassifier().start()
        elif module_code == f"{getenv('MODULE_ETL', 'ETL')}.{ETL_TASKS['calculate_repo_rating']}":
            RepositoryRatingOrchestrator().start_calculation()
        elif module_code == f"{getenv('MODULE_ETL', 'ETL')}.{ETL_TASKS['normalize_repo_rating']}":
            RepositoryRatingOrchestrator().start_normalization()
        elif module_code == f"{getenv('MODULE_ETL', 'ETL')}.{ETL_TASKS['calculate_user_weight']}":
            UserWeightClassifier().start()
        else:
            logger.warning(f'Invalid module code: {module_code}')

    def clear_ratings(self):
        with Session() as session:
            res = session[GITHUB['repo']].update_many({}, {'$unset': {'n_rating': 1, 'rating': 1}})
            # Display the number of documents modifiedEmpty DataFrame
            logger.info(f"Cleared ratings for {res.modified_count} repo documents")
