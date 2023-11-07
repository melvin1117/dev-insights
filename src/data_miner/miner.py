from os import getenv
from database.session import Session
from log_config import LoggerConfig

# Initialize the logger for this module
logger = LoggerConfig(__name__).logger


class DataMiner:
    def __init__(self):
        self.api_key_github = getenv('GITHUB_API_KEY')
    
    def start(self):
        with Session() as session:
            try:
                session['user'].insert_one({'name': 'Alice Doe', 'age': 22})
                session['user'].update_one({'name': 'Alice Doe'}, {'$set': {'age': 23}})
                logger.info("Data Inserted Successfully.")
            except Exception as e:
                logger.error(f"An error occurred: {e}")
