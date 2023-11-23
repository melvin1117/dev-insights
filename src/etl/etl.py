from database.session import Session
from log_config import LoggerConfig

# Initialize the logger for this module
logger = LoggerConfig(__name__).logger


class ETLProcess:
    def __init__(self):
        self.sample = "SAMPLE"

    def start(self):
        with Session() as session:
            try:
                session['user'].insert_one({'name': 'John Doe', 'age': 25})
                session['user'].update_one({'name': 'John Doe'}, {'$set': {'age': 26}})
                logger.debug("Data Inserted Successfully.")
            except Exception as e:
                print(f"An error occurred: {e}")
