from pymongo import MongoClient, database
from pymongo.errors import PyMongoError
from os import getenv
from log_config import LoggerConfig
from typing import Optional, Type

# Initialize the logger for this module
logger = LoggerConfig(__name__).logger

class Session:
    """
        Session class facilitates a session for MongoDB transactions.

        This context manager handles opening and closing connections,
        starting and committing transactions, and exception handling.
    """

    def __init__(self) -> None:
        self.host = getenv('DB_HOST')
        self.port = int(getenv('DB_PORT'))
        self.database_name = getenv('DB_NAME')
        self.username = getenv('DB_USER')
        self.password = getenv('DB_PASS')
        self.client = None 
        self.db = None 
        self.connection_url = f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/"

    def __enter__(self) -> database.Database:
        """Starts a new MongoDB client and session, and begins a transaction.

        Returns:
            database.Database: returns database class instance
        """
        try:
            self.client = MongoClient(self.connection_url)
            self.session = self.client.start_session()
            self.start_transaction()
            self.db = self.client[self.database_name]
            return self.db
        except PyMongoError as e:
            logger.error(f"An error occurred while starting a MongoDB session: {e}")
            raise

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[object]) -> None:
        """Commits or aborts the transaction, then ends the session and closes the client.

        Args:
            exc_type (Optional[Type[BaseException]]): indicates class of exception
            exc_val (Optional[BaseException]): indicates type of exception
            exc_tb (Optional[object]): exception traceback
        """
        if exc_type:
            logger.warning(f"Exception occurred while exiting: {exc_val}, aborting transaction.")
            self.abort_transaction()
        else:
            self.commit_transaction()
        self.session.end_session()
        self.client.close()


    def start_transaction(self) -> None:
        """Starts a new transaction for the session."""
        try:
            self.session.start_transaction()
        except PyMongoError as e:
            logger.error(f"An error occurred while starting a transaction: {e}")
            raise


    def commit_transaction(self) -> None:
        """Commits the ongoing transaction."""
        try:
            self.session.commit_transaction()
        except PyMongoError as e:
            logger.error(f"An error occurred while committing the transaction: {e}")
            raise

    def abort_transaction(self) -> None:
        """Aborts the ongoing transaction."""
        try:
            self.session.abort_transaction()
        except PyMongoError as e:
            logger.error(f"An error occurred while aborting the transaction: {e}")
            raise
