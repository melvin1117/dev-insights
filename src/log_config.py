import os
import logging

DEFAULT_LOGGING_LEVEL = 'INFO'
LOGS_FOLDER = 'logs'


class LoggerConfig:
    """Configure logging for the application based on environment variables."""
    def __init__(self, module_name):
        self.module_name = module_name
        self.app_name = os.getenv('APP_NAME', 'MyApp')
        self.logging_level = self.get_logging_level()
        self.logger = self.setup_logger()

    def get_logging_level(self) -> int:
        """Retrieves the logging level from the environment variable."""
        logging_level_name = os.getenv('LOGGING_LEVEL', DEFAULT_LOGGING_LEVEL).upper()
        logging_level = getattr(logging, logging_level_name, None)

        if not isinstance(logging_level, int):
            # Log a warning about the invalid logging level and use the default
            default_level = getattr(logging, DEFAULT_LOGGING_LEVEL)
            logging.basicConfig(level=default_level)  # Basic config to enable logging
            logging.warning(f'Invalid log level: {logging_level_name}. Defaulting to {DEFAULT_LOGGING_LEVEL} level.')
            return default_level

        return logging_level

    def setup_logger(self) -> logging.Logger:
        """Sets up and returns a logger for a specific module."""
        logger = logging.getLogger(f'{self.app_name}.{self.module_name}')
        logger.setLevel(self.logging_level)

        # Create file handler for the module and set level
        log_file_name = f'{self.module_name}.log'
        os.makedirs(os.path.join(os.getcwd(), LOGS_FOLDER), exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(LOGS_FOLDER, log_file_name))
        file_handler.setLevel(self.logging_level)

        # Create a formatter and add it to the handler
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                           datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)

        # Add the file handler to the logger
        if not logger.handlers:
            logger.addHandler(file_handler)

        return logger
