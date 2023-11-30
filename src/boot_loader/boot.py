from os import getenv
from etl.etl import ETLProcess
from data_miner.miner import DataMiner
from log_config import LoggerConfig

# Initialize the logger for this module
logger = LoggerConfig(__name__).logger


class Boot:
    def __init__(self, module_code: str):
        """Boot constructor

        Args:
            module_code (str): module code to for which the application is started
        """
        self.module_code = module_code

    def start(self):
        logger.info(f"Started execution for {self.module_code} module.")
        if self.module_code.startswith(getenv('MODULE_ETL', 'ETL')):
            ETLProcess().start(self.module_code)

        if self.module_code.startswith(getenv('MODULE_DATA_MINER', 'DATA_MINER')):
            DataMiner().start(self.module_code)
