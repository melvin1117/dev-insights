from os import getenv
from etl.etl import ETLProcess
from data_miner.miner import DataMiner
from log_config import LoggerConfig

# Initialize the logger for this module
logger = LoggerConfig(__name__).logger


class Boot:
    def __init__(self, module_code):
        self.module_code = module_code

    def start(self):
        logger.info(f"Started execution for {self.module_code} module.")
        if self.module_code == getenv('MODULE_ETL', 'ETL'):
            etl_process = ETLProcess()
            etl_process.start()

        if self.module_code == getenv('MODULE_DATA_MINER', 'DATA_MINER'):
            data_miner = DataMiner()
            data_miner.start()
