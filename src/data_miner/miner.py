from os import getenv
from data_miner.gh_miner import GitHubDataMiner
from assets.constants import DATA_MINER_TASKS
from log_config import LoggerConfig

# Initialize the logger for this module
logger = LoggerConfig(__name__).logger


class DataMiner:
    def start(self, module_code: str):
        """Starts the relevant module and its task
        Args:
            module_code (str): module and task code separated by dot
        """
        if module_code == f"{getenv('MODULE_DATA_MINER', 'DATA_MINER')}.{DATA_MINER_TASKS['mine_gh']}":
            GitHubDataMiner().start()
        else:
            logger.warning(f'Invalid module code: {module_code}')
