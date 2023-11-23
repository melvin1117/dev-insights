from os import getenv
from data_miner.gh_miner import GitHubDataMiner


class DataMiner:
    def start(self):
        gh_miner = GitHubDataMiner()
        gh_miner.start()
