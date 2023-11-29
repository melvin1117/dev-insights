import pandas as pd
import numpy as np
from typing import Dict
from etl.repo.repository_analyzer import RepositoryAnalyzer
from utils.df_chunk_concurrent_executor import DFChunkConcurrentExecutor
from asset.constants import REPO_RATING_WEIGHTS
from log_config import LoggerConfig
from database.session import Session
from asset.constants import GITHUB
from datetime import datetime
from utils.helper_functions import display_execution_time, normalize_to_1
from database.session import Session
from pymongo import UpdateOne
from asset.constants import GITHUB

# Initialize the logger for this module
logger = LoggerConfig(__name__).logger


class RepositoryRatingOrchestrator:
    def __init__(
        self,
        weights: Dict[str, Dict[str, float]] = REPO_RATING_WEIGHTS,
        workers_count: int = 5,
        chunk_size: int = 5000,
    ) -> None:
        """
        Initialize the RepositoryRatingOrchestrator.

        Parameters:
            weights (Dict[str, int]): Dictionary of weights for different features.
            workers_count (int): Number of concurrent workers for processing chunks.
            chunk_size (int): Size of data chunks for concurrent processing.
        """
        self.workers_count = workers_count
        self.chunk_size = chunk_size
        self.mean_std_dict = None
        self.weights = weights
        self.df = None
        self.repositories_saved = 0

    def fetch_repositories(self) -> int:
        """
        Fetch repositories from MongoDB and create a DataFrame.
        Connects to MongoDB and fetches repositories with missing ratings.
        Returns:
            int: size of the loaded records
        """
        with Session() as session:
            cursor = session[GITHUB["repo"]].find({})
            repo_list = list(cursor)
            logger.info(f"Total number of records fetched: {len(repo_list)}")
            self.df = pd.DataFrame(repo_list)
            return len(repo_list)

    def calculate_mean_std(self) -> None:
        """
        Calculate mean and standard deviation for selected columns grouped by language.
        """
        mean_std_dict = {}
        selected_columns = [
            "stargazers_count",
            "watchers_count",
            "forks_count",
            "open_issues_count",
            "contributors",
        ]

        for language, language_group in self.df.groupby("language"):
            language = language.lower()
            mean_std_dict[language] = {}
            for column in selected_columns:
                column = column.lower()
                if column == "contributors":
                    contributors_counts = [
                        len(contributors) for contributors in language_group[column]
                    ]
                    mean_std_dict[language][column] = {
                        "mean": np.mean(contributors_counts),
                        "std_dev": np.std(contributors_counts),
                    }
                else:
                    column_data = language_group[column].fillna(0)
                    mean_std_dict[language][column] = {
                        "mean": column_data.mean(),
                        "std_dev": column_data.std(),
                    }
        self.mean_std_dict = mean_std_dict
        logger.info(f"Calculated mean std for the given data frame {self.mean_std_dict}")

    def start(self) -> None:
        """
        Start the orchestration of calculating repository rating.
        """
        run_start_time = datetime.now()
        size = self.fetch_repositories()
        if size == 0:
            logger.warning("No records fetched")
        else:

            self.calculate_mean_std()
            # Check if 'n_rating' is present in the DataFrame
            if 'n_rating' in self.df.columns:
                # Filter rows where 'n_rating' is not present or is NaN
                self.df = self.df[pd.isna(self.df['n_rating'])]

            if len(self.df.index) > 0:
                self.run_concurrent_executor()
                self.df.to_csv("before_normalization.csv", sep=',', encoding='utf-8')
                self.normalize_ratings()
                self.df.to_csv("after_normalization.csv", sep=',', encoding='utf-8')
                self.repositories_saved  = self.bulk_save_repo_rating()
            else:
                logger.warning("No records without n_rating")

        logger.info(f"Calculated ratings for total {self.repositories_saved} repositories")
        display_execution_time(run_start_time, "Repository rating calculation via orchestrator completed")

    def run_concurrent_executor(self) -> None:
        """
        Run concurrent executor for processing data chunks.
        """
        concurrent_executor = DFChunkConcurrentExecutor(
            df=self.df,
            workers_count=self.workers_count,
            exec_func=self.process_chunk,
            chunk_size=self.chunk_size,
            executor_name=self.__class__.__name__,
        )
        # Start parallel execution
        concurrent_executor.start()

    def normalize_ratings(self) -> None:
        """
        Normalize repository ratings.
        """
        grouped_data = self.df.groupby("language")["rating"]
        min_ratings = grouped_data.min()
        max_ratings = grouped_data.max()
        logger.info(f"Min Max rating language wise \n Min Rating: {min_ratings} \n Max Rating: {max_ratings}")
        self.df["n_rating"] = self.df.apply(
            lambda row: normalize_to_1(
                row["rating"], min_ratings[row["language"]], max_ratings[row["language"]]
            ),
            axis=1,
        )
    
    def bulk_save_repo_rating(self) -> int:
        """Save the ratings to the database
        """
        with Session() as session:
            # Define a function to create UpdateOne objects
            def create_update(row):
                filter_query = {'gid': row['gid']}
                update_query = {"$set": {"rating": row["rating"], "n_rating": row["n_rating"]}}
                return UpdateOne(filter_query, update_query, upsert=False)

            # Apply the function to each row of the DataFrame
            bulk_operations = self.df.apply(create_update, axis=1).tolist()

            # Perform bulk update
            result = session[GITHUB['repo']].bulk_write(bulk_operations)

            # Return the number of records updated
            logger.info(f"Updated rating for {result.modified_count} repositories")
            return result.modified_count

    def merge_rating(self, chunk_df: pd.DataFrame):
        """Merge the ratings in the chunk_df to the source df

        Args:
            chunk_df (pd.DataFrame): Chunked data frame with ratings
        """
        mapping = dict(zip(chunk_df['gid'], chunk_df['rating']))
        
        # Check if 'rating' column exists in self.df
        if 'rating' in self.df:
            self.df['rating'] = self.df['gid'].map(mapping).fillna(self.df['rating'])
        else:
            self.df['rating'] = self.df['gid'].map(mapping)

    def process_chunk(self, chunk_df: pd.DataFrame) -> None:
        """
        Process a chunk of data using RepositoryAnalyzer.

        Parameters:
            chunk_df (pd.DataFrame): Chunk of data to be processed.
        """
        # Create RepositoryAnalyzer instance
        repo_analyzer = RepositoryAnalyzer(chunk_df, self.weights, self.mean_std_dict)
        # Analyze repositories
        updated_chunk_df= repo_analyzer.analyze()
        self.merge_rating(updated_chunk_df)
