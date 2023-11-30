from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import emoji
from textblob import TextBlob
import math
from typing import Dict, Any
from utils.helper_functions import detect_language, calculate_sentiment_with_emoticons, load_language_model, z_score_normalization, calculate_recency_factor
from utils.df_chunk_concurrent_executor import DFChunkConcurrentExecutor
from os import getenv
from asset.constants import REPO_DESCRIPTION_RATING_WEIGHTS
from log_config import LoggerConfig
from database.session import Session
from pymongo import UpdateOne
from asset.constants import GITHUB

# Initialize the logger for this module
logger = LoggerConfig(__name__).logger

RATING_DELTA: float = float(getenv('RATING_DELTA', 0.02))


class RepositoryAnalyzer:
    def __init__(self, df: Any, weights: Dict[str, float], mean_std_dict: Dict[str, Dict[str, float]], workers_count: int = 5, chunk_size: int = 100):
        """
        Initialize the RepositoryAnalyzer.

        Args:
            df (pd.DataFrame): The DataFrame containing repository data.
            weights (Dict[str, float]): Weights for different factors in the rating calculation.
            mean_std_dict (Dict[str, Dict[str, float]]): Mean and standard deviation data for different languages.
            workers_count (int): Number of workers for concurrent execution.
            chunk_size (int): Size of data chunks for concurrent execution.
        """
        self.df = df
        self.weights = weights
        self.mean_std_dict = mean_std_dict
        self.workers_count = workers_count
        self.chunk_size = chunk_size

    def bulk_save_repo_rating(self) -> int:
        """Save the ratings to the database
        """
        with Session() as session:
            # Define a function to create UpdateOne objects
            def create_update(row):
                filter_query = {'gid': row['gid']}
                update_query = {"$set": {"rating": row["rating"] }}
                return UpdateOne(filter_query, update_query, upsert=False)

            # Apply the function to each row of the DataFrame
            bulk_operations = self.df.apply(create_update, axis=1).tolist()

            # Perform bulk update
            result = session[GITHUB['repo']].bulk_write(bulk_operations)

            # Return the number of records updated
            logger.info(f"Updated rating for {result.modified_count} repositories")
            return result.modified_count
    
    def analyze(self) -> pd.DataFrame:
        """Analyze the repositories, run concurrent executor, normalize ratings, and optionally plot ratings.

        Returns:
            pd.DataFrame: updated dataframe with rating value
        """
        self.run_concurrent_executor()
        self.bulk_save_repo_rating()
        return self.df

    def analyze_repo_description(self, description: str, weights: Dict[str, float] = REPO_DESCRIPTION_RATING_WEIGHTS) -> float:
        """
        Analyze the impact rating of a repository description.

        Args:
            description (str): Repository description.
            weights (Dict[str, float]): Weights for different components of the impact rating.

        Returns:
            float: Absolute impact rating.
        """
        description_with_emojis = emoji.demojize(description)
        language_code = detect_language(description_with_emojis)
        description_nlp = load_language_model(language_code)

        sentiment_score = calculate_sentiment_with_emoticons(description_with_emojis)
        description_embedding = description_nlp(description_with_emojis).vector

        impact_rating = (
            weights['description'] * np.sum(description_embedding) +
            weights['sentiment'] * sentiment_score +
            weights['subjectivity'] * TextBlob(description).sentiment.subjectivity
        )

        return abs(impact_rating)

    def run_concurrent_executor(self) -> None:
        """
        Run concurrent executor for processing data chunks in parallel.
        """
        concurrent_executor = DFChunkConcurrentExecutor(
            df=self.df,
            workers_count=self.workers_count,
            exec_func=self.process_chunk,
            chunk_size=self.chunk_size,
            executor_name=self.__class__.__name__
        )

        # Start parallel execution
        concurrent_executor.start()

    def calculate_repo_rating(self, repository: Dict[str, Any]) -> float:
        """
        Calculate the rating for a repository.

        Args:
            repository (Dict[str, Any]): Repository data.

        Returns:
            float: Calculated repository rating.
        """
        # return random.uniform(0, 1)
        language = str(repository['language']).lower()
        mean_std_language = self.mean_std_dict.get(language, {})

        if not mean_std_language:
            logger.warning(f"Mean and std deviation data not available for language: {language}")
            raise Exception(f"Mean and std deviation data not available for language: {language}")

        # Extract relevant information
        stargazers = repository["stargazers_count"]
        forks = repository["forks_count"]
        contributors = len(repository["contributors"])
        pushed_at = datetime.strptime(repository["pushed_at"], "%Y-%m-%dT%H:%M:%SZ")
        recency_factor = calculate_recency_factor(pushed_at)
        description = repository["description"]
        description_factor = 0 if not description else self.analyze_repo_description(description)

        # Z-score normalization for relevant metrics
        stargazers_normalized = z_score_normalization(stargazers, mean_std_language["stargazers_count"]["mean"], mean_std_language["stargazers_count"]["std_dev"])
        forks_normalized = z_score_normalization(forks, mean_std_language["forks_count"]["mean"], mean_std_language["forks_count"]["std_dev"])
        contributors_normalized = z_score_normalization(contributors, mean_std_language["contributors"]["mean"], mean_std_language["contributors"]["std_dev"])

        # Calculate the rating without the PageRank factor
        rating = (
            self.weights[language]["stargazers"] * stargazers_normalized +
            self.weights[language]["forks"] * forks_normalized +
            self.weights[language]["contributors"] * contributors_normalized +
            self.weights[language]["recency"] * recency_factor +
            self.weights[language]["description"] * description_factor
        )

        # Save the rating for the repository
        repository["rating"] = rating

        if math.isnan(rating):
            logger.warning(f'Defaulting rating for {repository["gid"]} - {language} -  S: {stargazers_normalized}, F: {forks_normalized}, C: {contributors_normalized}, R: {recency_factor}, D: {description_factor}')
            return RATING_DELTA
        return rating

    def process_chunk(self, chunk: Any) -> None:
        """
        Process a chunk of data and calculate ratings for each repository.

        Args:
            chunk (pd.DataFrame): Data chunk.
        """
        for index, repository in chunk.iterrows():
            try:
                rating = self.calculate_repo_rating(repository)
                self.df.at[repository.name, "rating"] = rating
            except Exception as e:
                logger.error(f"Error processing repository {repository['gid']}: {e}")
