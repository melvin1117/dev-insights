from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from datetime import datetime
from sklearn.inspection import permutation_importance
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from utils.helper_functions import calculate_recency_factor, display_execution_time
from utils.file_loader import REPO_WEIGHTS_FILE_NAME
from asset.constants import GITHUB
from database.session import Session
from log_config import LoggerConfig

# Initialize the logger for this module
logger = LoggerConfig(__name__).logger

"""
Uses RandomForestRegressor with unsupervised learning
to calculate the repository column weights for further repo rating calculations.
"""

class RepositoryWeightClassifier:
    
    def __init__(self) -> None:
        self.df = None

    def fetch_repositories(self) -> pd.DataFrame:
        """
        Fetch repositories from MongoDB and return as a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing repository data.
        """
        try:
            with Session() as session:
                cursor = session[GITHUB['repo']].find({})
                repo_list = list(cursor)
                logger.info(f"Total number of records fetched: {len(repo_list)}")
                return pd.DataFrame(repo_list)
        except Exception as e:
            logger.error(f"Error fetching repositories: {e}")
            return pd.DataFrame()

    def _preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess repository data.
        """
        self.df['pushed_at'] = pd.to_datetime(self.df['pushed_at'], format="%Y-%m-%dT%H:%M:%SZ")
        self.df['recency_factor'] = self.df['pushed_at'].apply(calculate_recency_factor)

        self.df['contributors_factor'] = self.df['contributors'].apply(lambda x: len(x.keys()) if isinstance(x, dict) else 0)
        self.df['description'] = self.df['description'].fillna('')

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.df['description'])
        self.df['description_factor'] = tfidf_matrix.mean(axis=1).A.flatten()


    def start(self) -> None:
        """Starts the execution
        """
        start_time = datetime.now()
        logger.debug("Stared execution of calculation of repository columns weights")
        self.df = self.fetch_repositories()

        if not self.df.empty:
            weights = self.analyze_repositories()
            logger.info(f"Calculated weights: {weights}")
            # Save weights to a file (adjust the filename as needed)
            weights_file = f"asset/{REPO_WEIGHTS_FILE_NAME}.json"
            with open(weights_file, 'w') as file:
                json.dump(weights, file, indent=4)
            logger.info(f"Weights saved to {weights_file}")
            
        display_execution_time(start_time, "Repository weight classification completed")
    
    def analyze_repositories(self) -> dict:
        """
        Analyze repositories and calculate weights for each language group.


        Returns:
            dict: Dictionary containing language-wise weights.
        """
        try:
            self._preprocess_data()
            selected_columns = ["stargazers_count", "forks_count", "recency_factor", "contributors_factor",
                                "description_factor"]
            output_columns = ["stargazers", "forks", "recency", "contributors", "description"]
            unique_languages = self.df["language"].str.lower().unique()
            weights_dict = {}

            grouped = self.df.groupby("language")

            def calculate_weights(language_group):
                dummy_target = pd.Series(data=np.random.rand(len(language_group)), name="gid")
                model = RandomForestRegressor(random_state=42)
                model.fit(language_group[selected_columns], dummy_target)

                perm_importance = permutation_importance(model, language_group[selected_columns], dummy_target,
                                                         n_repeats=30, random_state=42)

                importance_scores = perm_importance.importances_mean
                recency_scale_factor = 0.1 / importance_scores[2]
                description_scale_factor = 0.05 / importance_scores[4]
                importance_scores[2] *= recency_scale_factor
                importance_scores[4] *= description_scale_factor

                weights = importance_scores / importance_scores.sum()
                return dict(zip(output_columns, weights))

            weights_dict = dict(zip(unique_languages, grouped.apply(calculate_weights)))
            return weights_dict

        except Exception as e:
            logger.error(f"Error analyzing repositories: {e}")
            return {}
