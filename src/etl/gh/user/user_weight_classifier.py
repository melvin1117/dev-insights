from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from datetime import datetime
from sklearn.inspection import permutation_importance
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from utils.helper_functions import calculate_oldness_factor, display_execution_time
from utils.file_loader import USER_WEIGHTS_FILE_NAME
from assets.constants import GITHUB
from database.session import Session
from log_config import LoggerConfig

# Initialize the logger for this module
logger = LoggerConfig(__name__).logger

"""
Uses RandomForestRegressor with unsupervised learning
to calculate the user column weights for further user rating calculations.
"""

class UserWeightClassifier:
    
    def __init__(self) -> None:
        self.df = None

    def fetch_users(self) -> pd.DataFrame:
        """
        Fetch users from MongoDB and return as a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing user data.
        """
        try:
            with Session() as session:
                cursor = session[GITHUB['user']].find({})
                user_list = list(cursor)
                logger.info(f"Total number of records fetched: {len(user_list)}")
                return pd.DataFrame(user_list)
        except Exception as e:
            logger.error(f"Error fetching users: {e}")
            return pd.DataFrame()

    def _preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess user data.
        """
        self.df['created_at'] = pd.to_datetime(self.df['created_at'], format="%Y-%m-%dT%H:%M:%SZ")
        self.df['oldness_factor'] = self.df['created_at'].apply(calculate_oldness_factor)
        self.df['contribution_factor'] = self.df['repos_contributed'].apply(lambda x: sum(len(values) for values in x.values()) if x else 0)
        self.df['bio'] = self.df['bio'].fillna('')

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.df['bio'])
        self.df['bio_factor'] = tfidf_matrix.mean(axis=1).A.flatten()


    def start(self) -> None:
        """Starts the execution
        """
        start_time = datetime.now()
        logger.debug("Stared execution of calculation of user columns weights")
        self.df = self.fetch_users()

        if not self.df.empty:
            weights = self.analyze_users()
            logger.info(f"Calculated user weights: {weights}")
            # Save weights to a file (adjust the filename as needed)
            weights_file = f"assets/{USER_WEIGHTS_FILE_NAME}.json"
            with open(weights_file, 'w') as file:
                json.dump(weights, file, indent=4)
            logger.info(f"Weights saved to {weights_file}")
            
        display_execution_time(start_time, "User weight classification completed")
    
    def analyze_users(self) -> dict:
        """
        Analyze users and calculate weights for each desired columns


        Returns:
            dict: Dictionary containing user column weights.
        """
        try:
            self._preprocess_data()
            selected_columns = ["public_repos", "public_gists", "followers", "following", "oldness_factor", "contribution_factor", "bio_factor"]
            output_columns = ["public_repos", "public_gists", "followers", "following", "oldness", "contributions", "bio"]
            dummy_target = pd.Series(data=np.random.rand(len(self.df)), name="gid")
            model = RandomForestRegressor(random_state=42)
            model.fit(self.df[selected_columns], dummy_target)

            perm_importance = permutation_importance(model, self.df[selected_columns], dummy_target,
                                                     n_repeats=30, random_state=42)

            importance_scores = perm_importance.importances_mean
            following_factor = 0.08 / importance_scores[3]
            bio_scale_factor = 0.08 / importance_scores[-1]
            importance_scores[3] *= following_factor
            importance_scores[-1] *= bio_scale_factor

            weights = importance_scores / importance_scores.sum()
            weights_dict = dict(zip(output_columns, weights))

            return weights_dict

        except Exception as e:
            logger.error(f"Error analyzing users: {e}")
            return {}
