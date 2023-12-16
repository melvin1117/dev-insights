from datetime import datetime
import pandas as pd
import numpy as np
import emoji
from textblob import TextBlob
import math
from typing import Dict, Any, List
from utils.helper_functions import detect_language, calculate_sentiment_with_emoticons, load_language_model, z_score_normalization, calculate_oldness_factor
# from utils.df_chunk_concurrent_executor import DFChunkConcurrentExecutor
from os import getenv
from log_config import LoggerConfig
from database.session import Session
from pymongo import UpdateOne
from assets.constants import GITHUB, USER_RATING_WEIGHTS, USER_BIO_RATING_WEIGHTS
from utils.location_geocoding_service import LocationGeocodingService

# Initialize the logger for this module
logger = LoggerConfig(__name__).logger

RATING_DELTA: float = float(getenv('RATING_DELTA', 0.02))


class UserRatingAnalyzer:
    def __init__(self, grouped_repo_df: pd.DataFrame, mean_std_dict: Dict[str, float], repo_lang_mean_dict: Dict[str, float], location_service: LocationGeocodingService,  weights: Dict[str, float] = USER_RATING_WEIGHTS):
        """
        Initialize the UserRatingAnalyzer.

        Args:
            weights (Dict[str, float]): Weights for different factors in the rating calculation.
            workers_count (int): Number of workers for concurrent execution.
            chunk_size (int): Size of data chunks for concurrent execution.
        """
        self.group_repo_df = grouped_repo_df
        self.mean_std_dict = mean_std_dict
        self.repo_lang_mean_dict = repo_lang_mean_dict
        self.location_service = location_service
        self.weights = weights
        self.total_user_updated = 0

    def analyze_user_bio(self, bio: str, weights: Dict[str, float] = USER_BIO_RATING_WEIGHTS) -> float:
        """
        Analyze the impact rating of a user bio.

        Args:
            bio (str): User bio.
            weights (Dict[str, float]): Weights for different components of the impact rating.

        Returns:
            float: Absolute impact rating.
        """
        bio_with_emojis = emoji.demojize(bio)
        language_code = detect_language(bio_with_emojis)
        bio_nlp = load_language_model(language_code)

        sentiment_score = calculate_sentiment_with_emoticons(bio_with_emojis)
        bio_embedding = bio_nlp(bio_with_emojis).vector

        impact_rating = (
            weights['bio'] * np.sum(bio_embedding) +
            weights['sentiment'] * sentiment_score +
            weights['subjectivity'] * TextBlob(bio).sentiment.subjectivity
        )

        return abs(impact_rating)

    def calculate_user_rating(self, user: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the rating for a user.

        Args:
            user (Dict[str, Any]): User data.

        Returns:
            Dict[str, Any]: Calculated user rating dictionary.
        """
        rating_dict = {}
        # Extract relevant information
        public_repos = 0 if user["public_repos"] == -1 else user["public_repos"]
        public_gists = 0 if user["public_gists"] == -1 else user["public_gists"]
        followers = 0 if user["followers"] == -1 else user["followers"]
        following = 0 if user["following"] == -1 else user["following"]
        contributions = sum(len(values) for values in user['repos_contributed'].values())
        # contributions = user['repos_contributed'].apply(lambda x: sum(len(values) for values in x.values()) if x else 0)
        created_at = datetime.strptime(user["created_at"], "%Y-%m-%dT%H:%M:%SZ")
        oldness_factor = calculate_oldness_factor(created_at)
        bio = user["bio"]
        bio_factor = 0 if not bio else self.analyze_user_bio(bio)

        # Z-score normalization for relevant metrics
        public_repos_normalized = z_score_normalization(public_repos, self.mean_std_dict["public_repos"]["mean"], self.mean_std_dict["public_repos"]["std_dev"])
        public_gists_normalized = z_score_normalization(public_gists, self.mean_std_dict["public_gists"]["mean"], self.mean_std_dict["public_gists"]["std_dev"])
        followers_normalized = z_score_normalization(followers, self.mean_std_dict["followers"]["mean"], self.mean_std_dict["followers"]["std_dev"])
        following_normalized = z_score_normalization(following, self.mean_std_dict["following"]["mean"], self.mean_std_dict["following"]["std_dev"])
        contributions_normalized = z_score_normalization(contributions, self.mean_std_dict["contributions"]["mean"], self.mean_std_dict["contributions"]["std_dev"])

        # Calculate the rating without the PageRank factor
        rating = (
            self.weights["public_repos"] * public_repos_normalized +
            self.weights["public_gists"] * public_gists_normalized +
            self.weights["followers"] * followers_normalized +
            self.weights["following"] * following_normalized +
            self.weights["contributions"] * contributions_normalized +
            self.weights["oldness"] * oldness_factor +
            self.weights["bio"] * bio_factor
        )

        if math.isnan(rating):
            logger.warning(f'Defaulting rating for {user["gid"]} -  PR: {public_repos}, PG: {public_gists},  FLW: {followers},  FLWG: {following}, C: {contributions}, O: {oldness_factor}, B: {bio_factor}')
            rating = RATING_DELTA

        lang_rating = self.calculate_lang_ratings(user_gid=user["gid"], general_rating=rating, user_contributions=user['repos_contributed'])
        rating_dict = {
            "general": rating,
            **lang_rating
        }
        return rating_dict
      
    def calculate_lang_ratings(self, user_gid: int,  general_rating: float, user_contributions: Dict[str, List[int]]) -> Dict[str, Dict[str, float]]:
        """Calculates the rating for languages user contributed

        Args:
            user_gid (int): user id
            general_rating (float): General rating of the user
            user_contributions (Dict[str, List[int]]): users contributions 

        Returns:
            Dict[str, Dict[str, float]]: rating by language
        """
        user_lang_dict = {}

        for lang, repo_gids in user_contributions.items():
            # Filter the grouped DataFrame for the current language's repository gids
            lang_repos = self.group_repo_df[self.group_repo_df['gid'].isin(repo_gids)]
            
            # Extract contributions for the given user_gid
            contributors = lang_repos['contributors'].apply(lambda x: x.get(str(user_gid), {}).get('contributions', 1))

            # Check if user_gid matches owner_gid, add n_rating to user's rating
            owner_rating = lang_repos[lang_repos['owner_gid'] == user_gid]['n_rating'].sum()

            # Calculate language-wise rating
            lang_rating = (lang_repos['n_rating'] * contributors).sum() + owner_rating
            
            # If there are missing repo_gids, use the mean n_rating for the language
            missing_gids = set(repo_gids) - set(lang_repos['gid'])
            
            if missing_gids:
                lang_rating += len(missing_gids) * self.repo_lang_mean_dict.get(lang, RATING_DELTA)


            # Add the language and ratings to user_lang_dict
            user_lang_dict[lang] = {
                'lang_rating': lang_rating,
                'final_rating': lang_rating + general_rating
            }

        return user_lang_dict

    def add_location(self, user: Dict[str, Any]) -> Dict[str, Any]:
        """add location lat and long for the given location

        Args:
            user (Dict[str, Any]): user data

        Returns:
            Dict[str, Any]: lat long of the location
        """
        address = user['location']
        location_obj = self.location_service.geocode(address)
        return location_obj or {}
    
    def bulk_save_user_rating(self, user_df: pd.DataFrame) -> int:
        """Save the user ratings and location to the database

        Args:
            user_df (pd.DataFrame): user chunk dataframe

        Returns:
            int: number of users updated
        """
        with Session() as session:
            # Define a function to create UpdateOne objects
            def create_update(row):
                filter_query = {'gid': row['gid']}
                update_query = {"$set": {"rating": row["rating"], "loc": row["loc"] }}
                return UpdateOne(filter_query, update_query, upsert=False)

            # Apply the function to each row of the DataFrame
            bulk_operations = user_df.apply(create_update, axis=1).tolist()

            # Perform bulk update
            result = session[GITHUB['user']].bulk_write(bulk_operations)

            # Return the number of records updated
            logger.info(f"Updated rating for {result.modified_count} users")
            return result.modified_count
    
    def update_rating_and_location(self, user: pd.Series) -> pd.Series:
        """updates rating and location to the user series

        Args:
            user (pd.Series): row data for a user

        Returns:
            pd.Series: row data with rating and location added in it
        """
        rating = self.calculate_user_rating(user)
        loc = self.add_location(user)
        return pd.Series({'rating': rating, 'loc': loc})
    
    def process_chunk(self, chunk: pd.DataFrame) -> None:
        """
        Process a chunk of data and calculate ratings for each user.

        Args:
            chunk (pd.DataFrame): User data chunk.
        """
        try:
            chunk = chunk.join(chunk.apply(self.update_rating_and_location, axis=1))
        except Exception as e:
            logger.error(f"Error processing rating of user: {e}")
        user_update_count = self.bulk_save_user_rating(chunk)
        chunk_size = len(chunk.index)
        self.total_user_updated += user_update_count
        if user_update_count != chunk_size:
            logger.warning(f"Could not update rating and location for {chunk_size - user_update_count}, only updated {user_update_count} users.")
