import pandas as pd
import numpy as np
from typing import Dict
from etl.gh.user.user_rating_analyzer import UserRatingAnalyzer
from utils.df_chunk_concurrent_executor import DFChunkConcurrentExecutor
from log_config import LoggerConfig
from database.session import Session
from assets.constants import GITHUB, PROFICIENCY, BEGINNER, INTERMEDIATE, EXPERT
from datetime import datetime
from utils.helper_functions import display_execution_time, normalize_to_1
from database.session import Session
from pymongo import UpdateOne
from utils.location_geocoding_service import LocationGeocodingService
from sklearn.mixture import GaussianMixture


# Initialize the logger for this module
logger = LoggerConfig(__name__).logger


class UserRatingOrchestrator:
    def __init__(self, workers_count: int = 5, chunk_size: int = 200) -> None:
        """
        Initialize the UserRatingOrchestrator.

        Args:
            workers_count (int): Number of workers for concurrent execution.
            chunk_size (int): Size of data chunks for concurrent execution.
        """
        self.grouped_repo_df = None
        self.language_mean_ratings = None
        self.user_df = None
        self.mean_std_dict = None
        self.workers_count = workers_count
        self.chunk_size = chunk_size
        self.location_service = None
        self.user_analyzer = None

    def calculate_mean_std(self) -> None:
        """
        Calculate mean and standard deviation for selected columns.
        """
        mean_std_dict = {}
        selected_columns = [
            "public_repos",
            "public_gists",
            "followers",
            "following",
            "contributions",
        ]

        for column in selected_columns:
            column = column.lower()
            if column == "contributions":
                contributions_counts =  self.user_df ['repos_contributed'].apply(lambda x: sum(len(value) for value in x.values())).tolist()
                mean_std_dict[column] = {
                    "mean": np.mean(contributions_counts),
                    "std_dev": np.std(contributions_counts),
                }
            else:
                column_data = self.user_df[column].fillna(0)
                mean_std_dict[column] = {
                    "mean": column_data.mean(),
                    "std_dev": column_data.std(),
                    }
        self.mean_std_dict = mean_std_dict
        logger.info(f"Calculated mean std for the given data frame {self.mean_std_dict}")


    def fetch_repositories(self) -> int:
        """
        Fetch repositories from MongoDB and create a DataFrame.
        Returns:
            int: size of the loaded records
        """
        with Session() as session:
            cursor = session[GITHUB["repo"]].find({})
            repo_list = list(cursor)
            logger.info(f"Total number of repositories records fetched: {len(repo_list)}")
            repo_df = pd.DataFrame(repo_list)
            self.grouped_repo_df = repo_df.groupby(['gid', 'owner_gid']).agg({
                'contributors': 'first',
                'n_rating': 'sum'
            }).reset_index()
            self.language_mean_ratings = repo_df.groupby('language')['n_rating'].mean().to_dict()
            return len(repo_list)

    def fetch_users(self) -> int:
        """
        Fetch users from MongoDB and create a DataFrame.
        Returns:
            int: size of the loaded records
        """
        with Session() as session:
            cursor = session[GITHUB["user"]].find({})
            user_list = list(cursor)
            logger.info(f"Total number of users records fetched: {len(user_list)}")
            self.user_df = pd.DataFrame(user_list)
            return len(user_list)

    def bulk_save_user_normalized_rating(self) -> int:
        """Save the ratings to the database
        """
        with Session() as session:
            # Define a function to create UpdateOne objects
            def create_update(row):
                filter_query = {'gid': row['gid']}
                update_query = {"$set": {"n_rating": row["n_rating"] }}
                return UpdateOne(filter_query, update_query, upsert=False)

            # Apply the function to each row of the DataFrame
            bulk_operations = self.user_df.apply(create_update, axis=1).tolist()

            # Perform bulk update
            result = session[GITHUB['user']].bulk_write(bulk_operations)

            # Return the number of records updated
            logger.info(f"Updated normalized rating for {result.modified_count} users")
            return result.modified_count

    def bulk_save_user_proficiency(self) -> int:
        """Save the proficiency to the database
        """
        with Session() as session:
            # Define a function to create UpdateOne objects
            def create_update(row):
                filter_query = {'gid': row['gid']}
                update_query = {"$set": {"proficiency": row["proficiency"] }}
                return UpdateOne(filter_query, update_query, upsert=False)

            # Apply the function to each row of the DataFrame
            bulk_operations = self.user_df.apply(create_update, axis=1).tolist()

            # Perform bulk update
            result = session[GITHUB['user']].bulk_write(bulk_operations)

            # Return the number of records updated
            logger.info(f"Updated proficiency for {result.modified_count} users")
            return result.modified_count

    def run_concurrent_executor(self) -> None:
        """
        Run concurrent executor for processing data chunks in parallel.
        """
        concurrent_executor = DFChunkConcurrentExecutor(
            df=self.user_df,
            workers_count=self.workers_count,
            exec_func=self.user_analyzer.process_chunk,
            chunk_size=self.chunk_size,
            executor_name=self.__class__.__name__
        )

        # Start parallel execution
        concurrent_executor.start()

    def start_calculation(self) -> None:
        """Starts the orchestration of the user rating calculation
        """
        run_start_time = datetime.now()
        self.fetch_repositories()
        total_user_count = self.fetch_users()
        if total_user_count == 0:
            logger.warning("No user record fetched. Try again.")
        else:
            self.calculate_mean_std()
            if 'rating' in self.user_df.columns:
                # Filter rows where 'rating' is not present or is NaN
                self.user_df = self.user_df[pd.isna(self.user_df['rating'])]
            new_user_df_size = len(self.user_df.index)
            if new_user_df_size > 0:
                self.location_service = LocationGeocodingService()
                self.user_df.drop(['rating', 'loc'], axis=1, inplace=True)
                self.user_analyzer = UserRatingAnalyzer(grouped_repo_df=self.grouped_repo_df, mean_std_dict=self.mean_std_dict, repo_lang_mean_dict=self.language_mean_ratings, location_service=self.location_service)
                logger.info(f"Starting calculation for {new_user_df_size} users.")
                self.run_concurrent_executor()
                if self.user_analyzer.total_user_updated:
                    self.user_df.to_csv("user_before_normalization.csv", sep=',', encoding='utf-8')
                    if self.user_analyzer.total_user_updated == new_user_df_size:
                        logger.info(f"Rating calculation completed for all users. Total users reflected: {self.user_analyzer.total_user_updated}")
                    else:
                        logger.info(f"Rating calculation only completed for few users. Rerun to calculate for all. \n Users reflected: {self.user_analyzer.total_user_updated} \n Users yet to be calculated: {new_user_df_size - self.user_analyzer.total_user_updated}")
                else:
                    logger.error(f"Unable to calculate rating for any users. Debug and retry the execution.")
            else:
                logger.warning("No records without rating. Which means all users rating has been calculated.")
                    
        display_execution_time(run_start_time, "User rating calculation via User Rating Orchestrator completed")


    def normalize_user_ratings(self) -> None:
        """
        Normalize repository ratings.
        """
        min_max_lang_dict = {}
        # Loop through each language and print min_val and max_val
        for lang in set(lang for r in self.user_df['rating'] for lang in r if lang != 'general'):
            min_val = min([r.get(lang, {}).get('final_rating', 0) for r in self.user_df['rating'] if lang in r])
            max_val = max([r.get(lang, {}).get('final_rating', 0) for r in self.user_df['rating'] if lang in r])
            min_max_lang_dict.update({
                lang: {
                    "min": min_val,
                    "max": max_val
                }
            })

        logger.info(f"Min Max value of rating based on language: {min_max_lang_dict}")
        # Creating the 'n_rating' column
        self.user_df['n_rating'] = self.user_df['rating'].apply(lambda row: {
            lang: normalize_to_1(row.get(lang, {}).get('final_rating', 0),
                                min_val=min_max_lang_dict[lang]['min'],
                                max_val=min_max_lang_dict[lang]['max'],
                                delta=0) * 100
            for lang in set(lang for lang in row if lang != 'general')
        })


    def start_normalization(self) -> None:
        """Starts the orchestration of the user normalization of rating
        """
        run_start_time = datetime.now()
        total_users = self.fetch_users()
        self.normalize_user_ratings()
        updated_user_count = self.bulk_save_user_normalized_rating()
        if total_users == updated_user_count:
            logger.info(f"Normalization of user rating completed for all users. Total users: {total_users}")
        else:
            logger.info(f"Normalization happened only for few users. Total users: {total_users}, Completed: {updated_user_count}, Pending: {total_users - updated_user_count}")
        display_execution_time(run_start_time, "User normalized rating calculation via User Rating Orchestrator completed")

    @staticmethod
    def assign_proficiency(rating: float, thresholds: Dict[str, Dict[str, float]]) -> str:
        """assigns proficiency based on the provided rating

        Args:
            rating (float): rating value
            thresholds (Dict[str, Dict[str, float]]): threshold of each language

        Returns:
            str: proficiency the rating belongs to 
        """
        proficiency = {}
        for language, rating in rating.items():
            t1, t2 = thresholds[language]['t1'], thresholds[language]['t2']
            if rating <= t1:
                proficiency[language] = BEGINNER
            elif t1 < rating <= t2:
                proficiency[language] = INTERMEDIATE
            else:
                proficiency[language] = EXPERT
        return proficiency

    def start_proficiency_calculation(self) -> None:
        """Start assigning proficiency to the user
        """
        run_start_time = datetime.now()
        total_user_count = self.fetch_users()
        thresholds = {}

        # Extract unique languages from the DataFrame
        languages = self.user_df['n_rating'].apply(lambda x: list(x.keys())).explode().unique()

        # Assign proficiency levels to users
        for language in languages:
            ratings = self.user_df['n_rating'].apply(lambda x: x.get(language, np.nan)).dropna().values
            # Fit Gaussian Mixture Model
            gmm = GaussianMixture(n_components=len(PROFICIENCY), random_state=0)
            gmm.fit(ratings.reshape(-1, 1))
            # Get means as thresholds
            thresholds[language] = dict(zip(['t1', 't2'], sorted(gmm.means_.flatten())))
        # Apply the function to create the 'proficiency' column
        self.user_df['proficiency'] = self.user_df['n_rating'].apply(lambda x: UserRatingOrchestrator.assign_proficiency(x, thresholds))
        records_saved_count = self.bulk_save_user_proficiency()
        if total_user_count == records_saved_count:
            logger.info(f"Proficiency assignment of user completed for all users. Total users: {total_user_count}")
        else:
            logger.info(f"Proficiency assignment happened only for few users. Total users: {total_user_count}, Completed: {records_saved_count}, Pending: {total_user_count - records_saved_count}")
        display_execution_time(run_start_time, "User proficiency assignment via User Rating Orchestrator completed")
