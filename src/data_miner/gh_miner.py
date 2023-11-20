from os import getenv
import time
from queue import Queue
from datetime import datetime, timedelta
from github import (
    Github,
    GithubException,
    RateLimitExceededException,
    UnknownObjectException,
)
from database.session import Session
from log_config import LoggerConfig
from data_miner.concurrent_executor import ConcurrentExecutor
from functools import wraps
from typing import Dict, Any, List, Union
from asset.constants import CONFIG_DATA
import random

# Initialize the logger for this module
logger = LoggerConfig(__name__).logger

MAX_FALLBACK_ATTEMPTS = int(getenv("MAX_FALLBACK_ATTEMPTS", 2))
GAP_BETWEEN_CALL_SEC = int(getenv("GAP_BETWEEN_CALL_SEC", 60))
FETCH_PAST_NUM_DAYS = int(getenv("FETCH_PAST_NUM_DAYS", 1100))
NUM_DAYS_CHUNK_SIZE = int(getenv("NUM_DAYS_CHUNK_SIZE", 30))
MAX_PAGE_PER_SESSION = int(getenv("MAX_PAGE_PER_SESSION", 10))
MAX_RECORD_PER_SESSION = int(getenv("MAX_RECORD_PER_SESSION", 150))


class GitHubDataMiner:
    def __init__(self) -> None:
        """
        Initialize GitHubDataMiner instance.
        """
        try:
            self.gh_tokens = [
                str(item) for item in getenv("GITHUB_API_KEYS").split(",")
            ]
        except Exception as err:
            logger.error(
                f"Error while loading github token, please check if GITHUB_API_KEYS is present. {err}"
            )
            raise Exception(
                f"Error while loading github token, please check if GITHUB_API_KEYS is present. {err}"
            )

        self.gh_token_queue = Queue()
        self.processed_languages = []
        for token in self.gh_tokens:
            self.gh_token_queue.put(token)

    def get_next_gh_token(self) -> str:
        """
        Get the next GitHub token from the queue.
        """
        return self.gh_token_queue.get()

    def release_gh_token(self, token: str) -> None:
        """
        Release the GitHub token back to the queue.
        """
        self.gh_token_queue.put(token)

    def get_formatted_repo_data(
        self, repo_dict: Dict[str, Any], contributors_gid: List[int], language: str
    ) -> Dict[str, Any]:
        """
        Get formatted repository data.
        """
        return {
            "gid": repo_dict.get("id"),
            "name": repo_dict.get("name"),
            "full_name": repo_dict.get("full_name", ""),
            "private": repo_dict.get("private", False),
            "owner_gid": repo_dict["owner"].get("id"),
            "description": repo_dict.get("description", ""),
            "created_at": repo_dict.get("created_at"),
            "updated_at": repo_dict.get("updated_at"),
            "pushed_at": repo_dict.get("pushed_at"),
            "size": repo_dict.get("size", -1),
            "stargazers_count": repo_dict.get("stargazers_count", -1),
            "watchers_count": repo_dict.get("watchers_count", -1),
            "language": language,
            "has_issues": repo_dict.get("has_issues", False),
            "has_projects": repo_dict.get("has_projects", False),
            "has_downloads": repo_dict.get("has_downloads", False),
            "has_wiki": repo_dict.get("has_wiki", False),
            "has_pages": repo_dict.get("has_pages", False),
            "has_discussions": repo_dict.get("has_discussions", False),
            "forks_count": repo_dict.get("forks_count", -1),
            "archived": repo_dict.get("archived", False),
            "disabled": repo_dict.get("disabled", False),
            "open_issues_count": repo_dict.get("open_issues_count", -1),
            "license": repo_dict.get("license", {}),
            "allow_forking": repo_dict.get("allow_forking", False),
            "is_template": repo_dict.get("is_template", False),
            "web_commit_signoff_required": repo_dict.get(
                "web_commit_signoff_required", False
            ),
            "topics": repo_dict.get("topics", []),
            "visibility": repo_dict.get("visibility", ""),
            "default_branch": repo_dict.get("default_branch", ""),
            "score": repo_dict.get("score", -1),
            "contributors_gid": contributors_gid,
        }

    def get_formatted_user_data(
        self, user_dict: Dict[str, Any], language: str, repo_id: int
    ) -> Dict[str, Any]:
        """
        Get formatted user data.
        """
        return {
            "login": user_dict.get("login"),
            "name": user_dict.get("name"),
            "gid": user_dict.get("id"),
            "blog": user_dict.get("blog", ""),
            "avatar_url": user_dict.get("avatar_url", ""),
            "location": user_dict.get("location", ""),
            "email": user_dict.get("email", ""),
            "hireable": user_dict.get("hireable", False),
            "bio": user_dict.get("bio", ""),
            "twitter_username": user_dict.get("twitter_username", ""),
            "public_repos": user_dict.get("public_repos", -1),
            "public_gists": user_dict.get("public_gists", -1),
            "followers": user_dict.get("followers", -1),
            "following": user_dict.get("following", -1),
            "created_at": user_dict.get("created_at"),
            "updated_at": user_dict.get("updated_at"),
            "private_gists": user_dict.get("private_gists", -1),
            "total_private_repos": user_dict.get("total_private_repos", -1),
            "owned_private_repos": user_dict.get("owned_private_repos", -1),
            "disk_usage": user_dict.get("disk_usage", -1),
            "collaboration_count": user_dict.get("collaborators", -1),
            "repo_contributed_gid": [repo_id],
            "languages_contributed": [language],
        }

    def get_last_fetched_date(self, language: str) -> datetime:
        """
        Get the last fetched date for a given language.
        """
        with Session() as session:
            try:
                last_fetched_data = session["gh-tracker"].find_one(
                    {"language": language}
                )
                if last_fetched_data:
                    return last_fetched_data["last_fetched_date"]
                else:
                    logger.info(
                        f"Get last fetched {language}: {datetime.utcnow() - timedelta(days=FETCH_PAST_NUM_DAYS)}"
                    )
                    return datetime.utcnow() - timedelta(days=FETCH_PAST_NUM_DAYS)
            except Exception as e:
                logger.error(f"Error in get_last_fetched_date: {e}. Returning default.")
                return datetime.utcnow() - timedelta(days=FETCH_PAST_NUM_DAYS)

    def update_last_fetched_date(
        self, language: str, last_fetched_date: datetime
    ) -> None:
        """
        Update the last fetched date for a given language.
        """
        with Session() as session:
            try:
                session["gh-tracker"].update_one(
                    {"language": language},
                    {"$set": {"last_fetched_date": last_fetched_date}},
                    upsert=True,
                )
                logger.info(f"Updated last fetched {language}: {last_fetched_date}")
            except Exception as e:
                logger.error(
                    f"Error in update_last_fetched_date for {language} -> {last_fetched_date}: {e}"
                )

    def insert_one_record_to_db(self, collection: str, data: Dict[str, Any]) -> None:
        """
        Insert one record into the database.
        """
        with Session() as session:
            try:
                session[collection].insert_one(data)
                logger.info(f"Data Inserted Successfully to {collection}")
            except Exception as e:
                logger.error(f"An error occurred while inserting to {collection}: {e}")

    def find_one(
        self, collection: str, key: str, data: Any
    ) -> Union[Dict[str, Any], None]:
        """
        Find one record in the database.
        """
        with Session() as session:
            try:
                return session[collection].find_one({key: data})
            except Exception as e:
                logger.error(f"An error occurred while finding to {collection}: {e}")

    def update_one_gh_user(
        self,
        collection: str,
        key: str,
        data: Any,
        existing_languages: List[str],
        existing_repos: List[int],
    ) -> None:
        """
        Update one GitHub user record in the database.
        """
        with Session() as session:
            try:
                session[collection].update_one(
                    {key: data},
                    {
                        "$set": {
                            "languages_contributed": list(existing_languages),
                            "repo_contributed_gid": list(existing_repos),
                        }
                    },
                )
            except Exception as e:
                logger.error(f"An error occurred while finding to {collection}: {e}")

    def wait_and_retry(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            fallback_attempts = 0
            while fallback_attempts < MAX_FALLBACK_ATTEMPTS:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error: {e}, {fallback_attempts}")
                    logger.info(
                        f"Waiting for {GAP_BETWEEN_CALL_SEC} sec before retrying  {fallback_attempts}"
                    )
                    time.sleep(GAP_BETWEEN_CALL_SEC)
                    fallback_attempts += 1
            raise Exception(f"Max fallback attempts reached. Unable to recover.")

        return wrapper

    @wait_and_retry
    def fetch_data(self, language: str) -> str:
        """
        Fetch data for a given language from GitHub API.

        Args:
            language (str): The programming language for which data needs to be fetched.

        Returns:
            str: A message indicating the completion of the data fetching process.
        """
        if language in self.processed_languages:
            logger.warning(f"Stopping {language} from executing twice")
            return f"Stopping {language} from executing twice"
        logger.info(f"Start fetching for {language}")
        try:
            token = self.get_next_gh_token()
            github_instance = Github(token)

            today = datetime.utcnow()
            last_fetched_date = self.get_last_fetched_date(language) + timedelta(
                days=1
            )  # start from next date
            if last_fetched_date > today:
                last_fetched_date = today
            end_date = last_fetched_date + timedelta(days=NUM_DAYS_CHUNK_SIZE)
            if end_date > today:
                end_date = today

            remaining_records = MAX_RECORD_PER_SESSION
            created_query = ""
            if end_date != last_fetched_date:
                created_query = f"created:{last_fetched_date.strftime('%Y-%m-%d')}..{end_date.strftime('%Y-%m-%d')}"
            query = f"language:{language} {created_query}"
            page = 1  # Start with page 1
            while remaining_records > 0 and page <= MAX_PAGE_PER_SESSION:
                # Use the created filter and page parameter in the search query
                logger.info(query)
                repositories = github_instance.search_repositories(
                    query=query, sort="stars", page=page
                )
                for repo in repositories:
                    contributors = repo.get_contributors()
                    contributors_gid = [c.id for c in contributors]
                    repo_data = self.get_formatted_repo_data(
                        repo_dict=repo._rawData,
                        contributors_gid=contributors_gid,
                        language=language,
                    )

                    # Save repo_data to the database immediately
                    logger.info(f"Storing repo data to db {language} {repo.id}")
                    existing_repo = self.find_one("gh-repo", "gid", repo_data["gid"])
                    if not existing_repo:
                        self.insert_one_record_to_db("gh-repo", repo_data)
                    logger.info(
                        f"num of contributors for {language} {repo.id} {len(contributors_gid)}"
                    )
                    for contributor in contributors:
                        user = github_instance.get_user_by_id(contributor.id)
                        user_data = self.get_formatted_user_data(
                            user_dict=user._rawData, language=language, repo_id=repo.id
                        )

                        # Save user_data to the database immediately
                        logger.info(
                            f'Checking if user exists in db  {language} {repo.id} {user_data["gid"]}'
                        )
                        logger.info(
                            f'Storing user data to db  {language} {repo.id} {user_data["gid"]}'
                        )
                        existing_user = self.find_one(
                            "gh-users", "gid", user_data["gid"]
                        )
                        if existing_user:
                            existing_languages = set(
                                existing_user["languages_contributed"]
                            )
                            existing_languages.add(
                                user_data["languages_contributed"][0]
                            )
                            existing_repos = set(existing_user["repo_contributed_gid"])
                            existing_repos.add(user_data["repo_contributed_gid"][0])
                            self.update_one_gh_user(
                                "gh-users",
                                "gid",
                                user_data["gid"],
                                existing_languages,
                                existing_repos,
                            )
                        else:
                            self.insert_one_record_to_db("gh-users", user_data)

                remaining_records -= repositories.totalCount

                if remaining_records > 0:
                    # If there are more records to fetch, wait for the 1-minute gap
                    logger.info(
                        f"Waiting for {GAP_BETWEEN_CALL_SEC} sec before fetching more records  {language} {page} {remaining_records}"
                    )
                    time.sleep(GAP_BETWEEN_CALL_SEC)

                # Increment the page for the next iteration
                page += 1
                logger.info(
                    f"page incremented for {language} to {page} {remaining_records}"
                )

        except RateLimitExceededException as rate_limit_exceeded:
            reset_time = datetime.utcfromtimestamp(
                rate_limit_exceeded.rate.reset
            ).strftime("%Y-%m-%d %H:%M:%S UTC")
            logger.error(
                f"Rate limit exceeded. Waiting until {reset_time} before retrying  {language} "
            )
            time.sleep(
                rate_limit_exceeded.rate.remaining + 5
            )  # Extra 5 seconds to be safe
            raise
        except UnknownObjectException as unknown_object_ex:
            logger.error(f"Unknown object exception: {unknown_object_ex} {language} ")
            raise
        except GithubException as github_ex:
            logger.error(f"GitHub API exception: {github_ex} {language} ")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e} {language} ")
            raise
        finally:
            if token:
                self.release_gh_token(token)
            logger.info(f"Releasing token for {language} {page} {remaining_records}")
            if remaining_records and remaining_records != MAX_RECORD_PER_SESSION:
                self.processed_languages.append(language)
                # Update the last fetched date for the language in db
                self.update_last_fetched_date(language, end_date)
            time.sleep(10)
        return language

    def start(self) -> None:
        """
        Start the data mining process for multiple languages concurrently.
        """

        languages = list(CONFIG_DATA["languages"].keys())
        # shuffle the languages so that sequence of execution is different and no priority is given to a language
        random.shuffle(languages)
        concurrent_exec = ConcurrentExecutor(
            languages, len(self.gh_tokens) - 1, self.fetch_data
        )
        concurrent_exec.start()
        print("Completed from GitHub Miner...")
