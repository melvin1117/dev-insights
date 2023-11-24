from os import getenv
import time
from queue import Queue
from datetime import datetime, timedelta
from utils.api_exceptions_utils import APIException, RateLimitExceededException
from database.session import Session
from log_config import LoggerConfig
from utils.concurrent_executor import ConcurrentExecutor
from typing import Dict, Any, List, Union
from asset.constants import CONFIG_DATA, AUTH_HEADER_NAME, AUTH_BEARER, GITHUB
import random
from utils.api_utils import ApiUtils
from asset.api_endpoints import GITHUB_ENDPOINTS
from operator import itemgetter
from utils.helper_functions import wait_and_retry
import asyncio

# Initialize the logger for this module
logger = LoggerConfig(__name__).logger

MAX_FALLBACK_ATTEMPTS = int(getenv("MAX_FALLBACK_ATTEMPTS", 2))
GAP_BETWEEN_CALL_SEC = int(getenv("GAP_BETWEEN_CALL_SEC", 60))
FETCH_PAST_NUM_DAYS = int(getenv("FETCH_PAST_NUM_DAYS", 1100))
NUM_DAYS_CHUNK_SIZE = int(getenv("NUM_DAYS_CHUNK_SIZE", 7))
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
            logger.error(f"Error while loading github token, please check if GITHUB_API_KEYS is present. {err}")
            raise Exception(f"Error while loading github token, please check if GITHUB_API_KEYS is present. {err}")

        self.gh_token_queue = Queue()
        self.language_fetch_date_info = {}
        self.run_start_time = None
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
        self, repo_dict: Dict[str, Any], contributors: List[Dict[str, Any]], language: str
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
            "contributors": contributors,
            "added_at": self.run_start_time
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
            "repos_contributed": {language: [repo_id]},
            "added_at": self.run_start_time
        }

    def get_last_fetched_date(self, language: str) -> datetime:
        """
        Get the last fetched date for a given language.
        """
        with Session() as session:
            fallback_last_fetch_date = datetime.utcnow() - timedelta(days=FETCH_PAST_NUM_DAYS)
            try:
                last_fetched_data = session[GITHUB['tracker']].find_one({"language": language})
                if last_fetched_data:
                    if isinstance(last_fetched_data["last_fetched_date"], str):
                        return datetime.strptime(last_fetched_data["last_fetched_date"])
                    return last_fetched_data["last_fetched_date"]
                else:
                    return fallback_last_fetch_date
            except Exception as e:
                logger.error(f"Error in get_last_fetched_date: {e}. Returning default.")
                return fallback_last_fetch_date

    def update_last_fetched_date(
        self, language: str, last_fetched_date: datetime = None
    ) -> None:
        """
        Update the last fetched date for a given language.
        """
        with Session() as session:
            try:
                if not last_fetched_date:
                    last_fetched_date = self.get_last_fetched_date(language)  + timedelta(days=NUM_DAYS_CHUNK_SIZE + 1)
                session[GITHUB['tracker']].update_one(
                    {"language": language},
                    {"$set": {"last_fetched_date": last_fetched_date}},
                    upsert=True,
                )
                logger.info(f"{language}: Updated last fetched to {last_fetched_date}")
            except Exception as e:
                logger.error(f"{language}: Error in update_last_fetched_date while updating to -> {last_fetched_date}: {e}")

    def insert_one_record_to_db(self, collection: str, data: Dict[str, Any]) -> None:
        """
        Insert one record into the database.
        """
        with Session() as session:
            try:
                session[collection].insert_one(data)
                logger.debug(f"Data Inserted Successfully to {collection}")
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
        repos_contributed: Dict[str, List[int]]
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
                            "repos_contributed": repos_contributed,
                        }
                    },
                )
            except Exception as e:
                logger.error(f"An error occurred while finding to {collection}: {e}")

    @wait_and_retry(max_attempts=MAX_FALLBACK_ATTEMPTS, gap_between_calls_sec=GAP_BETWEEN_CALL_SEC, allowed_exceptions=(RateLimitExceededException), method_name='fetch_wrap_up')
    async def fetch_data(self, language: str) -> str:
        """
        Fetch data for a given language from GitHub API.

        Args:
            language (str): The programming language for which data needs to be fetched.

        Returns:
            str: A message indicating the completion of the data fetching process.
        """
        logger.info(f"{language}: Start fetching...")
        try:
            token = self.get_next_gh_token()
            default_headers = {
                AUTH_HEADER_NAME: f"{AUTH_BEARER} {token}",
                "Accept": "application/vnd.github+json",
                "User-Agent": getenv('APP_NAME')
            }
            api = ApiUtils(base_url=GITHUB_ENDPOINTS['base_url'], default_headers=default_headers)

            if language not in self.language_fetch_date_info:
                today = datetime.utcnow()
                last_fetched_date = self.get_last_fetched_date(language) + timedelta(days=1)  # start from next date
                if last_fetched_date > today:
                    last_fetched_date = today
                end_date = last_fetched_date + timedelta(days=NUM_DAYS_CHUNK_SIZE)
                if end_date > today:
                    end_date = today
                self.language_fetch_date_info[language] = {
                    'last_fetched_date': last_fetched_date,
                    'end_date': end_date
                }
            else:
                last_fetched_date = self.language_fetch_date_info[language]['last_fetched_date']
                end_date = self.language_fetch_date_info[language]['end_date']

            records_fetched = 0
            created_query = ""
            if end_date != last_fetched_date:
                created_query = f"created:{last_fetched_date.strftime('%Y-%m-%d')}..{end_date.strftime('%Y-%m-%d')}"
            else:
                created_query = f"created:>={last_fetched_date.strftime('%Y-%m-%d')}"
                
            query = f"language:{language} {created_query}"
            page = 1  # Start with page 1
            repo_query_params = {"q": query, "sort": "stars", "page": page}
            repositories = await api.get(endpoint = GITHUB_ENDPOINTS['search_repo'], params=repo_query_params)
            total_repos = repositories['total_count']
            logger.info(f'{language}: Searched query: {query} with total repo count as {total_repos}')
            while (total_repos < MAX_RECORD_PER_SESSION and  records_fetched <= total_repos) or (total_repos > MAX_RECORD_PER_SESSION and records_fetched <= MAX_RECORD_PER_SESSION):
                # Use the created filter and page parameter in the search query
                if page > 1:
                    repo_query_params['page'] = page
                    repositories = await api.get(endpoint = GITHUB_ENDPOINTS['search_repo'], params=repo_query_params)
                
                for repo in repositories['items']:
                    # Get all the contributors for the repo
                    contributors = await api.get(endpoint=repo['contributors_url'].replace(GITHUB_ENDPOINTS['base_url'], ""))
                    
                    contributor_keys_to_retain = ['id', 'contributions', 'type', 'site_admin']
                    itemgetter_keys = itemgetter(*contributor_keys_to_retain)

                    repo_data = self.get_formatted_repo_data(
                        repo_dict=repo,
                        contributors=list(map(lambda d: dict(zip(contributor_keys_to_retain, itemgetter_keys(d))), contributors)),
                        language=language,
                    )

                    # Save repo_data to the database immediately
                    existing_repo = self.find_one(GITHUB['repo'], "gid", repo_data["gid"])
                    if not existing_repo:
                        logger.debug(f"{language}: Storing repo data to db. Repo id: {repo['id']}")
                        self.insert_one_record_to_db(GITHUB['repo'], repo_data)

                    for contributor in contributors:
                        user = await api.get(endpoint=f"{GITHUB_ENDPOINTS['users']}/{contributor['login']}")

                        # Save user_data to the database immediately
                        logger.debug(f"{language}: Checking if user exists in db. Repo id: {repo['id']} User id:{user['id']}")
                        existing_user = self.find_one(GITHUB['user'], 'gid', user['id'])
                        logger.debug(f"{language}: Storing user data to db. Repo id: {repo['id']} User id:{user['id']}")
                        if existing_user:
                            existing_repo_data = existing_user['repos_contributed']
                            if language in existing_repo_data:
                                repo_ids_set = set(existing_repo_data[language])
                                repo_ids_set.add(repo['id'])
                                existing_repo_data[language] = list(repo_ids_set)
                            else:
                                existing_repo_data[language] = [repo['id']]
                            self.update_one_gh_user(GITHUB['user'], 'gid', user['id'], existing_repo_data)
                        else:
                            user_data = self.get_formatted_user_data(user_dict=user, language=language, repo_id=repo['id'])
                            self.insert_one_record_to_db(GITHUB['user'], user_data)

                records_fetched += len(repositories['items'])

                # Increment the page for the next iteration
                page += 1
                logger.info(f"{language}: Page incremented to {page}. records fetch till now is {records_fetched}")
        except RateLimitExceededException as e:
            if e.status == 403:
                logger.warning(f"{language}: (Retry) Rate Limit Exceeded. with {e}")
                raise RateLimitExceededException(f"Rate Limit Exceeded. {e}")
            logger.error(f"{language}: (No Retry) Rate Limit Exceeded.. {e}")
            raise APIException(f"Rate Limit Exceeded.. {e}")
        except APIException as e:
            logger.error(f"{language}: API Exception. {e.status} - {e}")
            raise
        except Exception as e:
            logger.error(f"{language}: Unexpected error: {e}")
            raise
        finally:
            logger.info(f"{language or 'NO_LANG'}: In finally. Records fetched: {records_fetched or 0}. Query: {query or ''}")
            if token:
                logger.info(f"{language}: Token released.")
                self.release_gh_token(token)

        return language

    def fetch_wrap_up(self, language: str) -> None:
        """Executes things after fetch for a language is completed

        Args:
            language (str): Language name for which fetch is completed
        """
        # Update the last fetched date for the language in db
        if language in self.language_fetch_date_info:
            self.update_last_fetched_date(language, self.language_fetch_date_info[language]['end_date'])
        else:
            self.update_last_fetched_date(language)

    def process_language(self, language) -> str:
         result = asyncio.run(self.fetch_data(language))
         return result

    def start(self) -> None:
        """
        Start the data mining process for multiple languages concurrently.
        """
        time.sleep(60) # to make sure dependent services are up
        self.run_start_time = datetime.now()
        # languages = ["Python", "JavaScript", "Java", "Rust"]
        languages = list(CONFIG_DATA["languages"].keys())
        # shuffle the languages so that sequence of execution is different and no priority is given to a language
        random.shuffle(languages)
        concurrent_exec = ConcurrentExecutor(languages, len(self.gh_tokens), self.process_language)
        concurrent_exec.start()
        
        end_time = datetime.now()
        time_difference = end_time - self.run_start_time
        hours = time_difference.seconds // 3600
        minutes = (time_difference.seconds // 60) % 60
        seconds = time_difference.seconds % 60
        
        logger.info(f"Completed from GitHub Mining for Languages: {self.language_fetch_date_info.keys()}")
        logger.info(f"Time taken to execute the mining {hours} hours, {minutes} minutes, and {seconds} seconds.")
