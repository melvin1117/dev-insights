from os import getenv
from queue import Queue
import time
from github import Github, GithubException, RateLimitExceededException, UnknownObjectException
from datetime import datetime, timedelta
from database.session import Session
from log_config import LoggerConfig
from data_miner.concurrent_executor import ConcurrentExecutor

# Initialize the logger for this module
logger = LoggerConfig(__name__).logger

MAX_FALLBACK_ATTEMPTS = int(getenv('MAX_FALLBACK_ATTEMPTS'))
GAP_BETWEEN_CALL_SEC = int(getenv('GAP_BETWEEN_CALL_SEC'))

class GitHubDataMiner:
    def __init__(self):
        self.gh_tokens = [str(item) for item in getenv('GITHUB_API_KEYS').split(",")]
        self.max_record_per_session = int(getenv('MAX_RECORD_PER_SESSION'))
        self.num_records_fetched = int(getenv('NUM_RECORDS_FETCHED'))
        self.gh_token_queue = Queue()
        for token in self.gh_tokens:
            self.gh_token_queue.put(token)

    def get_next_gh_token(self):
        return self.gh_token_queue.get()

    def release_gh_token(self, token: str):
        self.gh_token_queue.put(token)

    @staticmethod
    def get_formatted_repo_data(repo_dict, contributors_gid, language):
        return {
            "gid": repo_dict.get('id'),
            "name": repo_dict.get('name'),
            "full_name": repo_dict.get('full_name',''),
            "private": repo_dict.get('private', False),
            "owner_gid": repo_dict['owner'].get('id'),
            "description": repo_dict.get('description',''),
            "created_at": repo_dict.get('created_at'),
            "updated_at": repo_dict.get('updated_at'),
            "pushed_at": repo_dict.get('pushed_at'),
            "size": repo_dict.get('size', -1),
            "stargazers_count": repo_dict.get('stargazers_count', -1),
            "watchers_count": repo_dict.get('watchers_count', -1),
            "language": repo_dict.get('language', language),
            "has_issues": repo_dict.get('has_issues', False),
            "has_projects": repo_dict.get('has_projects', False),
            "has_downloads": repo_dict.get('has_downloads', False),
            "has_wiki": repo_dict.get('has_wiki', False),
            "has_pages": repo_dict.get('has_pages', False),
            "has_discussions": repo_dict.get('has_discussions', False),
            "forks_count": repo_dict.get('forks_count', -1),
            "archived": repo_dict.get('archived', False),
            "disabled": repo_dict.get('disabled', False),
            "open_issues_count": repo_dict.get('open_issues_count', -1),
            "allow_forking": repo_dict.get('allow_forking', False),
            "is_template": repo_dict.get('is_template', False),
            "web_commit_signoff_required": repo_dict.get('web_commit_signoff_required', False),
            "topics": repo_dict.get('topics', []),
            "visibility": repo_dict.get('visibility',''),
            "default_branch": repo_dict.get('default_branch',''),
            "score": repo_dict.get('score', -1),
            "contributors_gid": contributors_gid
        }

    @staticmethod
    def get_formatted_user_data(user_dict, language, repo_id):
        return {
            "login": user_dict.get('login'),
            "name": user_dict.get('name'),
            "gid": user_dict.get('id'),
            "blog": user_dict.get('blog', ''),
            "avatar_url": user_dict.get('avatar_url', ''),
            "location": user_dict.get('location', ''),
            "email": user_dict.get('email', ''),
            "hireable": user_dict.get('hireable', False),
            "bio": user_dict.get('bio', ''),
            "twitter_username": user_dict.get('twitter_username', ''),
            "public_repos": user_dict.get('public_repos', -1),
            "public_gists": user_dict.get('public_gists', -1),
            "followers": user_dict.get('followers', -1),
            "following": user_dict.get('following', -1),
            "created_at": user_dict.get('created_at'),
            "updated_at": user_dict.get('updated_at'),
            "private_gists": user_dict.get('private_gists', -1),
            "total_private_repos": user_dict.get('total_private_repos', -1),
            "owned_private_repos": user_dict.get('owned_private_repos', -1),
            "disk_usage": user_dict.get('disk_usage', -1),
            "collaboration_count": user_dict.get('collaborators', -1),
            "repo_contributed_gid": [repo_id],
            "languages_contributed": [language]
        }

    def get_last_fetched_date(self, language):
        with Session() as session:
            try:
                last_fetched_data = session['gh-tracker'].find_one({"language": language})
                if last_fetched_data:
                    return last_fetched_data["last_fetched_date"]
                else:
                    # If no record found, return the past 6 months date
                    # return datetime.utcnow() - timedelta(days=180)
                    print(f'Get last fetched {language}: {datetime.utcnow() - timedelta(days=180)}\n', "#" * 30)
                    return datetime.utcnow() - timedelta(days=180)
            except Exception as e:
                print(f"Error in get_last_fetched_date: {e}\n")
                # Handle the error, you may choose to return a default date or raise an exception

    def update_last_fetched_date(self, language: str, last_fetched_date):
        print(f'Updated last fetched {language}: {last_fetched_date}\n', "#" * 30)
        with Session() as session:
            try:
                session['gh-tracker'].update_one(
                    {"language": language},
                    {"$set": {"last_fetched_date": last_fetched_date}},
                    upsert=True
                )
                print(f'Updated last fetched {language}: {last_fetched_date}', "#" * 30)
            except Exception as e:
                print(f"Error in update_last_fetched_date: {e}")
                # Handle the error, you may choose to log the error or raise an exception

    def insert_one_record_to_db(self, collection, data):
        with Session() as session:
            try:
                session[collection].insert_one(data)
                logger.info(f"Data Inserted Successfully to {collection}")
            except Exception as e:
                logger.error(f"An error occurred while inserting to {collection}: {e}")

    def find_one(self, collection, key, data):
        with Session() as session:
            try:
                return session[collection].find_one({key: data})
            except Exception as e:
                logger.error(f"An error occurred while finding to {collection}: {e}")

    def update_one_gh_user(self, collection, key, data, existing_languages, existing_repos):
        with Session() as session:
            try:
                session[collection].update_one(
                    {key: data},
                    {
                        "$set": {"languages_contributed": list(existing_languages), "repo_contributed_gid": list(existing_repos)}
                    },
                )
            except Exception as e:
                logger.error(f"An error occurred while finding to {collection}: {e}")

    def wait_and_retry(self, func):
        def wrapper(self, *args, **kwargs):
            fallback_attempts = 0
            while fallback_attempts < MAX_FALLBACK_ATTEMPTS:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Error: {e}, {fallback_attempts}")
                    print(f"Waiting for {GAP_BETWEEN_CALL_SEC} sec before retrying... {fallback_attempts}")
                    time.sleep(GAP_BETWEEN_CALL_SEC)
                    fallback_attempts += 1
            raise Exception(f"Max fallback attempts reached. Unable to recover.")
        return wrapper


    @wait_and_retry
    def fetch_data(self, language: str) -> str:
        print(f'Start fetching for {language}')
        try:
            token = self.get_next_gh_token()
            github_instance = Github(token)

            last_fetched_date = self.get_last_fetched_date(language)
            end_date = last_fetched_date + timedelta(days=30)

            remaining_records = self.max_record_per_session
            page = 1  # Start with page 1
            while remaining_records > 0:
                # Use the created filter and page parameter in the search query
                query = f"language:{language} created:{last_fetched_date.strftime('%Y-%m-%d')}..{end_date.strftime('%Y-%m-%d')}"
                print(query)
                repositories = github_instance.search_repositories(query=query, sort='stars', page=page)
                num_fetched = self.num_records_fetched
                for repo in repositories:
                    contributors = repo.get_contributors()
                    contributors_gid = [c.id for c in contributors]
                    repo_data = self.get_formatted_repo_data(repo_dict=repo._rawData, contributors_gid=contributors_gid, language=language)

                    # Save repo_data to the database immediately
                    print(f"Storing repo data to db...{language} {repo.id}\n", "=" * 30)
                    existing_repo = self.find_one('gh-repo', 'gid', repo_data["gid"])
                    if not existing_repo:
                        self.insert_one_record_to_db('gh-repo', repo_data)
                    print(f'num of contributors for {language} {repo.id} {len(contributors_gid)}')
                    for contributor in contributors:
                        user = github_instance.get_user_by_id(contributor.id)
                        user_data = self.get_formatted_user_data(user_dict=user._rawData, language=language, repo_id=repo.id)

                        # Save user_data to the database immediately
                        print(f'Checking if user exists in db  {language} {repo.id} {user_data["gid"]}...\n', "-" * 30)
                        print(f'Storing user data to db  {language} {repo.id} {user_data["gid"]}...\n', "+" * 30)
                        existing_user = self.find_one('gh-users', 'gid', user_data["gid"])
                        if existing_user:
                            existing_languages = set(existing_user["languages_contributed"])
                            existing_languages.add(user_data["languages_contributed"][0])
                            existing_repos = set(existing_user["repo_contributed_gid"])
                            existing_repos.add(user_data["repo_contributed_gid"][0])
                            self.update_one_gh_user('gh-users', 'gid', user_data["gid"], existing_languages, existing_repos)
                        else:
                            self.insert_one_record_to_db('gh-users', user_data)

                remaining_records -= num_fetched

                if remaining_records > 0:
                    # If there are more records to fetch, wait for the 1-minute gap
                    print(f"Waiting for {GAP_BETWEEN_CALL_SEC} sec before fetching more records... {language} {page} {remaining_records}\n")
                    time.sleep(GAP_BETWEEN_CALL_SEC)

                # Increment the page for the next iteration
                page += 1
                print(f'page incremented for {language} to {page} {remaining_records}\n')

        except RateLimitExceededException as rate_limit_exceeded:
            reset_time = datetime.utcfromtimestamp(rate_limit_exceeded.rate.reset).strftime('%Y-%m-%d %H:%M:%S UTC')
            print(f"Rate limit exceeded. Waiting until {reset_time} before retrying... {language} \n")
            time.sleep(rate_limit_exceeded.rate.remaining + 5)  # Extra 5 seconds to be safe
            raise
        except UnknownObjectException as unknown_object_ex:
            print(f"Unknown object exception: {unknown_object_ex} {language} \n")
            raise
        except GithubException as github_ex:
            print(f"GitHub API exception: {github_ex} {language} \n")
            raise
        except Exception as e:
            print(f"Unexpected error: {e} {language} \n")
            raise
        finally:
            self.release_gh_token(token)
            print(f'Releasing token for {language} {page} {remaining_records}\n')
            # Update the last fetched date for the language in db
            self.update_last_fetched_date(language, end_date + timedelta(days=30))
        return language
    
    def test(self, name):
        print(f"test fun called for {name}")

    def start(self):
        languages = ["Python", "Java", "JavaScript", "Ruby"]
        concurrent_exec = ConcurrentExecutor(languages, len(self.gh_tokens) - 1, self.fetch_data)
        concurrent_exec.start()