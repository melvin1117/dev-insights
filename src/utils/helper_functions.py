import asyncio
from functools import wraps
from log_config import LoggerConfig
from datetime import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from langdetect import detect
import spacy
from spacy.cli.download import download
import math
from os import getenv
from typing import Optional, Tuple, Callable, Any

nltk.download('vader_lexicon')

# Initialize the logger for this module
logger = LoggerConfig(__name__).logger

RATING_DELTA: float = float(getenv('RATING_DELTA', 0.02))


def display_execution_time(run_start_time: datetime, msg: str = "Completed running") -> None:
    """
    Display the time taken for execution.

    Args:
        run_start_time (datetime): The start time of the execution.
        msg (str): message to display
    """
    end_time = datetime.now()
    time_difference = end_time - run_start_time
    hours = time_difference.seconds // 3600
    minutes = (time_difference.seconds // 60) % 60
    seconds = time_difference.seconds % 60

    logger.info(f"{msg} {hours} hr {minutes} min {seconds} sec")


def normalize_to_1(value: float, min_val: float, max_val: float, delta: float = RATING_DELTA) -> float:
    """
    Normalize a value to the range [0, 1].

    Args:
        value (float): The value to be normalized.
        min_val (float): Minimum value in the range.
        max_val (float): Maximum value in the range.
        delta (float): Adjustment factor for zero values.

    Returns:
        float: The normalized value.
    """
    if math.isnan(value):
        value = min_val

    if max_val == min_val:
        return abs(value) if value else delta

    min_val, max_val = (
        min_val - delta if min_val >= 0 else min_val + delta,
        max_val + delta if max_val >= 0 else max_val - delta,
    )

    normalized_value = abs((value - min_val) / (max_val - min_val))
    return normalized_value


def calculate_recency_factor(transaction_date: datetime) -> float:
    """
    Calculate recency factor based on the transaction date.

    Args:
        transaction_date (datetime): The date of the transaction.

    Returns:
        float: The calculated recency factor.
    """
    number_of_days = (datetime.now() - transaction_date).days
    recency_factor = max(0, min(1, 1 - (number_of_days / 365)))
    return recency_factor


def calculate_sentiment_with_emoticons(text: str) -> float:
    """
    Calculate sentiment score using the SentimentIntensityAnalyzer.

    Args:
        text (str): The input text.

    Returns:
        float: The sentiment score.
    """
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']
    return sentiment_score


def load_language_model(language_code: str) -> spacy.language.Language:
    """
    Load the language model based on the language code.

    Args:
        language_code (str): The language code.

    Returns:
        spacy.language.Language: The loaded language model.
    """
    language_mapping = {
        'en': 'en_core_web_sm',
        'fr': 'fr_core_news_sm',
        'es': 'es_core_news_sm',
        'de': 'de_core_news_sm',
        'zh-cn': 'zh_core_web_sm',
        'zh-tw': 'zh_core_web_sm',
        'ja': 'ja_core_news_sm',
        'it': 'it_core_news_sm',
        'nl': 'nl_core_news_sm',
        'da': 'da_core_news_sm',
        'ca': 'ca_core_news_sm',
        'ro': 'ro_core_news_sm',
        'pt': 'pt_core_news_sm',
        'lt': 'pt_core_news_sm',
        # Add more languages as needed
    }

    language_model = language_mapping.get(language_code.lower(), 'en_core_web_sm')

    try:
        return spacy.load(language_model)
    except OSError:
        logger.warning(f"Language model '{language_model}' not found. Attempting to download...")
        download(language_model)
        return spacy.load(language_model)


def detect_language(text: str) -> str:
    """
    Detect the language of the input text.

    Args:
        text (str): The input text.

    Returns:
        str: The detected language code.
    """
    try:
        lang = detect(text)
    except:
        lang = 'en'
    return lang


def z_score_normalization(value: float, mean: float, std_dev: float) -> float:
    """
    Perform z-score normalization.

    Args:
        value (float): The value to be normalized.
        mean (float): Mean of the distribution.
        std_dev (float): Standard deviation of the distribution.

    Returns:
        float: The normalized value.
    """
    return abs((value - mean) / std_dev) if std_dev != 0 else 0.0


def wait_and_retry(
    max_attempts: int = 3,
    gap_between_calls_sec: int = 1,
    allowed_exceptions: Optional[Tuple[Exception, ...]] = None,
    method_name: Optional[str] = None,
) -> Callable:
    """
    Decorator for retrying a function with a specified gap between calls.

    Args:
        max_attempts (int): Maximum number of retry attempts.
        gap_between_calls_sec (int): Gap (in seconds) between retry attempts.
        allowed_exceptions (Tuple[Exception, ...]): Tuple of exceptions to catch and retry.
        method_name (str): Name of the method to be called on successful retry.

    Returns:
        Callable: The decorated function.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> None:
            fallback_attempts = 0
            success = False
            while fallback_attempts < max_attempts:
                try:
                    result = await func(self, *args, **kwargs)
                except Exception as e:
                    if allowed_exceptions is not None and isinstance(e, allowed_exceptions):
                        await _helper(fallback_attempts, *args)
                    elif allowed_exceptions is not None:
                        logger.error(f"Error: {e} is not part of allowed exceptions. No retry will be attempted.")
                        break
                    else:
                        # Retrying all types of exceptions as none are provided
                        await _helper(fallback_attempts, *args)
                else:
                    success = True
                    break

            if success:
                if method_name is not None:
                    final_exec_fn = getattr(self, method_name)
                    await final_exec_fn(result)
            else:
                logger.error("All retry attempts failed. Unable to recover.")

        async def _helper(fallback_attempts: int, args: Tuple) -> None:
            logger.info(
                f"Waiting for {gap_between_calls_sec} sec before retrying attempt number {fallback_attempts}, ARGS: {args}"
            )
            await asyncio.sleep(gap_between_calls_sec)
            fallback_attempts += 1

        return wrapper

    return decorator
