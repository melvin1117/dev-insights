import asyncio
from functools import wraps
from log_config import LoggerConfig

# Initialize the logger for this module
logger = LoggerConfig(__name__).logger

def wait_and_retry(max_attempts=3, gap_between_calls_sec=1, allowed_exceptions: tuple = None, method_name=None):
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
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

        async def _helper(fallback_attempts, args):
            logger.info(f"Waiting for {gap_between_calls_sec} sec before retrying attempt number {fallback_attempts}, ARGS: {args}")
            await asyncio.sleep(gap_between_calls_sec)
            fallback_attempts += 1

        return wrapper

    return decorator
