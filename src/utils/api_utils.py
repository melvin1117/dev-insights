import aiohttp
from utils.api_exceptions_utils import (
    APIException,
    UnauthorizedException,
    BadCredentialsException,
    BadUserAgentException,
    RateLimitExceededException,
    UnknownObjectException,
)
from log_config import LoggerConfig

# Initialize the logger for this module
logger = LoggerConfig(__name__).logger

class ApiUtils:
    def __init__(self, base_url, default_headers=None):
        self.base_url = base_url
        self.default_headers = default_headers or {}

    async def _make_request(
        self, method, endpoint, params=None, headers=None, data=None
    ):
        url = f"{self.base_url}{endpoint}"
        final_headers = {**self.default_headers, **(headers or {})}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method, url, params=params, headers=final_headers, json=data
                ) as response:
                    response_data = await response.json()

                    if 400 <= response.status < 600:
                        self._handle_error(response.status, response_data, response.headers, response.history)


                    return response_data
        except aiohttp.ClientError as e:
            logger.error(f"Client Error during API request: {e}")
            raise  # Re-raise the exception
        except Exception as e:
            logger.error(f"Error during API request: {e}")
            raise  # Re-raise the exception

    def _handle_error(self, status, response_data, headers, history):
        response_message = response_data.get("message", "").lower()
        if status == 401:
            raise UnauthorizedException(response_message or "Unauthorized", status, headers=headers, response_data=response_data, history=history)
        elif status == 403 and any(
            msg in response_message
            for msg in ["bad user agent", "missing or invalid user agent string"]
        ):
            raise BadUserAgentException(response_message or "Bad User Agent", status, headers=headers, response_data=response_data, history=history)
        elif status == 403 and self.isRateLimitError(response_message):
            raise RateLimitExceededException(response_message or "Rate Limit Exceeded", status, headers=headers, response_data=response_data, history=history)
        elif status == 403:
            raise BadCredentialsException(response_message or "Bad Credentials", status, headers=headers, response_data=response_data, history=history)
        elif status == 429:
            raise RateLimitExceededException(response_message or "Rate Limit Exceeded", status, headers=headers, response_data=response_data, history=history)
        elif status == 404:
            raise UnknownObjectException(response_message or "Unknown Object", status, headers=headers, response_data=response_data, history=history)
        else:
            raise APIException(f"Unexpected API response: {status} - {response_data}", status, headers=headers, response_data=response_data, history=history)

    async def get(self, endpoint, params=None, headers=None):
        return await self._make_request("GET", endpoint, params=params, headers=headers)

    async def post(self, endpoint, data=None, headers=None):
        return await self._make_request("POST", endpoint, headers=headers, data=data)

    async def put(self, endpoint, data=None, headers=None):
        return await self._make_request("PUT", endpoint, headers=headers, data=data)

    async def delete(self, endpoint, headers=None):
        return await self._make_request("DELETE", endpoint, headers=headers)
    
    @classmethod
    def isRateLimitError(cls, message: str) -> bool:
        return cls.isPrimaryRateLimitError(message) or cls.isSecondaryRateLimitError(message)

    @classmethod
    def isPrimaryRateLimitError(cls, message: str) -> bool:
        if not message:
            return False

        message = message.lower()
        return message.startswith("api rate limit exceeded")

    @classmethod
    def isSecondaryRateLimitError(cls, message: str) -> bool:
        if not message:
            return False

        message = message.lower()
        return (
            message.startswith("you have exceeded a secondary rate limit")
            or message.endswith("please retry your request again later.")
            or message.endswith("please wait a few minutes before you try again.")
        )
