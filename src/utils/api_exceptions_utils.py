class APIException(Exception):
    """Base class for API-related exceptions."""
    def __init__(self, message, status=None, headers=None, response_data=None, history=None):
        self.message = message
        self.status = status
        self.headers = headers
        self.response_data = response_data
        self.history = history
        super().__init__(self.message)

class UnauthorizedException(APIException):
    """Exception for unauthorized access (401)."""
    pass

class BadCredentialsException(APIException):
    """Exception for bad credentials."""
    pass

class BadUserAgentException(APIException):
    """Exception for bad user agent."""
    pass

class RateLimitExceededException(APIException):
    """Exception for rate limit exceeded (429)."""
    pass

class UnknownObjectException(APIException):
    """Exception for unknown object (404)."""
    pass
