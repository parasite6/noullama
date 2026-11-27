class NoullamaError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.message = message
        self.status = status


class BadRequest(NoullamaError):
    pass


class AuthError(NoullamaError):
    pass


class Forbidden(NoullamaError):
    pass


class UnknownModel(NoullamaError):
    pass


class AmbiguousModel(NoullamaError):
    pass


class InferenceFailed(NoullamaError):
    pass


class ModelUnavailable(NoullamaError):
    pass


class NotRunning(NoullamaError):
    pass


_STATUS_ERRORS = {
    400: BadRequest,
    401: AuthError,
    403: Forbidden,
    404: UnknownModel,
    409: AmbiguousModel,
    502: InferenceFailed,
    503: ModelUnavailable,
}


def error_for(status: int, message: str) -> NoullamaError:
    return _STATUS_ERRORS.get(status, NoullamaError)(message, status)
